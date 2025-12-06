module KPsweSim2D

using StaticArrays

export sw_KP_upwind, sw_KP_snapshots, animate_sw_KP, kp_plot_final,
       default_ic_dambreak, default_ic_sine, default_source_zero,
       default_bathymetry, g

# ------------------ Physics & constants ------------------
const g = 9.81
const HMIN = 1e-12 # For dry-wet region handling, but is not necessary for the test examples here


default_bathymetry(x) = zeros(length(x))

# ------------------ Limiters & helpers -------------------
@inline minmod2(a, b) = (a*b <= 0) ? 0.0 : sign(a)*min(abs(a), abs(b))
@inline function minmod3(a, b, c)
    s = sign(a)
    (sign(b) != s || sign(c) != s) && return 0.0
    return s * min(abs(a), abs(b), abs(c))
end

@inline function slope_limited(um, u0, up; limiter::Symbol=:mc)
    a = u0 - um
    b = up - u0
    limiter === :minmod && return minmod2(a,b)
    limiter === :mc     && return minmod3(0.5*(a+b), 2a, 2b)
    error("Unknown limiter $limiter")
end


function fill_reflective_ghosts_2D!(ηg, mg, qg, η, m, q)
    Nx, Ny = size(η)
    @assert size(m)  == (Nx,Ny) == size(q)
    @assert size(ηg) == (Nx+2,Ny+2) == size(mg) == size(qg)
    @inbounds begin
        # 1) copy interior to ghost arrays
        for j in 1:Nx, k in 1:Ny
            ηg[j+1,k+1] = η[j,k]
            mg[j+1,k+1] = m[j,k]
            qg[j+1,k+1] = q[j,k]
        end

        # 2) left and right vertical walls (x = const)
        for k in 2:Ny+1
            # left ghost column i = 1  (mirror of i = 2)
            ηg[1,k] = ηg[2,k]
            mg[1,k] = -mg[2,k]      # flip normal (x) momentum
            qg[1,k] =  qg[2,k]

            # right ghost column i = Nx+2 (mirror of Nx+1)
            ηg[Nx+2,k] = ηg[Nx+1,k]
            mg[Nx+2,k] = -mg[Nx+1,k]
            qg[Nx+2,k] =  qg[Nx+1,k]
        end

        # 3) bottom and top horizontal walls (y = const)
        for i in 2:Nx+1
            # bottom ghost row k = 1 (mirror of k = 2)
            ηg[i,1] = ηg[i,2]
            mg[i,1] = mg[i,2]
            qg[i,1] = -qg[i,2]      # flip normal (y) momentum

            # top ghost row k = Ny+2 (mirror of Ny+1)
            ηg[i,Ny+2] = ηg[i,Ny+1]
            mg[i,Ny+2] = mg[i,Ny+1]
            qg[i,Ny+2] = -qg[i,Ny+1]
        end

        # 4) corners applied last
        ηg[1,1]         = ηg[2,2]
        mg[1,1]         = -mg[2,2]
        qg[1,1]         = -qg[2,2]

        ηg[Nx+2,1]      = ηg[Nx+1,2]
        mg[Nx+2,1]      = -mg[Nx+1,2]
        qg[Nx+2,1]      = -qg[Nx+1,2]

        ηg[1,Ny+2]      = ηg[2,Ny+1]
        mg[1,Ny+2]      = -mg[2,Ny+1]
        qg[1,Ny+2]      = -qg[2,Ny+1]

        ηg[Nx+2,Ny+2]   = ηg[Nx+1,Ny+1]
        mg[Nx+2,Ny+2]   = -mg[Nx+1,Ny+1]
        qg[Nx+2,Ny+2]   = -qg[Nx+1,Ny+1]
    end

    return nothing
end


#Takes in ghosted array for slope calculation
function slopes_2D!(σx_η, σx_m, σx_q,
                    σy_η, σy_m, σy_q,
                    ηg, mg, qg; limiter::Symbol=:mc)

    Nx+2, Ny+2 = size(ηg)
    @assert size(mg) == (Nx+2,Ny+2) == size(q)g

    @inbounds for j in 2:Nx+1, k in 2:Ny+1
        # x-direction slopes
        σx_η[j,k] = slope_limited(ηg[j-1,k], ηg[j,k], ηg[j+1,k]; limiter=limiter)
        σx_m[j,k] = slope_limited(mg[j-1,k], mg[j,k], mg[j+1,k]; limiter=limiter)
        σx_q[j,k] = slope_limited(qg[j-1,k], qg[j,k], qg[j+1,k]; limiter=limiter)

        # y-direction slopes
        σy_η[j,k] = slope_limited(ηg[j,k-1], ηg[j,k], ηg[j,k+1]; limiter=limiter)
        σy_m[j,k] = slope_limited(mg[j,k-1], mg[j,k], mg[j,k+1]; limiter=limiter)
        σy_q[j,k] = slope_limited(qg[j,k-1], qg[j,k], qg[j,k+1]; limiter=limiter)
    end
    return nothing
end

#Flux - helpers for 2D
#NB! For them to be consistent with the scheme we 
@inline function F_phys(η, m, q, B)
    h = max(η - B, 0.0)
    u = (h > 0) ? m / h : 0.0
    v = (h > 0) ? q / h : 0.0
    return (m, m*u + 0.5*g*h^2,m*v)
end

@inline function G_phys(η, m, q, B)
    h = max(η - B, 0.0)
    u = (h > 0) ? m / h : 0.0
    v = (h > 0) ? q / h : 0.0
    return (q, q*u, q*v + 0.5*g*h^2)
end


# --------------- Well-balanced bathy source (ghosted) ---------------
# Build bathymetry at cell centers and face midpoints for KP07 scheme
function build_Btilde_KP07(x, y, bfun)
    Nx, Ny = length(x), length(y)
    dx = x[2]-x[1]
    dy = y[2]-y[1]

    #1) build bathymetry at corners
    xC = range(x[1]-dx/2, x[end]+dx/2, length=Nx+1)
    yC = range(y[1]-dy/2, y[end]+dy/2, length=Ny+1)
    Bcorner = [bfun(xC[j], yC[k]) for j in 1:Nx+1, k in 1:Ny+1 ]
    #2) build bathymetry at face midpoints
    Bfx = Array{Float64}(undef, Nx+1, Ny)
    for j in 1:Nx+1, k in 1:Ny
        Bfx[j,k] = 0.5*(Bcorner[j,k+1] + Bcorner[j,k]) #uses average at corners eq (3.13) (for j-1/2)
    end
    Bfy = Array{Float64}(undef, Nx, Ny+1)
    for j in 1:Nx, k in 1:Ny+1
        Bfy[j,k] = 0.5*(Bcorner[j+1,k] + Bcorner[j,k]) #uses average at corners eq (3.14) (for(k-1/2))
    end
    #3) build bathymetry at cell centers
    Bc = Array{Float64}(undef, Nx, Ny)
    for j in 1:Nx, k in 1:Ny
        Bc[j,k] = 0.25*(Bfx[j,k] + Bfx[j+1,k] + Bfy[j,k] + Bfy[j,k+1]) # eq (3.12) with (j,k) → (j-1/2,k-1/2)
    end
    return Bc, Bfx, Bfy, dx, dy
end

"""
    correct_eta_slopes_KP07!(σx_η, σy_η, η, Bfx, Bfy)
Apply the KP07 positivity correction (eqs. 3.20–3.23)
to the w = η component only.
"""
function correct_eta_slopes_KP07!(σx_η, σy_η, η, Bfx, Bfy)
    Nx, Ny = size(η)
    @assert size(σx_η) == (Nx,Ny)
    @assert size(σy_η) == (Nx,Ny)
    @assert size(Bfx)  == (Nx+1,Ny)
    @assert size(Bfy)  == (Nx,Ny+1)

    @inbounds for j in 1:Nx, k in 1:Ny
        w̄ = η[j,k]

        # ----- x-direction corrections (3.20)–(3.21) -----
        if j < Nx
            wE = w̄ + 0.5*σx_η[j,k]
            Bjp  = Bfx[j+1,k]          # B_{j+1/2,k}
            if wE < Bjp
                σx_η[j,k] = 2*(Bjp - w̄) #corrected slope which updates wE
            end
        end
        if j > 1
            wW = w̄ - 0.5*σx_η[j,k]
            Bjm = Bfx[j,k]             # B_{j-1/2,k}
            if wW < Bjm
                σx_η[j,k] = 2*(w̄ - Bjm)
            end
        end

        # ----- y-direction corrections (3.22)–(3.23) -----
        if k < Ny
            wN = w̄ + 0.5*σy_η[j,k]
            Bkp = Bfy[j,k+1]           # B_{j,k+1/2}
            if wN < Bkp
                σy_η[j,k] = 2*(Bkp - w̄)
            end
        end
        if k > 1
            wS = w̄ - 0.5*σy_η[j,k]
            Bkm = Bfy[j,k]             # B_{j,k-1/2}
            if wS < Bkm
                σy_η[j,k] = 2*(w̄ - Bkm)
            end
        end
    end
    return nothing
end


# KP-2007 source using precomputed face/center bathymetry (
function bathy_source_rate_KP07_eta!(S2, S3, η, Bc, Bfx, Bfy, dx,dy)
    Nx, Ny = size(η)
    @assert size(Bc)  == (Nx, Ny)
    @assert size(Bfx) == (Nx+1, Ny)   # vertical faces j-1/2,k
    @assert size(Bfy) == (Nx, Ny+1)   # horizontal faces j,k-1/2
    @assert size(S2)  == (Nx, Ny)
    @assert size(S3)  == (Nx, Ny)
    @inbounds for j in 1:Nx, k in 1:Ny
        S2[j,k] = -g * (η[j,k] - Bc[j,k]) * (Bfx[j+1,k] - Bfx[j,k]) / dx
        S3[j,k] = -g * (η[j,k] - Bc[j,k]) * (Bfy[j,k+1] - Bfy[j,k]) / dy  #eq (3.16)
    end
    return nothing
end

"""
    build_fluxes_KP07_2D!(Hx, Hy, η, m, q, dx, dy, Bc, Bfx, Bfy; limiter=:mc)

Build x- and y-direction numerical fluxes for the KP07 scheme on a rectangular grid.

- U = (w, hu, hv) = (η, m, q)
- Hx[j,k,:] ≈ H^x_{j-1/2,k},  size (Nx+1, Ny, 3)
- Hy[j,k,:] ≈ H^y_{j,k-1/2},  size (Nx,   Ny+1, 3)

`Bfx[j,k]` = B_{j-1/2,k},  `Bfy[j,k]` = B_{j,k-1/2}.
"""
function build_fluxes_KP07_2D!(Hx, Hy, η, m, q, dx, dy, Bc, Bfx, Bfy;
                               limiter::Symbol = :mc)

    Nx, Ny = size(η)
    @assert size(m)   == (Nx,Ny) == size(q)
    @assert size(Bc)  == (Nx,Ny)
    @assert size(Bfx) == (Nx+1,Ny)      # vertical faces j-1/2
    @assert size(Bfy) == (Nx,Ny+1)      # horizontal faces k-1/2
    @assert size(Hx)  == (Nx+1,Ny,3)
    @assert size(Hy)  == (Nx,Ny+1,3)

    #Make ghost arrays for BC handling
    ηg = similar(η, Nx+2, Ny+2)
    mg = similar(m,  Nx+2, Ny+2)
    qg = similar(q,  Nx+2, Ny+2)
    fill_reflective_ghosts_2D!(ηg, mg, qg, η, m, q)

    # ----- limited slopes for each component (cell centred) -----
    σx_η = similar(η); σx_m = similar(m); σx_q = similar(q)
    σy_η = similar(η); σy_m = similar(m); σy_q = similar(q)
    slopes_2D!(σx_η, σx_m, σx_q,σy_η, σy_m, σy_q,η, m, q; limiter=limiter)
    #Correction if slopes produce negative water heights
    correct_eta_slopes_KP07!(σx_η, σy_η, η, Bfx, Bfy)
    # Use similar slopes at edges
    σx_η

    #Small velocity desingularisation parameter
    eps_h = min(dx,dy)^4
    amax  = 0.0

    # ======================================================
    # X–direction faces: Hx[j,k,:]  ≈ flux at x_{j-1/2}, k
    #   j = 1      : left boundary face
    #   j = 2..Nx  : between cell (j-1,k) [left] and (j,k) [right]
    #   j = Nx+1   : right boundary face
    # ======================================================
    @inbounds for j in 1:Nx+1, k in 1:Ny+1

        # ---------- left/right state at this face ----------
        ηL = ηg[j,k] + 0.5*σx_η[j,k] # East from cell j-1 (consistent with bathymetry indexing)
        mL = mg[j,k] + 0.5*σx_m[j,k]
        qL = qg[j,k] + 0.5*σx_q[j,k]

        ηR = ηg[j+1,k] - 0.5*σx_η[j+1,k] #West from cell j+1
        mR = mg[j+1,k] - 0.5*σx_m[j+1,k]
        qR = qg[j+1,k] - 0.5*σx_q[j+1,k]
    end
        # Set boundaries in x-direction
        if j == 1
            # left boundary: ghost (mirror of cell 1 in x)
            ηL = η[1,k]
            mL = -m[1,k]
            qL =  q[1,k]

        # bathymetry at this vertical face
        Bj = Bfx[j,k]

        # depths
        hL = max(ηL - Bj, 0.0)
        hR = max(ηR - Bj, 0.0)

        # normal velocities in x (KP07 desingularisation)
        uL = (sqrt(2)*hL*mL) / sqrt(hL^4 + max(hL^4, eps_h))
        uR = (sqrt(2)*hR*mR) / sqrt(hR^4 + max(hR^4, eps_h))
        mL = hL*uL;  mR = hR*uR

        # one-sided speeds
        a_plus  = max(uL + sqrt(g*hL), uR + sqrt(g*hR), 0.0)
        a_minus = min(uL - sqrt(g*hL), uR - sqrt(g*hR), 0.0)
        denom   = a_plus - a_minus

        # physical fluxes F(U,B) in x-direction
        F1L, F2L, F3L = F_phys(ηL, mL, qL, Bj)
        F1R, F2R, F3R = F_phys(ηR, mR, qR, Bj)

        if denom == 0.0
            Hx[j,k,1] = 0.5*(F1L + F1R)
            Hx[j,k,2] = 0.5*(F2L + F2R)
            Hx[j,k,3] = 0.5*(F3L + F3R)
        else
            Hx[j,k,1] = (a_plus*F1L - a_minus*F1R + a_plus*a_minus*(ηR - ηL)) / denom
            Hx[j,k,2] = (a_plus*F2L - a_minus*F2R + a_plus*a_minus*(mR - mL)) / denom
            Hx[j,k,3] = (a_plus*F3L - a_minus*F3R + a_plus*a_minus*(qR - qL)) / denom
        end

        amax = max(amax, abs(a_plus), abs(a_minus))
    end

    # ======================================================
    # Y–direction faces: Hy[j,k,:]  ≈ flux at y_{k-1/2}, j
    #   k = 1      : bottom boundary face
    #   k = 2..Ny  : between cell (j,k-1) [bottom] and (j,k) [top]
    #   k = Ny+1   : top boundary face
    # ======================================================
    @inbounds for j in 1:Nx, k in 1:Ny+1

        # ---------- bottom state at this face ----------
        if k == 1
            # bottom boundary: ghost (mirror of cell 1 in y)
            ηB = η[j,1]
            mB =  m[j,1]
            qB = -q[j,1]
        elseif k == Ny+1
            # last interior cell is k-1 = Ny
            ηB = η[j,Ny] + 0.5*σy_η[j,Ny]
            mB = m[j,Ny] + 0.5*σy_m[j,Ny]
            qB = q[j,Ny] + 0.5*σy_q[j,Ny]
        else
            kb = k-1
            ηB = η[j,kb] + 0.5*σy_η[j,kb]
            mB = m[j,kb] + 0.5*σy_m[j,kb]
            qB = q[j,kb] + 0.5*σy_q[j,kb]
        end

        # ---------- top state at this face ----------
        if k == 1
            # top from first interior cell
            ηT = η[j,1] - 0.5*σy_η[j,1]
            mT = m[j,1] - 0.5*σy_m[j,1]
            qT = q[j,1] - 0.5*σy_q[j,1]
        elseif k == Ny+1
            # top boundary: ghost (mirror of cell Ny)
            ηT = η[j,Ny]
            mT =  m[j,Ny]
            qT = -q[j,Ny]
        else
            kt = k
            ηT = η[j,kt] - 0.5*σy_η[j,kt]
            mT = m[j,kt] - 0.5*σy_m[j,kt]
            qT = q[j,kt] - 0.5*σy_q[j,kt]
        end

        Bj = Bfy[j,k]

        hB = max(ηB - Bj, 0.0)
        hT = max(ηT - Bj, 0.0)

        # normal velocities in y (KP07)
        vB = (sqrt(2)*hB*qB) / sqrt(hB^4 + max(hB^4, eps_h))
        vT = (sqrt(2)*hT*qT) / sqrt(hT^4 + max(hT^4, eps_h))
        qB = hB*vB; qT = hT*vT

        b_plus  = max(vB + sqrt(g*hB), vT + sqrt(g*hT), 0.0)
        b_minus = min(vB - sqrt(g*hB), vT - sqrt(g*hT), 0.0)
        denom   = b_plus - b_minus

        G1B, G2B, G3B = G_phys(ηB, mB, qB, Bj)
        G1T, G2T, G3T = G_phys(ηT, mT, qT, Bj)

        if denom == 0.0
            Hy[j,k,1] = 0.5*(G1B + G1T)
            Hy[j,k,2] = 0.5*(G2B + G2T)
            Hy[j,k,3] = 0.5*(G3B + G3T)
        else
            Hy[j,k,1] = (b_plus*G1B - b_minus*G1T + b_plus*b_minus*(ηT - ηB)) / denom
            Hy[j,k,2] = (b_plus*G2B - b_minus*G2T + b_plus*b_minus*(mT - mB)) / denom
            Hy[j,k,3] = (b_plus*G3B - b_minus*G3T + b_plus*b_minus*(qT - qB)) / denom
        end

        amax = max(amax, abs(b_plus), abs(b_minus))
    end

    return amax
end


# ------------------ Euler step (reflective) ----------------
# Does the iterative Euler step with bathymetry source term
# η^{n+1}_j = η^n_j - (dt/Δx)(H1_{j+1/2}-H1_{j-1/2})
# m^{n+1}_j =  m^n_j - (dt/Δx)(H2_{j+1/2}-H2_{j-1/2}) + dt*S2_j
@inline function euler_step!(ηout, mout, η, m, Fhat, dt, dx, Bf, Bc)
    N = length(η)
    @assert size(Fhat,1) == N+1 && size(Fhat,2) == 2
    # momentum source from bathymetry
    S2 = similar(η)
    bathy_source_rate_KP07_eta!(S2, S3, η, Bf, Bc, dx)
    λ= dt/dx
    @inbounds for j in 1:N
        ηout[j] = η[j] - λ*(Fhat[j+1,1] - Fhat[j,1])
        mout[j] = m[j] - λ*(Fhat[j+1,2] - Fhat[j,2]) + dt*S2[j]
    end
    return nothing
end

#---------------- Main solver function ----------------------
function sw_KP_upwind(N, L, T; CFL::Float64 = 0.45, limiter::Symbol = :mc, ic_fun = default_ic_dambreak,
    bfun = default_bathymetry)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    #Initial condition: Convert (h,u) — to (η,m)
    h0, u0 = ic_fun(x)
    b0     = bfun(x)
    η  = h0 .+ b0
    m  = h0 .* u0
    #Create arrays
    H   = zeros(eltype(η), N+1, 2)  # face fluxes (H1,H2)
    η1  = similar(η);  m1 = similar(m)
    η2  = similar(η);  m2 = similar(m)

    t = 0.0
    while t < T - eps()
        Bf, Bc, dx = build_Btilde_faces_centers(x, bfun)  
        amax = build_fluxes_reflective!(H, η, m,dx; Bf=Bf, limiter=limiter)
        # KP07 CFL condition with safety factor 0.1
        dt = 0.1*dx/(2*amax)  
        euler_step!(η1, m1, η, m, H, dt, dx, Bf, Bc)
        # Use computed η1,m1 to rebuild fluxes
        _ = build_fluxes_reflective!(H, η1, m1,dx; Bf=Bf, limiter=limiter)
        euler_step!(η2, m2, η1, m1, H, dt, dx, Bf, Bc)
        # Heun average
        @inbounds for j in eachindex(η)
            η[j] = 0.5*(η[j] + η2[j])
            m[j] = 0.5*(m[j] + m2[j])
        end
        t += dt
    end
    return x, η, m
end


#----------- Adding plotting functions for KP-scheme --------------

# ---------------- Output directory  for plots----------------
const DEFAULT_OUTDIR = "Plots_Bathymetry"
ensure_outdir(outdir::AbstractString=DEFAULT_OUTDIR) = (isdir(outdir) || mkpath(outdir); outdir)

end #module KPsweSim2D