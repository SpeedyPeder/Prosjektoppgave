module sweSim1D

using StaticArrays

export sw_muscl_hll, sw_snapshots, animate_sw,
       default_ic_dambreak, default_ic_sine, default_source_zero,
       default_bathymetry, g

# ------------------ Physics & constants ------------------
const g = 9.81
const HMIN = 1e-12 # For dry-wet region handling, but is not necessary for the test examples here

@inline cons(h,u) = SVector(h, h*u)

@inline function sw_flux(h, m)
    u = m/h
    return SVector(m, m*u + 0.5*g*h*h)
end

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

# ------------------ Riemann solvers ----------------------
@inline function rusanov_flux(hL,uL,hR,uR)
    mL = hL*uL; mR = hR*uR
    FL = sw_flux(hL, mL)
    FR = sw_flux(hR, mR)
    cL = sqrt(g*hL); cR = sqrt(g*hR)
    smax = max(abs(uL)+cL, abs(uR)+cR)
    return 0.5*(FL + FR) - 0.5*smax*(SVector(hR,mR) - SVector(hL,mL))
end

@inline function hll_flux(hL,uL,hR,uR)
    mL = hL*uL; mR = hR*uR
    FL = sw_flux(hL, mL); FR = sw_flux(hR, mR)
    cL = sqrt(g*hL); cR = sqrt(g*hR)
    sL = min(uL - cL, uR - cR)
    sR = max(uL + cL, uR + cR)
    if sL >= 0
        return FL
    elseif sR <= 0
        return FR
    else
        return (sR*FL - sL*FR + sL*sR*(SVector(hR,mR) - SVector(hL,mL))) / (sR - sL)
    end
end


# --------------- Well-balanced bathy source (ghosted) ---------------
# Linear reconstruction of bathymetry at faces and centers
@inline function build_Btilde_faces_centers(x::AbstractVector, bfun)
    N  = length(x)
    dx = x[2] - x[1]
    xf = similar(x, N+1)
    @inbounds begin
        xf[1] = x[1] - 0.5*dx
        for i in 2:N
            xf[i] = 0.5*(x[i-1] + x[i])
        end
        xf[N+1] = x[N] + 0.5*dx
    end
    Bf = bfun(xf)                  # face values Bj+1/2
    Bc = similar(x)                
    @inbounds for i in 1:N
        Bc[i] = 0.5*(Bf[i] + Bf[i+1])  # cell values Bj = (Bf_left + Bf_right)/2
    end
    return Bf, Bc, dx
end

# KP-2007 source using precomputed face/center bathymetry (source term multiplied by dt)
function bathy_source_kp07!(η, dx, Bf, Bc)
    S = similar(η) 
    @inbounds for i in eachindex(η)
        S[i] = - g * (η[i]-Bc[i]) * (Bf[i+1] - Bf[i])/dx
    end
    return nothing
end

# ----------- Desingualirazation of velocity (KP07 eq. 2.19) -----------
@inline function u_corr_KP(mL, mR, hL, hR, eps_h)
    uL = similar(mL)
    uR = similar(mR)  
    uL  = (sqrt(2)*hL*mL)/sqrt(hL^4 + max(hL^4, eps_h))
    uR  = (sqrt(2)*hR*mR)/sqrt(hR^4 + max(hR^4, eps_h))
    return uL, uR
end

# ------------------ Flux builder (reflective BC) --------

"""
    build_fluxes_reflective!(Fhat, h, m; Bf, Bc, limiter=:mc, solver=:hll) -> amax

MUSCL reconstruction on (η, u) with Kurganov–Petrova (2007) corrections.
Depths at faces use the FACE bathymetry Bf (eq. 2.14). Reflective walls enforced.
"""
function build_fluxes_reflective!(Fhat, h, m; Bf, Bc, limiter::Symbol=:mc, solver::Symbol=:hll)
    N = length(h)
    @assert length(m) == N && length(Bc) == N && length(Bf) == N+1
    @assert size(Fhat,1) == N+1 && size(Fhat,2) == 2

    # ---- Ghosts (reflective)
    hg = similar(h, N+2);  hg[2:N+1] = h
    mg = similar(m, N+2);  mg[2:N+1] = m
    Bcg = similar(Bc, N+2); Bcg[2:N+1] = Bc
    hg[1]=hg[2]; mg[1]=-mg[2]; Bcg[1]=Bcg[2]
    hg[end]=hg[end-1]; mg[end]=-mg[end-1]; Bcg[end]=Bcg[end-1]
    # ---- Reconstruct on (ηg,(mg))
    ηg = similar(hg)
    @inbounds for j in eachindex(hg)
        ηg[j] = hg[j] + Bcg[j]
    end
    # slopes on (η,m)
    sη = similar(ηg); sm = similar(mg)
    @inbounds for j in 2:N+1
        sη[j] = slope_limited(ηg[j-1], ηg[j], ηg[j+1]; limiter=limiter)
        sm[j] = slope_limited(mg[j-1], mg[j], mg[j+1]; limiter=limiter)
    end
    sη[1]=sη[2]; sm[1]=sm[2]; sη[end]=sη[end-1]; sm[end]=sm[end-1]

    # KP07 per-cell correction (eqs. 2.15–2.16) on η
    ηL = similar(Bf)  
    ηR  = similar(Bf)  
    @inbounds for j in 1:N+1
        ηL = ηg[j] + 0.5*sη[j]  
        ηR = ηg[j+1] - 0.5*sη[j+1]   
        if ηL < Bf[j+1]           # (2.15)
            ηL = Bf[j+1]
            ηR = 2*ηg[j] - Bf[j+1]
        end
        if ηR < Bf[j]           # (2.16)
            ηR = Bf[j]
            ηL = 2*ηg[j] - Bf[j]
        end
    end

    # Corrected momenta at faces
    # desingularized velocities (2.19),
    # one-sided speeds (2.22)-(2.23), CU flux
    eps_h = dx^4  # KP recommendation
    amax = 0.0
    @inbounds for j in 1:N+1
        hL = ηL[j] - Bf[j]
        hR = ηR[j] - Bf[j+1]

        uL, uR = u_corr_KP(mL[j], mR[j], hL, hR, eps_h)
        
        #Recompute momenta/discharges
        mL[j] = uL * hL
        mR[j] = uR * hR

        #Compute aplus and aminus for CU flux
        a_plus  = max(uL + sqrt(g*hL), uR + sqrt(g*hR), 0.0)
        a_minus = min(uL - sqrt(g*hL), uR - sqrt(g*hR), 0.0)

        # Compute fluxes H[j]
        H1 = similar(mL)
        H2 = similar(mL)
        denom = a_plus - a_minus
        if denom == 0.0
            H1[j] = 0.5*(mL[j] + mR[j])
            H2[j] = 0.5 * ((mL[j]^2)/hL + 0.5*g*hL^2 + (mR[j]^2)/hR + 0.5*g*hR^2)
        else
            H1[j] = (a_plus*mL[j] - a_minus*mR[j] + a_plus*a_minus*(hR - hL)) / denom
            H2[j] = (a_plus*mL[j]^2/(hL) - a_minus*mR[j]^2/hR + a_plus*a_minus*(mR[j] - mL[j])) / denom
        end
        Fhat[j,1] = H1[j]
        Fhat[j,2] = H2[j]
        amax = max(amax, abs(a_plus), abs(a_minus))
    end
    return amax
end

# ------------------ Euler step (reflective) ----------------
@inline function euler_step_reflective!(
    hout, mout, h, m, Fhat, dt_dx, x, t,
    source_fun, Bf, Bc)
    @inbounds for i in eachindex(h)
        hout[i] = h[i] - dt_dx*(Fhat[i+1,1] - Fhat[i,1])
        mout[i] = m[i] - dt_dx*(Fhat[i+1,2] - Fhat[i,2])
    end
    source_fun(hout, mout, x, t)
    # KP07 bed source: note the correct order (m, h, dt_dx, Bf)
    bathy_source_kp07!(η, dx, Bf, Bc)
end

# ------------------ Public API -----------------------------
function default_ic_dambreak(x; hl=1.0, hr=0.0, ul=0.0, ur=0.0, xc=0.5)
    h = map(xi-> (xi < xc ? hl : hr), x)
    u = map(xi-> (xi < xc ? ul : ur), x)
    return h, u
end

function default_ic_sine(x; h0=1.0, amp=0.1, u0=0.0)
    h = @. h0 + amp*sin(2π*x)
    u = fill(u0, length(x))
    return h, u
end
default_source_zero(h, m, x, t) = nothing



#---------------- Main solver function ----------------------
"""
    sw_muscl_hll(N, L, T; CFL=0.4, limiter=:mc, solver=:hll,
                 ic_fun=default_ic_sine, source_fun=default_source_zero,
                 bfun=default_bathymetry)

Advance to time T and return (x, h, m).
"""
function sw_muscl_hll(N, L, T; CFL=0.4, limiter::Symbol=:mc, solver::Symbol=:hll,
                      ic_fun=default_ic_sine, source_fun=default_source_zero,
                      bfun=default_bathymetry)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx

    hvec, uvec = ic_fun(x)
    h = collect(hvec)
    m = similar(h); @inbounds for i in eachindex(h); m[i] = h[i]*uvec[i]; end

    Fhat = zeros(eltype(h), length(h)+1, 2)
    h1 = similar(h); m1 = similar(h)
    h2 = similar(h); m2 = similar(h)

    t = 0.0
    while t < T - eps()
        # --- Stage A: build bathymetry once and use everywhere
        Bf, Bc, _ = build_Btilde_faces_centers(x, bfun)

        amax = build_fluxes_reflective!(Fhat, h, m; Bf=Bf, Bc=Bc, limiter=limiter, solver=solver)
        dt = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        dt_dx = dt/dx

        # Euler step + KP source using SAME (Bf,Bc)
        @inbounds for i in eachindex(h)
            h1[i] = h[i] - dt_dx*(Fhat[i+1,1] - Fhat[i,1])
            m1[i] = m[i] - dt_dx*(Fhat[i+1,2] - Fhat[i,2])
        end
        source_fun(h1, m1, x, t)
        bathy_source_kp07!(h1, m1, dt_dx, Bf, Bc)

        # --- Stage B: rebuild fluxes at (h1,m1) with SAME rule
        Bf, Bc, _ = build_Btilde_faces_centers(x, bfun)
        _ = build_fluxes_reflective!(Fhat, h1, m1; Bf=Bf, Bc=Bc, limiter=limiter, solver=solver)

        # Second Euler + KP source
        @inbounds for i in eachindex(h)
            h2[i] = h1[i] - dt_dx*(Fhat[i+1,1] - Fhat[i,1])
            m2[i] = m1[i] - dt_dx*(Fhat[i+1,2] - Fhat[i,2])
        end
        source_fun(h2, m2, x, t+dt)
        bathy_source_kp07!(h2, m2, dt_dx, Bf, Bc)

        # Heun average
        @inbounds for i in eachindex(h)
            h[i] = 0.5*(h[i] + h2[i])
            m[i] = 0.5*(m[i] + m2[i])
        end
        t += dt
    end
    return x, h, m
end


# ------------------ Snapshot function ----------------------
function sw_snapshots(N, L, times; CFL=0.4, limiter::Symbol=:mc, solver::Symbol=:hll,
                      ic_fun=default_ic_sine, source_fun=default_source_zero,
                      bfun=default_bathymetry)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx

    hvec, uvec = ic_fun(x)
    h = collect(hvec)
    m = similar(h); @inbounds for i in eachindex(h); m[i] = h[i]*uvec[i]; end

    Fhat = zeros(eltype(h), length(h)+1, 2)
    h1 = similar(h); m1 = similar(h)
    h2 = similar(h); m2 = similar(h)

    times = sort(collect(times))
    results = Dict{Float64, Tuple{Vector{Float64},Vector{Float64}}}()
    t = 0.0
    if !isempty(times) && isapprox(times[1], 0.0; atol=1e-15)
        results[0.0] = (copy(h), copy(m))
        times = times[2:end]
    end

    while !isempty(times)
        target = first(times)
        while t < target - eps()
            b = bfun(x)
            Bf, Bc, _ = build_Btilde_faces_centers(x, bfun)
            amax = build_fluxes_reflective!(Fhat, h, m; Bf=Bf, Bc=Bc, limiter=limiter, solver=solver)
            dt = (amax > 0) ? min(CFL*dx/amax, target - t) : (target - t)
            dt_dx = dt/dx

            euler_step_reflective!(h1,m1, h,m,   Fhat, dt_dx, x, t,    source_fun, Bf, Bc)
            _ = build_fluxes_reflective!(Fhat, h1,m1; Bf=Bf, Bc=Bc, limiter=limiter, solver=solver)
            euler_step_reflective!(h2,m2, h1,m1, Fhat, dt_dx, x, t+dt, source_fun, Bf, Bc)
        

            @inbounds for i in eachindex(h)
                h[i] = 0.5*(h[i] + h2[i])
                m[i] = 0.5*(m[i] + m2[i])
            end
            t += dt
        end
        results[target] = (copy(h), copy(m))
        times = times[2:end]
    end
    return x, results
end

# ------------------ Animation function ----------------------
function animate_sw(N, L, T; CFL=0.4, limiter::Symbol=:mc, solver::Symbol=:hll,
                    ic_fun=default_ic_sine, source_fun=default_source_zero, bfun=default_bathymetry,
                    fps::Integer=30, ylim_h=(0.0,2.0), ylim_u=(-2.0,2.0),
                    path::AbstractString="shallow_water.gif")
    @eval using Plots
    x, h, m = sw_muscl_hll(N, L, 0.0; CFL=CFL, limiter=limiter, solver=solver,
                           ic_fun=ic_fun, source_fun=source_fun, bfun=bfun)
    b = bfun(x)

    anim = Animation()
    t = 0.0
    dx = L/N
    Fhat = zeros(eltype(h), length(h)+1, 2)
    h1 = similar(h); m1 = similar(h)
    h2 = similar(h); m2 = similar(h)

    while t < T - eps()
        # frame
        u = similar(h); @inbounds for i in eachindex(h); u[i] = m[i]/h[i] ; end
        η = h .+ b
        p1 = plot(x, η, xlabel="x", ylabel="elevation", ylim=ylim_h, label="η=h+b",
                  title="t=$(round(t,digits=3))")
        plot!(p1, x, b, label="b(x)", ls=:dash)
        p2 = plot(x, u, xlabel="x", ylabel="u", ylim=ylim_u, label=false)
        frame(anim, plot(p1, p2, layout=(2,1)))

        # step
        Bf, Bc, _ = build_Btilde_faces_centers(x, bfun)
        amax = build_fluxes_reflective!(Fhat, h, m; Bf=Bf, Bc=Bc, limiter=limiter, solver=solver)
        dt = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        dt_dx = dt/dx

        euler_step_reflective!(h1,m1, h,m,   Fhat, dt_dx, x, t,    source_fun, Bf, Bc)
        _ = build_fluxes_reflective!(Fhat, h1,m1; Bf=Bf, Bc=Bc, limiter=limiter, solver=solver)
        euler_step_reflective!(h2,m2, h1,m1, Fhat, dt_dx, x, t+dt, source_fun, Bf, Bc)

        @inbounds for i in eachindex(h)
            h[i] = 0.5*(h[i] + h2[i])
            m[i] = 0.5*(m[i] + m2[i])
        end
        t += dt
    end

    gif(anim, path, fps=fps)
    return x, h, m
end

end # module
