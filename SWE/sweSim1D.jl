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

# KP-2007 source using precomputed face/center bathymetry (
function bathy_source_rate_KP07_eta!(S2, η, Bf, Bc, dx)
    @assert length(S2) == length(η) == length(Bc)
    @assert length(Bf) == length(η) + 1
    @inbounds for j in eachindex(η)
        S2[j] = -g * (η[j] - Bc[j]) * (Bf[j+1] - Bf[j]) / dx
    end
    return nothing
end

# ----------- Desingualirazation of velocity (KP07 eq. 2.19) -----------
@inline function u_corr_KP(mL, mR, hL, hR, eps_h)
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
function build_fluxes_reflective!(Fhat, η, m, dx; Bf, Bc, limiter::Symbol=:mc)
    N = length(η)
    @assert length(m) == N && length(Bc) == N && length(Bf) == N+1
    @assert size(Fhat,1) == N+1 && size(Fhat,2) == 2

    # ---- Ghosts (reflective)
    ηg = similar(η, N+2);  ηg[2:N+1] = η
    mg = similar(m, N+2);  mg[2:N+1] = m
    Bcg = similar(Bc, N+2); Bcg[2:N+1] = Bc
    ηg[1]= ηg[2];      mg[1]=-mg[2];       Bcg[1]=Bcg[2]
    ηg[end]=ηg[end-1]; mg[end]=-mg[end-1]; Bcg[end]=Bcg[end-1]

    hg = similar(ηg)
    # ---- Reconstruct on (ηg,(mg))
    @inbounds for j in eachindex(hg)
        hg[j] = ηg[j] - Bcg[j]
    end
    # slopes on (η,m)
    sη = similar(ηg); sm = similar(mg)
    @inbounds for j in 2:N+1
        #Limited slope coefficients for η and m
        sη[j] = slope_limited(ηg[j-1], ηg[j], ηg[j+1]; limiter=limiter)
        sm[j] = slope_limited(mg[j-1], mg[j], mg[j+1]; limiter=limiter)
    end
    sη[1]=sη[2]; sm[1]=sm[2]; sη[end]=sη[end-1]; sm[end]=sm[end-1]

    mL = similar(Bf)  
    mR  = similar(Bf) 
    ηL = similar(Bf)  
    ηR  = similar(Bf)   
    @inbounds for j in 1:N+1
        # Reconstructed states at faces j+1/2
        ηL[j] = ηg[j] + 0.5*sη[j]  
        ηR[j] = ηg[j+1] - 0.5*sη[j+1] 
        mL[j] = mg[j] + 0.5*sm[j]
        mR[j] = mg[j+1] - 0.5*sm[j+1] 
        # KP07 per-cell correction (eqs. 2.15–2.16) on η 
        if ηL[j] < Bf[j+1]           # (2.15)
            ηL[j] = Bf[j+1]
            ηR[j] = 2*ηg[j] - Bf[j+1]
        end
        if ηR[j] < Bf[j]           # (2.16)
            ηL[j] = 2*ηg[j] - Bf[j]
            ηR[j] = Bf[j]
        end
    end
    # Desingularized velocities (2.19),
    # one-sided speeds (2.22)-(2.23), CU flux
    eps_h = dx^4  # KP recommendation
    amax = 0.0
    @inbounds for j in 1:N+2
        hL = ηL[j] - Bf[j]
        hR = ηR[j] - Bf[j]

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
        #Have tried to replace (hu)^2/h with hu*u here, it should be equivalent, 
        # as momentum is reconstructed from h and corrected u at interfaces
        if denom == 0.0
            H1[j] = 0.5*(mL[j] + mR[j])
            H2[j] = 0.5 * (mL[j]*uL + 0.5*g*hL^2 + (mR[j]*uR + 0.5*g*hR^2))
        else
            H1[j] = (a_plus*mL[j] - a_minus*mR[j] + a_plus*a_minus*(ηR[j] - ηL[j])) / denom
            H2[j] = (a_plus*mL[j]*hL- a_minus*mR[j]*hR + a_plus*a_minus*(mR[j] - mL[j])) / denom
        end
        Fhat[j,1] = H1[j]
        Fhat[j,2] = H2[j]
        amax = max(amax, abs(a_plus), abs(a_minus))
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
    bathy_source_rate_KP07_eta!(S2, η, Bf, Bc, dx)
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
        amax = build_fluxes_reflective!(H, η, m,dx; Bf=Bf, Bc=Bc, limiter=limiter)
        # KP07 CFL condition with safety factor 0.8
        dt = 0.8*dx/(2*amax)  
        euler_step!(η1, m1, η, m, H, dt, dx, Bf, Bc)
        # Use computed η1,m1 to rebuild fluxes
        _ = build_fluxes_reflective!(H, η1, m1,dx; Bf=Bf, Bc=Bc, limiter=limiter)
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

        amax = build_fluxes_reflective!(Fhat, h, m, dx; Bf=Bf, Bc=Bc, limiter=limiter)
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
        _ = build_fluxes_reflective!(Fhat, h1, m1, dx; Bf=Bf, Bc=Bc, limiter=limiter)

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
            amax = build_fluxes_reflective!(Fhat, h, m, dx; Bf=Bf, Bc=Bc, limiter=limiter)
            dt = (amax > 0) ? min(CFL*dx/amax, target - t) : (target - t)
            dt_dx = dt/dx

            euler_step_reflective!(h1,m1, h,m,   Fhat, dt_dx, x, t,    source_fun, Bf, Bc)
            _ = build_fluxes_reflective!(Fhat, h1, m1, dx; Bf=Bf, Bc=Bc, limiter=limiter)
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
    x, h, m = sw_muscl_hll(N, L, 0.0; CFL=CFL, limiter=limiter,
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
        amax = build_fluxes_reflective!(Fhat, h, m, dx; Bf=Bf, Bc=Bc, limiter=limiter)
        dt = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        dt_dx = dt/dx

        euler_step_reflective!(h1,m1, h,m,   Fhat, dt_dx, x, t,    source_fun, Bf, Bc)
        _ = build_fluxes_reflective!(Fhat, h1, m1, dx; Bf=Bf, Bc=Bc, limiter=limiter)
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

#----------- Adding plotting functions for KP-scheme --------------
# ---------------- KP: Final-state plotting helper ----------------
"""
    kp_plot_final(x, η, m, bfun; ylim_η=nothing, ylim_u=nothing, T=NaN)

Produces a 2-panel plot:
  top: free-surface η together with bathymetry b(x)
  bot: velocity u = m / (η - b)
"""
function kp_plot_final(x, η, m, bfun; ylim_η=nothing, ylim_u=nothing, T=NaN)
    @eval using Plots
    b = bfun(x)
    h = @. η - b
    u = similar(η)
    @inbounds for i in eachindex(η)
        u[i] = (h[i] > HMIN) ? (m[i]/h[i]) : 0.0
    end
    title_top = isnan(T) ? "Shallow water (KP CU scheme)" :
                           "Shallow water (KP CU scheme), T=$(T)"                      
    p1 = plot(x, η, lw=2, label="η", xlabel="x", ylabel="elevation",
              title=title_top, ylim=ylim_η)
    plot!(p1, x, b, lw=2, ls=:dash, label="b(x)")
    p2 = plot(x, u, lw=2, label="u", xlabel="x", ylabel="u", ylim=ylim_u)
    display(plot(p1, p2, layout=(2,1), size=(950,650)))
    return nothing
end

# ---------------- KP: Snapshots ----------------
"""
    sw_KP_snapshots(N, L, times; CFL=0.45, limiter=:mc, ic_fun=default_ic_dambreak, bfun=default_bathymetry)

Advance the KP solver and return (x, snapshots) where
`snapshots[t] => (η, m)` at each requested time t in `times`.
"""
function sw_KP_snapshots(N, L, times; CFL::Float64=0.45, limiter::Symbol=:mc,
                         ic_fun=default_ic_dambreak, bfun=default_bathymetry)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx

    # IC in (η, m)
    h0, u0 = ic_fun(x)
    b0     = bfun(x)
    η  = h0 .+ b0
    m  = h0 .* u0

    H   = zeros(eltype(η), N+1, 2)
    η1  = similar(η);  m1 = similar(m)
    η2  = similar(η);  m2 = similar(m)

    # sort & prepare output
    times = sort(collect(times))
    results = Dict{Float64, Tuple{Vector{Float64},Vector{Float64}}}()
    t = 0.0
    if !isempty(times) && isapprox(times[1], 0.0; atol=1e-15)
        results[0.0] = (copy(η), copy(m))
        times = times[2:end]
    end

    while !isempty(times)
        target = first(times)
        while t < target - eps()
            Bf, Bc, _dx = build_Btilde_faces_centers(x, bfun)
            h = @. η - Bc
            amax = build_fluxes_reflective!(H, h, m; Bf=Bf, Bc=Bc, limiter=limiter)
            # KP-like CFL (use your preferred factor)
            dt = (amax > 0) ? min(0.8*dx/(2*amax), target - t) : (target - t)

            # Heun (2-stage RK2)
            euler_step!(η1, m1, η,  m,  H, dt, dx, Bf, Bc)
            h1 = @. η1 - Bc
            _  = build_fluxes_reflective!(H, h1, m1; Bf=Bf, Bc=Bc, limiter=limiter)
            euler_step!(η2, m2, η1, m1, H, dt, dx, Bf, Bc)

            @inbounds for j in eachindex(η)
                η[j] = 0.5*(η[j] + η2[j])
                m[j] = 0.5*(m[j] + m2[j])
            end
            t += dt
        end
        results[target] = (copy(η), copy(m))
        popfirst!(times)
    end
    return x, results
end

# ---------------- KP: Animation ----------------
"""
    animate_sw_KP(N, L, T; CFL=0.45, limiter=:mc, ic_fun=default_ic_dambreak,
                  bfun=default_bathymetry, fps=30, ylim_η=(nothing,nothing),
                  ylim_u=(nothing,nothing), path="shallow_water_KP.gif")

Run the KP solver while writing frames each sub-step.
Top panel: η with b(x); bottom: velocity u.
"""
function animate_sw_KP(N, L, T; CFL::Float64=0.45, limiter::Symbol=:mc,
                       ic_fun=default_ic_dambreak, bfun=default_bathymetry,
                       fps::Integer=30, ylim_η=(nothing,nothing), ylim_u=(nothing,nothing),
                       path::AbstractString="shallow_water_KP.gif")
    @eval using Plots

    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx

    h0, u0 = ic_fun(x)
    b  = bfun(x)
    η  = h0 .+ b
    m  = h0 .* u0

    H   = zeros(eltype(η), N+1, 2)
    η1  = similar(η);  m1 = similar(m)
    η2  = similar(η);  m2 = similar(m)

    anim = Animation()
    t = 0.0
    while t < T - eps()
        # frame
        h = @. η - b
        u = similar(η)
        @inbounds for i in eachindex(η)
            u[i] = (h[i] > HMIN) ? (m[i]/h[i]) : 0.0
        end
        p1 = plot(x, η, lw=2, label="η", xlabel="x", ylabel="elevation",
                  title="t=$(round(t,digits=3))", ylim=ylim_η)
        plot!(p1, x, b, lw=2, ls=:dash, label="b(x)")
        p2 = plot(x, u, lw=2, label="u", xlabel="x", ylabel="u", ylim=ylim_u)
        frame(anim, plot(p1, p2, layout=(2,1)))

        # step
        Bf, Bc, _dx = build_Btilde_faces_centers(x, bfun)
        h = @. η - Bc
        amax = build_fluxes_reflective!(H, h, m; Bf=Bf, Bc=Bc, limiter=limiter)
        dt  = (amax > 0) ? min(0.8*dx/(2*amax), T - t) : (T - t)

        euler_step!(η1, m1, η,  m,  H, dt, dx, Bf, Bc)
        h1 = @. η1 - Bc
        _  = build_fluxes_reflective!(H, h1, m1; Bf=Bf, Bc=Bc, limiter=limiter)
        euler_step!(η2, m2, η1, m1, H, dt, dx, Bf, Bc)

        @inbounds for j in eachindex(η)
            η[j] = 0.5*(η[j] + η2[j])
            m[j] = 0.5*(m[j] + m2[j])
        end
        t += dt
    end

    gif(anim, path, fps=fps)
    return x, η, m
end

end # module