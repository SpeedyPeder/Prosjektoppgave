module sweSim1D

using StaticArrays

export sw_KP_upwind, sw_KP_snapshots, animate_sw_KP, kp_plot_final,
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
    Bf = bfun(xf)                  # face values Bj-1/2
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


"""
    build_fluxes_reflective!(H, η, m, dx; Bf, limiter=:mc)

Inputs:
- η: free surface elevation (cell-centered)
- m: discharge (cell-centered)
- dx: grid spacing
- Bf: bathymetry at faces (length N+1)
- limiter: slope limiter symbol (:mc or :minmod)

Outputs:
- H: (N+1)×2 array with fluxes [H₁, H₂] at each interface x_{f−1/2}
- returns amax, the maximum wave speed for CFL condition
"""
function build_fluxes_reflective!(H, η, m, dx; Bf, limiter::Symbol=:mc)
    N = length(η)
    @assert size(H,1) == N+1 && size(H,2) == 2
    @assert length(Bf) == N+1

    # ---------------- Ghost cells (reflective BCs) ----------------
    ηg = similar(η, N+2);  ηg[2:N+1] = η
    mg = similar(m,  N+2); mg[2:N+1] = m
    ηg[1] = ηg[2];          ηg[end] = ηg[end-1]
    mg[1] = -mg[2];         mg[end] = -mg[end-1]

    # ---------------- Slope reconstruction ----------------
    sη = similar(ηg); sm = similar(mg)
    @inbounds for j in 2:N+1
        sη[j] = slope_limited(ηg[j-1], ηg[j], ηg[j+1]; limiter=limiter)
        sm[j] = slope_limited(mg[j-1], mg[j], mg[j+1]; limiter=limiter)
    end
    sη[1] = sη[2];  sη[end] = sη[end-1]
    sm[1] = sm[2];  sm[end] = sm[end-1]

    eps_h = dx^4
    amax = 0.0

    # ---------------- Face loop (interfaces x_{f−1/2}) ----------------
    @inbounds for f in 1:N+1
        # Provisional MUSCL states from left (f) and right (f+1) cells
        ηL = ηg[f]   + 0.5*sη[f]
        ηR = ηg[f+1] - 0.5*sη[f+1]
        mL = mg[f]   + 0.5*sm[f]
        mR = mg[f+1] - 0.5*sm[f+1]

        # --- KP positivity correction at this face ---
        Bj = Bf[f]
        if ηL < Bj
            ηL = Bj
            ηR = 2*ηg[f] - Bj
        end
        if ηR < Bj
            ηR = Bj
            ηL = 2*ηg[f+1] - Bj
        end

        # --- Compute depths at face (h = η - B) --- 
        hL = max(ηL - Bj, 0.0) #Should be positive so max is probably not needed, but keep it for safety
        hR = max(ηR - Bj, 0.0)

        # --- Desingularize velocities (KP07 eq. 2.19) ---
        uL = (sqrt(2)*hL*mL) / sqrt(hL^4 + max(hL^4, eps_h))
        uR = (sqrt(2)*hR*mR) / sqrt(hR^4 + max(hR^4, eps_h))

        # Recompute momenta
        mL = hL * uL
        mR = hR * uR

        # --- One-sided speeds (2.22)-(2.23) ---
        a_plus  = max(uL + sqrt(g*hL), uR + sqrt(g*hR), 0.0)
        a_minus = min(uL - sqrt(g*hL), uR - sqrt(g*hR), 0.0)
        denom = a_plus - a_minus

        # --- Physical fluxes F(U,B) with U = (η, m) ---
        F1L, F1R = mL, mR
        # Uses the left and right heights and speed instead of η and Bf, but should be equivalent
        F2L = mL*uL + 0.5*g*hL^2   
        F2R = mR*uR + 0.5*g*hR^2

        # --- Central-upwind flux (eq. 2.18 in KP07) ---
        if denom == 0.0
            H1 = 0.5*(F1L + F1R)
            H2 = 0.5*(F2L + F2R)
        else
            H1 = (a_plus*F1L - a_minus*F1R + a_plus*a_minus*(ηR - ηL)) / denom
            H2 = (a_plus*F2L - a_minus*F2R + a_plus*a_minus*(mR - mL)) / denom
        end

        H[f,1] = H1
        H[f,2] = H2
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

# ---------------- KP: Final-state plotting helper ----------------
"""
    kp_plot_final(x, η, m, bfun; ylim_η=nothing, ylim_u=nothing, T=NaN)

Produces a 2-panel plot:
  top: free-surface η together with bathymetry b(x)
  bot: velocity u = m / (η - b)
"""
function kp_plot_final(x, η, m, bfun; T=NaN,
                       outdir::AbstractString=DEFAULT_OUTDIR,
                       filename::Union{Nothing,String}=nothing,
                       save::Bool=true)
    @eval using Plots
    ensure_outdir(outdir)

    b = bfun(x)               # vectorized b(x)
    h = @. η - b
    u = similar(η)
    @inbounds for i in eachindex(η)
        u[i] = (h[i] > HMIN) ? (m[i]/h[i]) : 0.0
    end

    title_top = isnan(T) ? "Shallow water (KP CU scheme)" :
                           "Shallow water (KP CU scheme), T=$(T)"

    p1 = plot(x, η, lw=2, label="η", xlabel="x", ylabel="elevation",
              title=title_top)
    plot!(p1, x, b, lw=2, ls=:dash, label="b(x)")
    p2 = plot(x, u, lw=2, label="u", xlabel="x", ylabel="u")
    fig = plot(p1, p2, layout=(2,1), size=(950,650))
    display(fig)
    if save
        default_name = isnan(T) ? "final.png" : "final_T=$(round(T,digits=3)).png"
        fname = something(filename, default_name)
        savefig(fig, joinpath(outdir, fname))
    end
    return nothing
end

# ---------------- KP: Snapshots ----------------
"""
    sw_KP_snapshots(N, L, times;
                    CFL=0.45, limiter=:mc,
                    ic_fun=default_ic_dambreak,
                    bfun=default_bathymetry,
                    makeplot=false,
                    outdir=DEFAULT_OUTDIR,
                    filename=nothing,
                    save=true)

Advance the KP solver and return `(x, snapshots)` where
`snapshots[t] => (η, m)` at each requested time `t` in `times`.

If `makeplot=true`, also produces a single figure with one subplot per
snapshot time (η and b(x) in each panel) and, if `save=true`,
saves it to `joinpath(outdir, filename_or_default)`.
"""
function sw_KP_snapshots(N, L, times;
                         CFL::Float64=0.45,
                         limiter::Symbol=:mc,
                         ic_fun=default_ic_dambreak,
                         bfun=default_bathymetry,
                         makeplot::Bool=false,
                         outdir::AbstractString=DEFAULT_OUTDIR,
                         filename::Union{Nothing,String}=nothing,
                         save::Bool=true)

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
    times_vec = sort(collect(times))
    results = Dict{Float64, Tuple{Vector{Float64},Vector{Float64}}}()
    t = 0.0
    if !isempty(times_vec) && isapprox(times_vec[1], 0.0; atol=1e-15)
        results[0.0] = (copy(η), copy(m))
        times_vec = times_vec[2:end]
    end

    while !isempty(times_vec)
        target = first(times_vec)
        while t < target - eps()
            Bf, Bc, _dx = build_Btilde_faces_centers(x, bfun)

            amax = build_fluxes_reflective!(H, η, m, dx; Bf=Bf, limiter=limiter)
            dt   = (amax > 0) ? min(0.8*dx/(2*amax), target - t) : (target - t)

            # Heun (2-stage RK2)
            euler_step!(η1, m1, η,  m,  H, dt, dx, Bf, Bc)
            _  = build_fluxes_reflective!(H, η1, m1, dx; Bf=Bf, limiter=limiter)
            euler_step!(η2, m2, η1, m1, H, dt, dx, Bf, Bc)

            @inbounds for j in eachindex(η)
                η[j] = 0.5*(η[j] + η2[j])
                m[j] = 0.5*(m[j] + m2[j])
            end
            t += dt
        end
        results[target] = (copy(η), copy(m))
        popfirst!(times_vec)
    end

    # ---------- Optional combined plot of all snapshots ----------
    if makeplot
        @eval using Plots
        ensure_outdir(outdir)

        ts = sort(collect(keys(results)))
        ns = length(ts)
        fig = plot(layout=(ns,1), size=(950,300*ns))

        b = bfun(x)

        for (k, tk) in enumerate(ts)
            ηk, mk = results[tk]
            hk = @. ηk - b
            uk = similar(ηk)
            @inbounds for i in eachindex(ηk)
                uk[i] = (hk[i] > HMIN) ? (mk[i]/hk[i]) : 0.0
            end

            # elevation + bathymetry in panel k
            plot!(fig[k], x, ηk, lw=2, label="η(x,t=$(round(tk,digits=2)))",
                  xlabel="x", ylabel="elevation",
                  title="Snapshot at t = $(round(tk,digits=3))")
            plot!(fig[k], x, b, lw=2, ls=:dash, label="b(x)")
        end

        display(fig)

        if save
            default_name = "snapshots.png"
            fname = something(filename, default_name)
            savefig(fig, joinpath(outdir, fname))
        end
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
                       fps::Integer=30,
                       path::AbstractString="shallow_water_KP.gif",   # kept for compatibility
                       outdir::AbstractString=DEFAULT_OUTDIR)
    @eval using Plots
    ensure_outdir(outdir)
    fullpath = joinpath(outdir, path)  # save into KP/

    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx

    h0, u0 = ic_fun(x)
    b  = bfun(x)
    η  = h0 .+ b
    m  = h0 .* u0

    H   = zeros(eltype(η), N+1, 2)
    η1  = similar(η);  m1 = similar(m)
    η2  = similar(η);  m2 = similar(m)

    # helpers
    make_frame = function(t)
        h = @. η - b
        u = similar(η)
        @inbounds for i in eachindex(η)
            u[i] = (h[i] > HMIN) ? (m[i]/h[i]) : 0.0
        end
        p1 = plot(x, η, lw=2, label="η", xlabel="x", ylabel="elevation",
                  title="t=$(round(t,digits=3))")
        plot!(p1, x, b, lw=2, ls=:dash, label="b(x)")
        p2 = plot(x, u, lw=2, label="u", xlabel="x", ylabel="u")
        plot(p1, p2, layout=(2,1))
    end

    anim = Animation()
    frame(anim, make_frame(0.0))   # ensure non-empty

    t = 0.0
    while t < T - eps()
        Bf, Bc, _dx = build_Btilde_faces_centers(x, bfun)
        amax = build_fluxes_reflective!(H, η, m, dx; Bf=Bf, limiter=limiter)
        cmax = (isfinite(amax) && amax > 0) ? amax : 0.0
        dt  = (cmax > 0) ? min(0.8*dx/(2*cmax), T - t) : (T - t)
        dt  = max(dt, 0.0)

        euler_step!(η1, m1, η,  m,  H, dt, dx, Bf, Bc)
        _  = build_fluxes_reflective!(H, η1, m1, dx; Bf=Bf, limiter=limiter)
        euler_step!(η2, m2, η1, m1, H, dt, dx, Bf, Bc)

        @inbounds for j in eachindex(η)
            η[j] = 0.5*(η[j] + η2[j])
            m[j] = 0.5*(m[j] + m2[j])
        end
        t += dt

        frame(anim, make_frame(t))
    end

    # safety: ensure at least one frame
    if length(anim.frames) == 0
        frame(anim, make_frame(0.0))
    elseif t < T - 1e-12
        frame(anim, make_frame(T))
    end

    gif(anim, fullpath, fps=fps)
    return x, η, m
end


end # module