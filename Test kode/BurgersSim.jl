module BurgersSim

using Plots

# --- Physics ---
flux(u) = 0.5u^2
initial_condition(x) = sin(2π*x)

function initial_condition_tophat(x; a=0.0, b=0.1, u_in=1.0, u_out=0.0, L=1.0)
    xa = mod1(x - a, L)
    xb = mod1(b - a, L)   # interval length in [0,L)
    return (xa < xb) ? u_in : u_out
end

# Optional Riemann IC and exact shock
mod1(y, L) = y - L*floor(y/L)


function initial_condition_riemann(x; uL=1.0, uR=0.0, x0=0.5, L=1.0)
    y = mod1(x - x0, L)
    return (y > 0) ? uR : uL
end

function exact_burgers_shock(x, t; uL=1.0, uR=0.0, x0=0.5, L=1.0)
    s  = 0.5*(uL + uR)
    y  = mod1(x - x0, L)
    ys = mod1(s*t, L)
    return (y < ys) ? uR : uL
end

# --- Limiters ---
@inline function minmod2(a, b)
    (a*b <= 0) && return 0.0
    s = sign(a)
    return s * min(abs(a), abs(b))
end

@inline function minmod3(a, b, c)
    s = sign(a)
    (sign(b) != s || sign(c) != s) && return 0.0
    return s * min(abs(a), abs(b), abs(c))
end

@inline function slope_limited(um, u0, up; limiter::Symbol = :mc)
    a = u0 - um
    b = up - u0
    if limiter === :minmod
        return minmod2(a, b)
    elseif limiter === :mc
        return minmod3(0.5*(a+b), 2a, 2b)
    else
        error("Unknown limiter $limiter. Use :mc or :minmod.")
    end
end

# --- Godunov flux (exact for Burgers) ---
@inline function godunov_flux_burgers(uL, uR)
    if uL > uR
        s = 0.5*(uL + uR)             # shock speed
        return s > 0 ? flux(uL) : flux(uR)
    else
        if uL >= 0
            return flux(uL)
        elseif uR <= 0
            return flux(uR)
        else
            return 0.0                 # rarefaction fan contains 0
        end
    end
end

# Build numerical interface fluxes with MUSCL reconstruction (periodic)
function build_fluxes!(fhat, u; limiter::Symbol = :mc)
    slopes = similar(u)
    inds = axes(u, 1)
    fi, li = firstindex(u), lastindex(u)

    @inbounds for i in inds
        im = (i == fi) ? li : i - 1
        ip = (i == li) ? fi : i + 1
        slopes[i] = slope_limited(u[im], u[i], u[ip]; limiter = limiter)
    end

    amax = 0.0
    @inbounds for i in inds
        ip = (i == li) ? fi : i + 1
        uL = u[i]  + 0.5*slopes[i]
        uR = u[ip] - 0.5*slopes[ip]
        fhat[i] = godunov_flux_burgers(uL, uR)
        amax = max(amax, abs(uL), abs(uR), abs(u[i]))  # for CFL
    end
    return amax
end

@inline function euler_step!(uout, u, fhat, dt_dx)
    inds = axes(u, 1)
    fi, li = firstindex(u), lastindex(u)
    @inbounds for i in inds
        im = (i == fi) ? li : i - 1
        uout[i] = u[i] - dt_dx*(fhat[i] - fhat[im])
    end
end

"""
    burgers_muscl_godunov(N, L, T; CFL=0.45, limiter=:mc, ic=initial_condition)

Second-order TVD solver for inviscid Burgers on [0,L] with periodic BCs:
- MUSCL reconstruction with `limiter` (:mc or :minmod)
- Godunov Riemann solver
- SSPRK2 time stepping
Returns (x, u) at t = T (x are cell centers).
"""
function burgers_muscl_godunov(N, L, T; CFL::Float64 = 0.45, limiter::Symbol = :mc,
                               ic::Function = initial_condition)
    dx = L / N
    x  = @. (0.5:1:N-0.5) * dx
    u  = [ic(xi) for xi in x]

    fhat = similar(u)
    u1   = similar(u)
    u2   = similar(u)

    t = 0.0
    while t < T - eps()
        amax = build_fluxes!(fhat, u; limiter = limiter)
        dt = (amax > 0) ? CFL * dx / amax : (T - t)
        dt = min(dt, T - t)
        dt_dx = dt / dx

        euler_step!(u1, u, fhat, dt_dx)
        _ = build_fluxes!(fhat, u1; limiter = limiter)
        euler_step!(u2, u1, fhat, dt_dx)

        @inbounds for i in eachindex(u)
            u[i] = 0.5*(u[i] + u2[i])   # SSPRK2 combine
        end

        t += dt
    end
    return x, u
end

# ---------- Optional helpers (snapshots & animation) ----------

"""
    burgers_snapshots(N, L, times; CFL=0.45, limiter=:mc, ic=initial_condition)
Return (x, Dict{Float64,Vector}) with solutions at requested times.
"""
function burgers_snapshots(N, L, times; CFL::Float64 = 0.45, limiter::Symbol = :mc,
                           ic::Function = initial_condition)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = [ic(xi) for xi in x]
    fhat = similar(u); u1 = similar(u); u2 = similar(u)

    times = sort(collect(times))
    results = Dict{Float64, Vector{Float64}}()
    t = 0.0
    if !isempty(times) && isapprox(times[1], 0.0; atol = 1e-15)
        results[0.0] = copy(u)
        times = times[2:end]
    end

    while !isempty(times)
        target = first(times)
        while t < target - eps()
            amax = build_fluxes!(fhat, u; limiter = limiter)
            dt = (amax > 0) ? min(CFL*dx/amax, target - t) : (target - t)
            dt_dx = dt/dx

            euler_step!(u1, u, fhat, dt_dx)
            _ = build_fluxes!(fhat, u1; limiter = limiter)
            euler_step!(u2, u1, fhat, dt_dx)

            @inbounds for i in eachindex(u)
                u[i] = 0.5*(u[i] + u2[i])
            end
            t += dt
        end
        results[target] = copy(u)
        popfirst!(times)
    end
    return x, results
end

"""
    animate_burgers(N, L, T; CFL=0.45, limiter=:mc, fps=30, ylim=(-1.1,1.1), path="burgers_evolution.gif", ic=initial_condition)
Evolves and saves a GIF. Returns (x, uT).
"""
function animate_burgers(N, L, T; CFL::Float64=0.45, limiter::Symbol=:mc,
                         fps::Integer=30, ylim=(-1.1,1.1),
                         path::AbstractString="burgers_evolution.gif",
                         ic::Function = initial_condition)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = [ic(xi) for xi in x]
    fhat = similar(u); u1 = similar(u); u2 = similar(u)

    t = 0.0
    anim = Animation()

    # first frame
    p = plot(x, u, xlabel="x", ylabel="u", ylim=ylim, label=false,
             title = "t = $(round(t, digits=3))")
    frame(anim, p)

    while t < T - eps()
        amax = build_fluxes!(fhat, u; limiter=limiter)
        dt = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        dt_dx = dt/dx

        euler_step!(u1, u, fhat, dt_dx)
        _ = build_fluxes!(fhat, u1; limiter=limiter)
        euler_step!(u2, u1, fhat, dt_dx)

        @inbounds for i in eachindex(u)
            u[i] = 0.5*(u[i] + u2[i])
        end
        t += dt

        p = plot(x, u, xlabel="x", ylabel="u", ylim=ylim, label=false,
                 title = "t = $(round(t, digits=3))")
        frame(anim, p)
    end

    gif(anim, path, fps=fps)
    return x, u
end

# ---------- Additional numerical flux builders ----------

# First-order Godunov (upwind for Burgers) using piecewise-constant states
function build_fluxes_godunov_firstorder!(fhat, u)
    inds = axes(u, 1); fi, li = firstindex(u), lastindex(u)
    amax = 0.0
    @inbounds for i in inds
        ip = (i == li) ? fi : i + 1
        uL = u[i]
        uR = u[ip]
        fhat[i] = godunov_flux_burgers(uL, uR)
        amax = max(amax, abs(uL), abs(uR))
    end
    return amax
end

# Lax–Friedrichs (Rusanov) flux with global alpha = max |u|
function build_fluxes_lax_friedrichs!(fhat, u)
    inds = axes(u, 1); fi, li = firstindex(u), lastindex(u)
    amax = 0.0
    @inbounds for i in inds
        amax = max(amax, abs(u[i]))
    end
    @inbounds for i in inds
        ip = (i == li) ? fi : i + 1
        uL = u[i]
        uR = u[ip]
        fhat[i] = 0.5*(flux(uL) + flux(uR)) - 0.5*amax*(uR - uL)
    end
    return amax
end

# --- Fixed Finite-Volume Lax–Wendroff (Richtmyer two-step) ---
"""
    build_fluxes_lax_wendroff!(fhat, u, lam)

"""
function build_fluxes_lax_wendroff!(fhat, u, lam::Float64; add_visc::Bool=false, epsv::Float64=0.5)
    fi, li = firstindex(u), lastindex(u)
    amax = 0.0
    @inbounds for i in fi:li
        ip = (i == li) ? fi : i+1
        fL = 0.5*u[i]^2
        fR = 0.5*u[ip]^2
        uhalf = 0.5*(u[i] + u[ip]) - 0.5*lam*(fR - fL)
        F = 0.5*uhalf^2
        if add_visc
            α = max(abs(u[i]), abs(u[ip]), abs(uhalf))
            F -= 0.5*epsv*α*(u[ip] - u[i])   # Needs damping near shocks because it blows up otherwise
        end
        fhat[i] = F
        amax = max(amax, abs(u[i]), abs(u[ip]), abs(uhalf))
    end
    return amax
end




function burgers_upwind_godunov(N, L, T; CFL::Float64=0.45, ic::Function = initial_condition)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = [ic(xi) for xi in x]
    fhat = similar(u); u1 = similar(u)

    t = 0.0
    while t < T - eps()
        amax = build_fluxes_godunov_firstorder!(fhat, u)
        dt = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        euler_step!(u1, u, fhat, dt/dx)
        u, u1 = u1, u
        t += dt
    end
    return x, u
end

function burgers_lax_friedrichs(N, L, T; CFL::Float64=0.45, ic::Function = initial_condition)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = [ic(xi) for xi in x]
    fhat = similar(u); u1 = similar(u)

    t = 0.0
    while t < T - eps()
        amax = build_fluxes_lax_friedrichs!(fhat, u)
        dt = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        euler_step!(u1, u, fhat, dt/dx)
        u, u1 = u1, u
        t += dt
    end
    return x, u
end

# Fixed FV Lax–Wendroff (uses a single λ per step consistently)
function burgers_lax_wendroff(N, L, T; CFL::Float64=0.95, ic::Function=initial_condition)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = [ic(xi) for xi in x]
    fhat = similar(u); u1 = similar(u)
    t = 0.0
    while t < T - eps()
        umax0 = maximum(abs, u)
        dt    = (umax0 > 0) ? CFL*dx/umax0 : (T - t)
        dt    = min(dt, T - t)
        lam   = dt/dx
        umax_half = build_fluxes_lax_wendroff!(fhat, u, lam, add_visc = true)  # returns max(|u|,|u_half|)
        umax = max(umax0, umax_half)
        dt1  = (umax > 0) ? CFL*dx/umax : (T - t)
        dt1  = min(dt1, T - t)
        if dt1 < dt - eps()    
            lam = dt1/dx
            _   = build_fluxes_lax_wendroff!(fhat, u, lam, add_visc=true)  
            dt  = dt1
        end
        @inbounds begin
            fi, li = firstindex(u), lastindex(u)
            for i in fi:li
                im    = (i == fi) ? li : i-1
                u1[i] = u[i] - lam*(fhat[i] - fhat[im])
            end
        end
        u, u1 = u1, u
        t += dt
    end
    return x, u
end

"""
    burgers_compare_at(N, L, T; CFL=0.45, limiter=:mc, ic=initial_condition)

Return (x, Dict{String,Vector}) of solutions from several schemes at time T.
"""
function burgers_compare_at(N, L, T; CFL::Float64=0.45, limiter::Symbol=:mc,
                            ic::Function = initial_condition)
    x1, u_up   = burgers_upwind_godunov(N, L, T; CFL=CFL, ic=ic)
    x2, u_lf   = burgers_lax_friedrichs(N, L, T; CFL=CFL, ic=ic)
    x3, u_lw   = burgers_lax_wendroff(N, L, T; CFL=CFL, ic=ic)
    x4, u_mg   = burgers_muscl_godunov(N, L, T; CFL=CFL, limiter=limiter, ic=ic)
    results = Dict(
        "Upwind/Godunov (1st)" => u_up,
        "Lax–Friedrichs (1st)" => u_lf,
        "Lax–Wendroff (2nd)"   => u_lw,
        "MUSCL–Godunov (TVD)"  => u_mg,
    )
    return x3, results
end

"""
    burgers_compare_vs_exact(N, L, T;
        CFL=0.45, limiter=:mc,
        uL=1.0, uR=0.0, x0=0.5)

Run several schemes at time T for a Riemann IC (uL|uR at x0) and return
(x, Dict(label=>u), u_exact).
"""
function burgers_compare_vs_exact(N, L, T;
        CFL::Float64=0.45, limiter::Symbol=:mc,
        uL::Float64=1.0, uR::Float64=0.0, x0::Float64=0.5)

    ic = x -> initial_condition_riemann(x; uL=uL, uR=uR, x0=x0, L=L)

    x1, u_up = burgers_upwind_godunov(N, L, T; CFL=CFL, ic=ic)
    x2, u_lf = burgers_lax_friedrichs(N, L, T; CFL=CFL, ic=ic)
    x3, u_lw = burgers_lax_wendroff(N, L, T; CFL=CFL, ic=ic)         
    x4, u_mg = burgers_muscl_godunov(N, L, T; CFL=CFL, limiter=limiter, ic=ic)

    u_ex = exact_burgers_riemann(x1, T; uL=uL, uR=uR, x0=x0, L=L)

    results = Dict(
        "Upwind/Godunov (1st)" => u_up,
        "Lax–Friedrichs (1st)" => u_lf,
        "Lax–Wendroff (2nd)"    => u_lw,
        "MUSCL–Godunov (TVD)"  => u_mg,
    )
    return x1, results, u_ex
end

end # module
