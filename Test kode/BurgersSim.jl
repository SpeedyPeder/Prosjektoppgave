module BurgersSim

using Plots

# --- Physics ---
flux(u) = 0.5u^2
initial_condition(x) = sin(2π*x)

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
            return 0.0           # rarefaction fan contains 0
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
    burgers_muscl_godunov(N, L, T; CFL=0.45, limiter=:mc)

Second-order TVD solver for inviscid Burgers on [0,L] with periodic BCs:
- MUSCL reconstruction with `limiter` (:mc or :minmod)
- Godunov Riemann solver
- SSPRK2 time stepping
Returns (x, u) at t = T (x are cell centers).
"""
function burgers_muscl_godunov(N, L, T; CFL::Float64 = 0.45, limiter::Symbol = :mc)
    dx = L / N
    x  = @. (0.5:1:N-0.5) * dx
    u  = initial_condition.(x)

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
    burgers_snapshots(N, L, times; CFL=0.45, limiter=:mc)

Return (x, Dict{Float64,Vector}) with solutions at requested times.
"""
function burgers_snapshots(N, L, times; CFL::Float64 = 0.45, limiter::Symbol = :mc)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = initial_condition.(x)
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
        times = times[2:end]
    end
    return x, results
end

"""
    animate_burgers(N, L, T; CFL=0.45, limiter=:mc, fps=30, ylim=(-1.1,1.1), path="burgers_evolution.gif")

Evolves and saves a GIF. Returns (x, uT).
"""
function animate_burgers(N, L, T; CFL::Float64=0.45, limiter::Symbol=:mc,
                         fps::Integer=30, ylim=(-1.1,1.1),
                         path::AbstractString="burgers_evolution.gif")
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = initial_condition.(x)
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

# Lax–Wendroff (Richtmyer 2-step)
function build_fluxes_lax_wendroff!(fhat, u, dt_dx)
    inds = axes(u, 1); fi, li = firstindex(u), lastindex(u)
    amax = 0.0
    @inbounds for i in inds
        ip = (i == li) ? fi : i + 1
        u_half = 0.5*(u[i] + u[ip]) - 0.5*dt_dx*(flux(u[ip]) - flux(u[i]))
        fhat[i] = flux(u_half)
        amax = max(amax, abs(u[i]), abs(u[ip]), abs(u_half))
    end
    return amax
end

# ---------- Drivers ----------

function burgers_upwind_godunov(N, L, T; CFL::Float64=0.45)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = initial_condition.(x)
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

function burgers_lax_friedrichs(N, L, T; CFL::Float64=0.45)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = initial_condition.(x)
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

function burgers_lax_wendroff(N, L, T; CFL::Float64=0.45)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = initial_condition.(x)
    fhat = similar(u); u1 = similar(u)

    t = 0.0
    while t < T - eps()
        a_guess = maximum(abs, u)
        dt = (a_guess > 0) ? min(CFL*dx/a_guess, T - t) : (T - t)
        dt_dx = dt/dx
        _ = build_fluxes_lax_wendroff!(fhat, u, dt_dx)
        euler_step!(u1, u, fhat, dt_dx)
        u, u1 = u1, u
        t += dt
    end
    return x, u
end

"""
    burgers_compare_at(N, L, T; CFL=0.45, limiter=:mc)

Return (x, Dict{String,Vector}) of solutions from several schemes at time T.
"""
function burgers_compare_at(N, L, T; CFL::Float64=0.45, limiter::Symbol=:mc)
    x1, u_up   = burgers_upwind_godunov(N, L, T; CFL=CFL)
    x2, u_lf   = burgers_lax_friedrichs(N, L, T; CFL=CFL)
    #x3, u_lw   = burgers_lax_wendroff(N, L, T; CFL=CFL)
    x4, u_mg   = burgers_muscl_godunov(N, L, T; CFL=CFL, limiter=limiter)
    results = Dict(
        "Upwind/Godunov (1st)" => u_up,
        "Lax–Friedrichs (1st)" => u_lf,
        #"Lax–Wendroff (2nd)"   => u_lw,
        "MUSCL–Godunov (TVD)"  => u_mg,
    )
    return x1, results
end


end # module
