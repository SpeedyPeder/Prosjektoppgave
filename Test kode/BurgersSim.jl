module BurgersSim

using Plots

# --- Physics ---
flux(u) = 0.5*u^2
function initial_condition(x)
    return  (0.0 <= x && x <= 0.1) ? 1.0 : 0.0 #IC1
    #return sin(2π*x)                          #IC2
end

# === Exact solution for IC1 ========================================
analytic_burgers_pulse_outflow(x::AbstractVector, T; xR=0.1, uL=1.0, uR=0.0) = begin
    xs = xR + 0.5*(uL + uR)*T     # = xR + 0.5*T
    @. x < xs ? uL : uR
end

# === Flux limiter function ===============================================
function flux_limiter(r::Float64, limiter::Symbol)
    if limiter == :minmod
        return max(0.0, min(1.0, r))
    elseif limiter == :mc
        return max(0.0, min(2r, (1+r)/2, 2))
    elseif limiter == :superbee
        return max(0.0, min(2r, 1.0), min(r, 2.0))
    elseif limiter == :vanleer
        return (r + abs(r)) / (1 + abs(r))
    else
        error("Unknown limiter: $limiter")
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

function fill_ghosts_outflow!(u)
    # left boundary (x=0): copy first interior value
    u[2] = u[3]
    u[1] = u[3]
    # right boundary (x=L): copy last interior value
    u[end-1] = u[end-2]
    u[end]   = u[end-2]
    return nothing
end

# Build numerical interface fluxes with MUSCL reconstruction (periodic, upwind interface r)
function build_fluxes!(fhat, u; limiter::Symbol = :mc)
    @assert length(fhat) == length(u)
    ε = 1e-14
    UL = similar(u); UR = similar(u)
    @inbounds for i in 2:length(u)-3
        im1 = i-1; ip1 = i+1; ip2 = i+2

        a_int = 0.5*(u[i] + u[ip1]) 
        # upwind, interface-based Sweby ratio
        r = (a_int >= 0.0) ?
            (u[i]   - u[im1]) / ((u[ip1] - u[i]) + ε) :
            (u[ip2] - u[ip1]) / ((u[ip1] - u[i]) + ε)
        ϕ = flux_limiter(r, limiter)

        # MUSCL reconstruction at i+1/2
        UL[i] = u[i]   + 0.5*ϕ*(u[i]   - u[im1])
        UR[i] = u[ip1] - 0.5*ϕ*(u[ip2] - u[ip1])

        fhat[i] = godunov_flux_burgers(UL[i], UR[i])
    end

    return maximum(abs.(u[3:end-2]))  # CFL from interior
end

@inline function euler_step!(uout, u, fhat, dt_dx)
    @inbounds for i in 3:length(u)-2
        uout[i] = u[i] - dt_dx*(fhat[i] - fhat[i-1])
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
    x  = @. (0.5:1:N-0.5) * dx            # interior cell centers (for plotting)

    u   = zeros(N+4)                      # add two ghost cells per side
    u[3:N+2] .= initial_condition.(x)

    fhat = zeros(N+4)
    u1   = similar(u)
    u2   = similar(u)

    t = 0.0
    while t < T - eps()
        fill_ghosts_outflow!(u)
        amax = build_fluxes!(fhat, u; limiter = limiter)
        dt   = (amax > 0) ? min(CFL * dx / amax, T - t) : (T - t)
        dt_dx = dt / dx

        euler_step!(u1, u, fhat, dt_dx)

        fill_ghosts_outflow!(u1)
        _ = build_fluxes!(fhat, u1; limiter = limiter)
        euler_step!(u2, u1, fhat, dt_dx)

        @inbounds for i in 3:length(u)-2
            u[i] = 0.5*(u[i] + u2[i])     # SSPRK2 combine (interior only)
        end

        t += dt
    end

    return x, @view u[3:end-2]           # return interior solution
end

# ---------- Optional helpers (snapshots & animation) ----------

"""
    burgers_snapshots(N, L, times; CFL=0.45, limiter=:mc)

Return (x, Dict{Float64,Vector}) with solutions at requested times.
Uses outflow (non-reflecting) BCs with two ghost cells per side.
"""
function burgers_snapshots(N, L, times; CFL::Float64 = 0.45, limiter::Symbol = :mc)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx                  # interior centers

    # allocate with ghosts
    u   = zeros(N+4)
    u[3:N+2] .= initial_condition.(x)
    fhat = zeros(N+4); u1 = similar(u); u2 = similar(u)

    times = sort(collect(times))
    results = Dict{Float64, Vector{Float64}}()
    t = 0.0
    if !isempty(times) && isapprox(times[1], 0.0; atol = 1e-15)
        results[0.0] = collect(u[3:end-2])      # store interior only
        times = times[2:end]
    end

    while !isempty(times)
        target = first(times)
        while t < target - eps()
            fill_ghosts_outflow!(u)
            amax = build_fluxes!(fhat, u; limiter = limiter)
            dt   = (amax > 0) ? min(CFL*dx/amax, target - t) : (target - t)
            dt_dx = dt/dx

            euler_step!(u1, u, fhat, dt_dx)

            fill_ghosts_outflow!(u1)
            _ = build_fluxes!(fhat, u1; limiter = limiter)
            euler_step!(u2, u1, fhat, dt_dx)

            @inbounds for i in 3:length(u)-2
                u[i] = 0.5*(u[i] + u2[i])       # SSPRK2 combine (interior)
            end
            t += dt
        end
        results[target] = collect(u[3:end-2])    # store interior
        times = times[2:end]
    end
    return x, results
end


"""
    animate_burgers(N, L, T; CFL=0.45, limiter=:mc, fps=30, ylim=(-1.1,1.1), path="burgers_evolution.gif")

Evolves and saves a GIF. Returns (x, uT). Uses outflow BCs with ghosts.
"""
function animate_burgers(N, L, T; CFL::Float64=0.45, limiter::Symbol=:mc,
                         fps::Integer=30, ylim=(-1.1,1.1),
                         path::AbstractString="burgers_evolution.gif")
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx                  # interior centers

    # allocate with ghosts
    u   = zeros(N+4)
    u[3:N+2] .= initial_condition.(x)
    fhat = zeros(N+4); u1 = similar(u); u2 = similar(u)

    t = 0.0
    anim = Animation()

    # first frame (plot interior only)
    p = plot(x, u[3:end-2], xlabel="x", ylabel="u", ylim=ylim, label=false,
             title = "t = $(round(t, digits=3))")
    frame(anim, p)

    while t < T - eps()
        fill_ghosts_outflow!(u)
        amax = build_fluxes!(fhat, u; limiter=limiter)
        dt   = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        dt_dx = dt/dx

        euler_step!(u1, u, fhat, dt_dx)

        fill_ghosts_outflow!(u1)
        _ = build_fluxes!(fhat, u1; limiter=limiter)
        euler_step!(u2, u1, fhat, dt_dx)

        @inbounds for i in 3:length(u)-2
            u[i] = 0.5*(u[i] + u2[i])           # SSPRK2 combine (interior)
        end
        t += dt

        p = plot(x, u[3:end-2], xlabel="x", ylabel="u", ylim=ylim, label=false,
                 title = "t = $(round(t, digits=3))")
        frame(anim, p)
    end

    gif(anim, path, fps=fps)
    return x, u[3:end-2]
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
    x3, u_lw   = burgers_lax_wendroff(N, L, T; CFL=CFL)
    x4, u_mg   = burgers_muscl_godunov(N, L, T; CFL=CFL, limiter=limiter)
    results = Dict(
        "Upwind/Godunov (1st)" => u_up,
        "Lax–Friedrichs (1st)" => u_lf,
        "Lax–Wendroff (2nd)"   => u_lw,
        "MUSCL–Godunov (TVD)"  => u_mg,
    )
    return x1, results
end

#--------- Limiter comparison helper ----------
function compare_limiters_zoom(
        N::Int, L::Real, T::Real;
        CFL=0.45,
        limiters=[:minmod, :mc, :superbee, :vanleer],
        shock_center=0.5, zoom_halfwidth=L/2,
        do_plot::Bool=true)

    xs  = Dict{Symbol, Vector{Float64}}()
    num = Dict{Symbol, Vector{Float64}}()

    for lim in limiters
        x, u = burgers_muscl_godunov(N, L, T; CFL=CFL, limiter=lim)
        xs[lim]  = collect(x)
        num[lim] = collect(u)
    end

    if do_plot
        @eval using Plots
        plt = plot(xlabel="x", ylabel="u",
                   title="Limiter comparison near shock at x = $shock_center",
                   legend=:topright, grid=false)

        for (i, lim) in enumerate(limiters)
            plot!(plt, xs[lim], num[lim];
                  lw=2, label="Limiter: $lim",
                  linestyle=[:solid, :dash, :dot, :dashdot][mod1(i,4)])
        end

        # --- Zoom window
        xlow  = shock_center - zoom_halfwidth
        xhigh = shock_center + zoom_halfwidth
        xlims!(plt, (xlow, xhigh))

        # --- Auto y-range but clipped to a sensible interval
        all_u = vcat(values(num)...)
        umin, umax = extrema(all_u)
        margin = 0.05 * (umax - umin)
        ylims!(plt, (umin - margin, umax + margin))

        display(plt)
        try savefig(plt, "burgers_limiters_zoom_t$(T).png") catch; end
    end

    return (; x=xs, u=num)
end

end # module
