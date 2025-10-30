module BurgersSim

using Plots

# --- Physics ---
flux(u) = 0.5*u^2
function initial_condition(x)
    return  (0.0 <= x && x <= 0.1) ? 1.0 : 0.0 #IC1
    #return sin(2π*x)                          #IC2
end

# === Exact solution for IC1 ========================================
function analytic_burgers_pulse_outflow(x::AbstractVector, T; xR=0.1, uL=1.0, uR=0.0)
    xs = xR + 0.5*(uL + uR)*T  # shock position

    # Flat left of shock
    xL = [minimum(x), xs]
    uLvec = fill(uL, length(xL))

    # Flat right of shock
    xRvec = [xs, maximum(x)]
    uRvec = fill(uR, length(xRvec))

    # Concatenate for a discontinuous plot (gap at xs)
    x_plot = vcat(xL, xRvec)
    u_plot = vcat(uLvec, uRvec)
    return x_plot, u_plot
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

# Grid and init helper (with 2 ghost cells per side)
function _alloc_with_ghosts(N, L)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = zeros(N+4); u[3:N+2] .= initial_condition.(x)
    fhat = zeros(N+4); u1 = similar(u); u2 = similar(u)
    return dx, x, u, fhat, u1, u2
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
function build_fluxes_godunov_firstorder_outflow!(fhat, u)
    @inbounds for i in 2:length(u)-2      # interfaces i+1/2
        uL = u[i]
        uR = u[i+1]
        fhat[i] = godunov_flux_burgers(uL, uR)
    end
    return maximum(abs.(u[3:end-2]))      # CFL from interior
end

# Lax–Friedrichs (Rusanov) flux with global alpha = max |u|
function build_fluxes_lax_friedrichs_outflow!(fhat, u)
    α = maximum(abs.(u[3:end-2]))         # global wave speed from interior
    @inbounds for i in 2:length(u)-2
        uL = u[i]; uR = u[i+1]
        fhat[i] = 0.5*(flux(uL) + flux(uR)) - 0.5*α*(uR - uL)
    end
    return α
end

# Lax–Wendroff (Richtmyer 2-step)
function build_fluxes_lax_wendroff_outflow!(fhat, u, dt_dx)
    amax = 0.0
    @inbounds for i in 2:length(u)-2
        u_half = 0.5*(u[i] + u[i+1]) - 0.5*dt_dx*(flux(u[i+1]) - flux(u[i]))
        fhat[i] = flux(u_half)
        amax = max(amax, abs(u[i]), abs(u[i+1]), abs(u_half))
    end
    return amax
end

# ---------- Drivers ----------

function burgers_upwind_godunov_outflow(N, L, T; CFL=0.45)
    dx, x, u, fhat, u1, u2 = _alloc_with_ghosts(N, L)
    t = 0.0
    while t < T - eps()
        fill_ghosts_outflow!(u)
        amax = build_fluxes_godunov_firstorder_outflow!(fhat, u)
        dt = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        dt_dx = dt/dx
        euler_step!(u1, u, fhat, dt_dx)
        fill_ghosts_outflow!(u1)
        _ = build_fluxes_godunov_firstorder_outflow!(fhat, u1)
        euler_step!(u2, u1, fhat, dt_dx)
        @inbounds for i in 3:length(u)-2
            u[i] = 0.5*(u[i] + u2[i])
        end
        t += dt
    end
    return x, @view u[3:end-2]
end

function burgers_lax_friedrichs_outflow(N, L, T; CFL=0.45)
    dx, x, u, fhat, u1, u2 = _alloc_with_ghosts(N, L)
    t = 0.0
    while t < T - eps()
        fill_ghosts_outflow!(u)
        α = build_fluxes_lax_friedrichs_outflow!(fhat, u)
        dt = (α > 0) ? min(CFL*dx/α, T - t) : (T - t)
        dt_dx = dt/dx
        euler_step!(u1, u, fhat, dt_dx)
        fill_ghosts_outflow!(u1)
        _ = build_fluxes_lax_friedrichs_outflow!(fhat, u1)
        euler_step!(u2, u1, fhat, dt_dx)
        @inbounds for i in 3:length(u)-2
            u[i] = 0.5*(u[i] + u2[i])
        end
        t += dt
    end
    return x, @view u[3:end-2]
end

function burgers_lax_wendroff_outflow(N, L, T; CFL=0.45)
    dx, x, u, fhat, u1, u2 = _alloc_with_ghosts(N, L)
    t = 0.0
    while t < T - eps()
        # predictor for dt using interior |u|
        a_guess = maximum(abs.(u[3:end-2]))
        dt = (a_guess > 0) ? min(CFL*dx/a_guess, T - t) : (T - t)
        dt_dx = dt/dx

        fill_ghosts_outflow!(u)
        _ = build_fluxes_lax_wendroff_outflow!(fhat, u, dt_dx)
        euler_step!(u1, u, fhat, dt_dx)

        fill_ghosts_outflow!(u1)
        _ = build_fluxes_lax_wendroff_outflow!(fhat, u1, dt_dx)
        euler_step!(u2, u1, fhat, dt_dx)

        @inbounds for i in 3:length(u)-2
            u[i] = 0.5*(u[i] + u2[i])
        end
        t += dt
    end
    return x, @view u[3:end-2]
end

""""
    burgers_compare_at(N, L, T;
        method::Symbol = :muscl,
        limiter::Symbol = :mc,
        CFL::Float64 = 0.45,
        xR::Float64 = 0.1,
        show_analytic::Bool = true,
        title::AbstractString = "",
        savepath::Union{Nothing,AbstractString} = nothing)

Make a *single* plot for the chosen method with the numerical result shown as dots.
Valid `method` values:
  :upwind      -> first-order Godunov (outflow)
  :laxfriedrichs or :lf  -> Lax–Friedrichs (outflow)
  :laxwendroff  or :lw   -> Lax–Wendroff (outflow)
  :muscl       -> MUSCL–Godunov (outflow), uses `limiter` (:minmod, :mc, :superbee, :vanleer)

Returns (x, u) and displays the plot. If `savepath` is given, saves the figure there.
"""
function burgers_compare_at(N, L, T;
        method::Symbol = :muscl,
        limiter::Symbol = :mc,
        CFL::Float64 = 0.45,
        xR::Float64 = 0.1,
        show_analytic::Bool = true,
        title::AbstractString = "",
        savepath::Union{Nothing,AbstractString} = nothing)

    # run selected (all outflow-BC variants)
    local x, u, mlabel
    if method === :upwind
        x, u = burgers_upwind_godunov_outflow(N, L, T; CFL=CFL)
        mlabel = "Upwind/Godunov (1st)"
    elseif method === :laxfriedrichs || method === :lf
        x, u = burgers_lax_friedrichs_outflow(N, L, T; CFL=CFL)
        mlabel = "Lax–Friedrichs (1st)"
    elseif method === :laxwendroff || method === :lw
        x, u = burgers_lax_wendroff_outflow(N, L, T; CFL=CFL)
        mlabel = "Lax–Wendroff (2nd)"
    elseif method === :muscl
        x, u = burgers_muscl_godunov(N, L, T; CFL=CFL, limiter=limiter)
        mlabel = "MUSCL–Godunov ($(String(limiter)))"
    else
        error("Unknown method: $method")
    end

    # Ensure x is a Vector for plotting helper (in case it’s a range)
    xv = collect(x)
    uv = collect(u)

    # build plot: numeric dots
    plt_title = isempty(title) ? "N=$N, T=$T, method=$mlabel" : title
    plt = scatter(xv, uv; ms=5, label=mlabel,
                  xlabel="x", ylabel="u", title=plt_title, legend=:bottomright)

    # overlay analytic with vertical jump
    if show_analytic
        xA, uA = analytic_burgers_pulse_outflow(xv, T; xR=xR)
        plot!(plt, xA, uA; lw=3, color=:red, label="analytic")
    end

    display(plt)
    if savepath !== nothing
        try
            savefig(plt, savepath)
        catch e
            @warn "Could not save figure" error=e path=savepath
        end
    end

    return x, u
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
