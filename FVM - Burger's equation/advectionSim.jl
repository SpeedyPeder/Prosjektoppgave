module AdvectionSim

using Plots

# --- initial_condition -------------------------------

function initial_condition(x)
    return (0.0 <= x && x <= 0.1) ? 1.0 : 0.0
    # return sin(2π*x)
end

function classic_advection_IC(x)
    if 0.1 ≤ x ≤ 0.4
            # Normalize cosine so it is 1 at the center and 0 at edges
            return 0.5 * (1 + cos(pi * (x - 0.25) / 0.15))
    end
        # Step function
    if 0.5 ≤ x ≤ 0.9
        return 1.0
    end
    return 0.0
end

# --- Flux limiter ------------------------------------------------------------

"""
    flux_limiter(r, limiter)

Standard TVD flux limiters: :minmod, :mc, :superbee, :vanleer.
"""
function flux_limiter(r::Real, limiter::Symbol)
    if limiter === :minmod
        return max(0.0, min(1.0, r))
    elseif limiter === :mc
        return max(0.0, min(2r, (1+r)/2, 2.0))
    elseif limiter === :superbee
        return max(0.0, min(2r, 1.0), min(r, 2.0))
    elseif limiter === :vanleer
        return (r + abs(r)) / (1 + abs(r))
    else
        error("Unknown limiter: $limiter")
    end
end

# --- Periodic ghost cells ----------------------------------------------------

"""
    fill_ghosts_periodic!(u)

Two ghost cells on each side.
Interior cells are indices 3 : length(u)-2.
"""
function fill_ghosts_periodic!(u)
    # left ghosts copy from right interior
    u[2] = u[end-2]   # u_0   <- u_N
    u[1] = u[end-3]   # u_-1  <- u_{N-1}

    # right ghosts copy from left interior
    u[end-1] = u[3]   # u_{N+1}  <- u_1
    u[end]   = u[4]   # u_{N+2}  <- u_2
    return nothing
end

# --- MUSCL reconstruction + upwind flux for linear advection -----------------
function build_fluxes_muscl_advection!(fhat, u; a::Real = 1.0, limiter::Symbol = :mc)
    @assert length(fhat) == length(u)
    ε  = 1e-5
    UL = similar(u)
    UR = similar(u)
    @inbounds for i in 2:length(u)-2   # interface i+1/2
        rL = (u[i]- u[i-1])/ (u[i+1]-u[i] + ε)
        ϕL = flux_limiter(rL, limiter)
        UL[i] = u[i] + 0.5 * ϕL * (u[i+1]-u[i])

        rR = (u[i+1]- u[i])/ (u[i+2]-u[i+1] + ε)
        ϕR = flux_limiter(rR, limiter)
        UR[i] = u[i+1] - 0.5 * ϕR * (u[i+2]-u[i+1])   

        # Upwind flux for linear advection
        if a > 0
            fhat[i] = a * UL[i]
        elseif a < 0
            fhat[i] = a * UR[i]
        else
            fhat[i] = 0.0
        end
    end

    return abs(a)
end


# --- Single Euler step -------------------------------------------------------

@inline function euler_step!(uout, u, fhat, dt_dx)
    @inbounds for i in 3:length(u)-2
        uout[i] = u[i] - dt_dx * (fhat[i] - fhat[i-1])
    end
    return nothing
end

# --- Grid + allocation helper ------------------------------------------------

"""
    _alloc_with_ghosts(N, L, ic)

Allocate grid and arrays with 2 ghost cells per side.
Returns (dx, x, u, fhat, u1, u2) where `x` are cell centers in [0,L].
`ic` is a function `ic(x)` used to fill the initial condition.
"""
function _alloc_with_ghosts(N::Int, L, ic::Function)
    dx = L / N
    x  = @. (0.5:1:N-0.5) * dx

    u = zeros(N+4)
    u[3:N+2] .= ic.(x)

    fhat = zeros(N+4)
    u1   = similar(u)
    u2   = similar(u)
    return dx, x, u, fhat, u1, u2
end

# --- Main MUSCL solver for advection ----------------------------------------
function advection_muscl(N::Int, L, T;
                         a::Real = 1.0,
                         CFL::Float64 = 0.45,
                         limiter::Symbol = :mc,
                         ic::Function = initial_condition)

    dx, x, u, fhat, u1, u2 = _alloc_with_ghosts(N, L, ic)
    t = 0.0
    while t < T - eps()
        fill_ghosts_periodic!(u)
        amax = build_fluxes_muscl_advection!(fhat, u; a=a, limiter=limiter)
        if amax == 0
            # no advection; just jump to final time
            break
        end
        dt  = min(CFL * dx / amax, T - t)
        dt_dx = dt / dx

        # SSPRK2 stage 1
        euler_step!(u1, u, fhat, dt_dx)

        # SSPRK2 stage 2
        fill_ghosts_periodic!(u1)
        _ = build_fluxes_muscl_advection!(fhat, u1; a=a, limiter=limiter)
        euler_step!(u2, u1, fhat, dt_dx)

        @inbounds for i in 3:length(u)-2
            u[i] = 0.5 * (u[i] + u2[i])
        end

        t += dt
    end

    return x, @view u[3:end-2]
end





function analytic_advection(x::AbstractVector, T;
                            a::Real = 1.0,
                            ic::Function = initial_condition,
                            L::Real = 1.0)
    u_exact = similar(x)
    @inbounds for i in eachindex(x)
        ξ = x[i] - a*T
        # Map ξ into [0, L) using periodicity
        ξ = ξ - floor(ξ / L) * L
        u_exact[i] = ic(ξ)
    end
    return u_exact
end


function advection_compare_limiters_separate(N, L, T;
        a::Real = 1.0,
        CFL::Float64 = 0.45,
        limiters = [:minmod, :mc, :superbee, :vanleer],
        ic::Function = mixed_cosine_step_IC,
        saveprefix::Union{Nothing,AbstractString} = nothing)

    results = Dict{Symbol,NamedTuple}()

    for lim in limiters
        x, u_num = advection_muscl(N, L, T; a = a, CFL = CFL, limiter = lim, ic = ic)
        u_exact = analytic_advection(collect(x), T; a = a, ic = ic, L = L)
        plt = plot(x, u_exact; lw = 2, label = "exact", xlabel = "x", ylabel = "u",
                   title = "Limiter: $(String(lim)), N=$N, T=$T, a=$a")
        scatter!(plt, x, u_num; label = "numeric",ms = 3)
        display(plt)
        if saveprefix !== nothing
            fname = "$(saveprefix)_$(String(lim)).png"
            try
                dir = dirname(fname)
                if !isempty(dir); mkpath(dir); end
                savefig(plt, fname)
            catch err
                @warn "Could not save figure" error=err path=fname
            end
        end
        results[lim] = (x = collect(x), u = collect(u_num), u_exact = u_exact)
    end
    return results
end



end # module
