module BurgersFDSim

# IC and flux
flux(u) = 0.5u^2
initial_condition(x) = sin(2π*x)   

#Index helpers
@inline previdx(i, fi, li) = (i == fi) ? li : i-1
@inline nextidx(i, fi, li) = (i == li) ? fi : i+1

"""
    burgers_fd_lf(N, L, T; CFL=0.45)

Finite-difference Lax–Friedrichs (first order).
Periodic BCs on [0,L]. Returns (x, u(T)).
"""
function burgers_fd_lf(N::Int, L::Float64, T::Float64; CFL::Float64=0.45)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = initial_condition.(x)
    u1 = similar(u)

    t = 0.0
    fi, li = firstindex(u), lastindex(u)

    while t < T - eps()
        umax = maximum(abs, u)
        dt   = (umax > 0) ? CFL*dx/umax : (T - t)
        dt   = min(dt, T - t)
        lam  = dt/dx

        @inbounds for i in eachindex(u)
            im = previdx(i, fi, li)
            ip = nextidx(i, fi, li)
            # Lax–Friedrichs:
            # u^{n+1}_i = 0.5(u_{i-1}+u_{i+1}) - 0.5*lam*(f_{i+1}-f_{i-1})
            u1[i] = 0.5*(u[im] + u[ip]) - 0.5*lam*(flux(u[ip]) - flux(u[im]))
        end

        u .= u1
        t += dt
    end
    return x, u
end

"""
    burgers_fd_lw(N, L, T; CFL=0.95)

Finite-difference Lax–Wendroff (second order; may oscillate near shocks).
Periodic BCs on [0,L]. Returns (x, u(T)).
"""
function burgers_fd_lw(N::Int, L::Float64, T::Float64; CFL::Float64=0.95)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx
    u  = initial_condition.(x)
    u1 = similar(u)           # stage
    f  = similar(u)
    fi, li = firstindex(u), lastindex(u)

    t = 0.0
    while t < T - eps()
        umax = maximum(abs, u)
        dt   = (umax > 0) ? CFL*dx/umax : (T - t)
        dt   = min(dt, T - t)
        lam  = dt/dx

        @inbounds for i in eachindex(u); f[i] = flux(u[i]); end

        # Predictor (forward Euler using centered space)
        @inbounds for i in eachindex(u)
            im = previdx(i, fi, li)
            ip = nextidx(i, fi, li)
            u1[i] = u[i] - 0.5*lam*(f[ip] - f[im])
        end

        # Recompute flux at predictor and correct
        @inbounds for i in eachindex(u); f[i] = flux(u1[i]); end
        @inbounds for i in eachindex(u)
            im = previdx(i, fi, li)
            ip = nextidx(i, fi, li)
            u[i] = 0.5*(u[i] + (u[i] - lam*(f[ip] - f[im])))
        end

        t += dt
    end
    return x, u
end

end # module
