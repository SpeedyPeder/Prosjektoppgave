############################################################
# z_test_run.jl
#
# Test of well-balanced CU scheme (test_CDKLM)
# on Example 2 from CDKL:
# Geostrophic steady state with a periodic bottom,
# periodic BCs.
############################################################

using Plots

# Load your scheme implementation
include("x_test_boundaries.jl")
using .test_CDKLM

# ---------------------------------------------------------
# Example 2 setup (geostrophic equilibrium with periodic B)
#
# Domain: x ∈ [-5, 5], periodic
# Bottom: B(x) = f/g * sin(π x / 5), with f = g = 1
# Flow:
#   h(x) ≡ 1,
#   u(x) ≡ 0,
#   v(x) = (π/5) cos(π x / 5),
#   so K(x) ≡ 1 via K = g(h + B - V).
# ---------------------------------------------------------

# Bathymetry function B(x,y)
bfun(x, y) = sin(pi * x / 5)   # independent of y

# Set initial condition to the analytic equilibrium
function set_equilibrium_example2!(st, p)
    nx, ny = p.nx, p.ny
    x      = p.x

    # h ≡ 1, u ≡ 0, v(x) = (π/5) cos(π x / 5)
    for i in 1:nx, j in 1:ny
        h = 1.0
        u = 0.0
        v = (pi/5) * cos(pi * x[i] / 5)

        st.h[i,j]  = h
        st.hu[i,j] = h * u
        st.hv[i,j] = h * v
    end

    @views begin
        st.q[1, :, :] .= st.h
        st.q[2, :, :] .= st.hu
        st.q[3, :, :] .= st.hv
    end

    return nothing
end

# Compute equilibrium variables u, v, K, L from current state
function compute_equilibrium_vars(st, p)
    # Make copies of hu, hv because build_velocities may modify them
    h  = st.h
    hu = copy(st.hu)
    hv = copy(st.hv)

    u, v = test_CDKLM.build_velocities(p.x, p.y, h, hu, hv, p.Hmin)
    _, _, _, Vc, K, L = test_CDKLM.build_UV_KL(h, u, v, st.f, st.Bc,
                                               p.dx, p.dy, p.g)
    return u, v, K, L
end

############################################################
# Main driver: build grid, initialize, evolve, plot
############################################################

function main(;
    nx::Int = 200,           
    ny::Int = 200,            
    xL::Float64 = -5.0,
    xR::Float64 =  5.0,
    yL::Float64 =  0.0,
    yR::Float64 =  1.0,
    g::Float64  =  1.0,
    f_hat::Float64 = 1.0,    # f(y) = 1 + 0*y
    beta::Float64 = 0.0,
    CFL::Float64 = 0.5,
    Tfinal::Float64 = 200.0, # long run to imitate the paper
    Hmin::Float64 = 1e-6,
    limiter::Symbol = :minmod,
    bc::Symbol = :periodic,
)

    # --- 1) Grid ---
    x = collect(LinRange(xL, xR, nx))
    y = collect(LinRange(yL, yR, ny))

    dx = x[2] - x[1]
    dy = y[2] - y[1]

    # crude max wave speed estimate: c ≈ √(g h), with h ≈ 1
    cmax = sqrt(g)
    dt   = CFL * min(dx, dy) / cmax

    println("dx = $dx, dy = $dy, dt = $dt, Tfinal = $Tfinal")

    # --- 2) Init state & params (this builds B, f, work arrays, etc.) ---
    st, p = test_CDKLM.init_state(
        x, y, bfun, f_hat, beta;
        g = g, dt = dt, Hmin = Hmin, limiter = limiter, bc = bc
    )

    # --- 3) Overwrite q with analytic equilibrium ---
    set_equilibrium_example2!(st, p)

    # --- 4) Compute equilibrium variables at t = 0 ---
    u0, v0, K0, L0 = compute_equilibrium_vars(st, p)

    # Make 1D slices (middle in y)
    jmid  = Int(cld(p.ny, 2))
    xline = p.x
    u0_1d = u0[:, jmid]
    K0_1d = K0[:, jmid]

    # Save initial conservative fields for error tracking
    h_init  = copy(st.h)
    hu_init = copy(st.hu)
    hv_init = copy(st.hv)

    # --- 5) Time integration ---
    t      = 0.0
    nsteps = Int(ceil(Tfinal / p.dt))
    println("Running $nsteps steps...")

    for n in 1:nsteps
        test_CDKLM.step_RK2!(st, p)
        t += p.dt

        if n % 100 == 0 || n == nsteps
            err_h  = maximum(abs.(st.h  .- h_init))
            err_hu = maximum(abs.(st.hu .- hu_init))
            err_hv = maximum(abs.(st.hv .- hv_init))
            @info "step $n, t=$(round(t,digits=3))" err_h err_hu err_hv
        end
    end

    # --- 6) Equilibrium variables at final time ---
    uf, vf, Kf, Lf = compute_equilibrium_vars(st, p)
    uf_1d = uf[:, jmid]
    Kf_1d = Kf[:, jmid]

    # --- 7) Plots: initial vs final u(x), K(x) ---
    plt1 = plot(
        xline, u0_1d,
        label = "u(x,0)", lw = 2,
        xlabel = "x", ylabel = "u",
        title = "Example 2: u at t=0 and t=$(round(t,digits=2))",
    )
    plot!(plt1, xline, uf_1d, label = "u(x,T)", lw = 2, ls = :dash)

    plt2 = plot(
        xline, K0_1d,
        label = "K(x,0)", lw = 2,
        xlabel = "x", ylabel = "K",
        title = "Example 2: K at t=0 and t=$(round(t,digits=2))",
    )
    plot!(plt2, xline, Kf_1d, label = "K(x,T)", lw = 2, ls = :dash)

    display(plt1)
    display(plt2)

    return st, p
end

main()
