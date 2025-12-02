cd(@__DIR__)

include("z_test.jl")   # module WBShallowWater

using Plots
theme(:default)

function run_sim_equil_18()
    # ---------------- Parameters ----------------
    Nx, Ny = 100, 50
    ng     = 2

    Lx, Ly = 1.0, 0.5
    dx, dy = Lx / Nx, Ly / Ny

    x0, y0 = 0.0, 0.0
    g      = 9.81

    fhat   = 1.0e-4      # nonzero Coriolis!
    beta   = 0.0         # keep f constant for now
    theta  = 1.3

    par = WBShallowWater.Params(Nx, Ny, ng, dx, dy, x0, y0, g, fhat, beta, theta)

    # ---------------- Allocate fields ----------------
    S    = WBShallowWater.State(par)
    TF   = WBShallowWater.TopographyFaces(par)
    E    = WBShallowWater.Equilibrium(par)
    sl   = WBShallowWater.EquilSlopes(par)
    Frec = WBShallowWater.FaceReconstruction(par)
    Fx   = WBShallowWater.FluxX(par)
    Gy   = WBShallowWater.FluxY(par)
    Src  = WBShallowWater.Sources(par)
    k1   = WBShallowWater.State(par)
    k2   = WBShallowWater.State(par)
    Sst  = WBShallowWater.State(par)

    # Handy sizes
    nx, ny = size(S.h)

    # ---------------- Bottom & provisional IC ----------------
    # (1.8) requires B_x = 0, so let B = B(y) only
    Bfun(x,y) = 0.1 * sin(2π * y / Ly)

    # Target mean free-surface level and velocity
    w_ref = 1.0          # reference w
    u0    = 0.5          # constant x-velocity
    v0    = 0.0

    # Provisional IC: h ≈ w_ref - B(y); will be adjusted to enforce L const
    ic(x,y) = WBShallowWater.ICValue(w_ref - Bfun(x,y), u0, v0, Bfun(x,y))

    # Build topography & set ICs
    WBShallowWater.build_topography_faces!(TF, par, Bfun)
    WBShallowWater.set_initial_conditions!(S, par, ic)
    S.B .= TF.Bc   # ensure consistency Bc ↔ S.B

    # ---------------- Enforce L ≡ const (discrete version of (1.8)) ----------------
    # Compute U (and L) for the provisional state
    WBShallowWater.compute_equilibrium!(E, S, par)

    # Choose L0 so that depths stay positive
    L0 = g * w_ref   # so h ~ w_ref - B - U; U is small for realistic f,u0

    for j in 1:ny, i in 1:nx
        S.h[i,j] = L0 / g - S.B[i,j] - E.U[i,j]
    end

    # Optional: recompute equilibrium once after adjusting h
    WBShallowWater.compute_equilibrium!(E, S, par)

    # ---------------- Time integration ----------------
    # CFL-based dt
    cmax_est = sqrt(g * w_ref)
    CFL      = 0.2
    dt       = CFL * min(dx, dy) / cmax_est
    Nt       = 1000
    t        = 0.0

    println("Using dt = $dt, total time T = $(dt*Nt)")

    for step in 1:Nt
        WBShallowWater.rk2_step!(S, dt, TF, par,
                                 E, sl, Frec, Fx, Gy, Src,
                                 k1, k2, Sst)
        t += dt
    end

    println("Finished at t = ", t)

    # ---------------- Extract physical domain ----------------
    ix = (ng+1):(ng+Nx)
    jy = (ng+1):(ng+Ny)

    x_coords = [WBShallowWater.x_center(i, par) for i in ix]
    y_coords = [WBShallowWater.y_center(j, par) for j in jy]

    h_phys = S.h[ix, jy]
    B_phys = S.B[ix, jy]
    w_phys = h_phys .+ B_phys
    η_phys = w_phys .- w_ref

    # Safe velocities
    u_phys = similar(h_phys)
    v_phys = similar(h_phys)
    eps_h  = 1e-10

    for J in eachindex(jy)
        for I in eachindex(ix)
            h = h_phys[I,J]
            if h > eps_h
                u_phys[I,J] = S.hu[ix[I], jy[J]] / h
                v_phys[I,J] = S.hv[ix[I], jy[J]] / h
            else
                u_phys[I,J] = 0.0
                v_phys[I,J] = 0.0
            end
        end
    end

    # Diagnostics
    η_flat  = vec(η_phys)
    u_flat  = vec(u_phys .- u0)
    v_flat  = vec(v_phys .- v0)

    h_min = minimum(h_phys)
    h_max = maximum(h_phys)
    η_max = maximum(abs, η_flat)
    u_dev = maximum(abs, u_flat)
    v_dev = maximum(abs, v_flat)

    println("min h     = ", h_min, ", max h = ", h_max)
    println("max |η|   = ", η_max)
    println("max |u-u0| = ", u_dev)
    println("max |v-v0| = ", v_dev)

    # ---------------- Plots ----------------
    p1 = heatmap(
        x_coords, y_coords, η_phys',
        aspect_ratio = :equal,
        xlabel = "x", ylabel = "y",
        title = "η (free surface perturbation) at t = $(round(t, digits=3))",
        colorbar_title = "η",
    )

    p2 = heatmap(
        x_coords, y_coords, (u_phys .- u0)',
        aspect_ratio = :equal,
        xlabel = "x", ylabel = "y",
        title = "u - u₀ at t = $(round(t, digits=3))",
        colorbar_title = "u - u₀",
    )

    p3 = heatmap(
        x_coords, y_coords, v_phys',
        aspect_ratio = :equal,
        xlabel = "x", ylabel = "y",
        title = "v at t = $(round(t, digits=3))",
        colorbar_title = "v",
    )

    jmid_idx = div(length(jy), 2)
    η_mid = η_phys[:, jmid_idx]

    p4 = plot(
        x_coords, η_mid,
        xlabel = "x", ylabel = "η(x, y_mid)",
        title  = "Midline η at t = $(round(t, digits=3))",
    )

    display(plot(p1, p2, p3, p4, layout = (2, 2)))
end

run_sim_equil_18()
