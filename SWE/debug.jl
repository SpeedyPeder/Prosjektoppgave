# debug_CDKLM.jl
include("sweSim2D-Rotational.jl")
using .RotSW_CDKLM
using Plots
using Printf

# =========================
# Debug State with Visualization
# =========================
mutable struct DebugState
    step::Int
    stage::Int  # 0 = initial, 1 = RK stage 1, 2 = RK stage 2
    u::Array{Float64,2}
    v::Array{Float64,2}
    h::Array{Float64,2}
    Uface::Array{Float64,2}
    Vface::Array{Float64,2}
    K::Array{Float64,2}
    L::Array{Float64,2}
    uE::Array{Float64,2}; uW::Array{Float64,2}; uN::Array{Float64,2}; uS::Array{Float64,2}
    vE::Array{Float64,2}; vW::Array{Float64,2}; vN::Array{Float64,2}; vS::Array{Float64,2}
    KE::Array{Float64,2}; KW::Array{Float64,2}; KN::Array{Float64,2}; KS::Array{Float64,2}
    LE::Array{Float64,2}; LW::Array{Float64,2}; LN::Array{Float64,2}; LS::Array{Float64,2}
    hE::Array{Float64,2}; hW::Array{Float64,2}; hN::Array{Float64,2}; hS::Array{Float64,2}
    F::Array{Float64,3}
    G::Array{Float64,3}
    SB::Array{Float64,3}
    SC::Array{Float64,3}
    dq::Array{Float64,3}
end

function DebugState()
    return DebugState(0, 0, 
        zeros(0,0), zeros(0,0), zeros(0,0),
        zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0),
        zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0),
        zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0),
        zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0),
        zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0),
        zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0),
        zeros(3,0,0), zeros(3,0,0), zeros(3,0,0), zeros(3,0,0), zeros(3,0,0))
end

# =========================
# Debug Versions of Key Functions
# =========================
function debug_build_velocities(x, y, h, hu, hv, Hmin, debug::DebugState)
    println("=== build_velocities ===")
    u, v = RotSW_CDKLM.build_velocities(x, y, h, hu, hv, Hmin)
    
    debug.u = copy(u)
    debug.v = copy(v)
    
    println("  u range: $(minimum(u)) to $(maximum(u))")
    println("  v range: $(minimum(v)) to $(maximum(v))")
    println("  Cells with u < 0.05: $(count(x -> x < 0.05, u))")
    println("  Cells with v < 0.05: $(count(x -> x < 0.05, v))")
    
    return u, v
end

function debug_build_UV_KL(h, u, v, f, Bc, dx, dy, g, debug::DebugState)
    println("=== build_UV_KL ===")
    Uface, Vface, Uc, Vc, K, L = RotSW_CDKLM.build_UV_KL(h, u, v, f, Bc, dx, dy, g)
    
    debug.Uface = copy(Uface)
    debug.Vface = copy(Vface) 
    debug.K = copy(K)
    debug.L = copy(L)
    
    println("  Uface range: $(minimum(Uface)) to $(maximum(Uface))")
    println("  Vface range: $(minimum(Vface)) to $(maximum(Vface))")
    println("  K range: $(minimum(K)) to $(maximum(K))")
    println("  L range: $(minimum(L)) to $(maximum(L))")
    
    # Check theoretical values for constant flow
    if maximum(abs.(u .- 0.1)) < 1e-10 && maximum(abs.(v)) < 1e-10
        println("  THEORETICAL CHECK for constant flow u=0.1, v=0:")
        println("  Expected K = g*(h + B) = $(g * (h[1,1] + Bc[1,1]))")
        println("  Expected L = g*(h + B) = $(g * (h[1,1] + Bc[1,1]))")
        println("  Actual K error: $(maximum(abs.(K .- g*(h .+ Bc))))")
        println("  Actual L error: $(maximum(abs.(L .- g*(h .+ Bc))))")
    end
    
    return Uface, Vface, Uc, Vc, K, L
end

function debug_reconstruct_p(u, v, K, L; limiter=:minmod, bc=:reflective, debug::DebugState)
    println("=== reconstruct_p ===")
    uE,uW,uN,uS, vE,vW,vN,vS, KE,KW,KN,KS, LE,LW,LN,LS = 
        RotSW_CDKLM.reconstruct_p(u, v, K, L; limiter=limiter, bc=bc)
    
    # Store all reconstructed values
    debug.uE = copy(uE); debug.uW = copy(uW); debug.uN = copy(uN); debug.uS = copy(uS)
    debug.vE = copy(vE); debug.vW = copy(vW); debug.vN = copy(vN); debug.vS = copy(vS)
    debug.KE = copy(KE); debug.KW = copy(KW); debug.KN = copy(KN); debug.KS = copy(KS)
    debug.LE = copy(LE); debug.LW = copy(LW); debug.LN = copy(LN); debug.LS = copy(LS)
    
    println("  uE range: $(minimum(uE)) to $(maximum(uE))")
    println("  uW range: $(minimum(uW)) to $(maximum(uW))")
    println("  vE range: $(minimum(vE)) to $(maximum(vE))")
    println("  vW range: $(minimum(vW)) to $(maximum(vW))")
    println("  KE range: $(minimum(KE)) to $(maximum(KE))")
    println("  KW range: $(minimum(KW)) to $(maximum(KW))")
    
    return uE,uW,uN,uS, vE,vW,vN,vS, KE,KW,KN,KS, LE,LW,LN,LS
end

function debug_reconstruct_h(h, Uf, Vf, KE, KW, LN, LS, Bfx, Bfy, g, debug::DebugState)
    println("=== reconstruct_h ===")
    hE, hW, hN, hS = RotSW_CDKLM.reconstruct_h(h, Uf, Vf, KE, KW, LN, LS, Bfx, Bfy, g)
    
    debug.hE = copy(hE); debug.hW = copy(hW); debug.hN = copy(hN); debug.hS = copy(hS)
    
    println("  hE range: $(minimum(hE)) to $(maximum(hE))")
    println("  hW range: $(minimum(hW)) to $(maximum(hW))")
    println("  hN range: $(minimum(hN)) to $(maximum(hN))")
    println("  hS range: $(minimum(hS)) to $(maximum(hS))")
    
    return hE, hW, hN, hS
end

function debug_build_F(hE, hW, uE, uW, vE, vW, g, bc, debug::DebugState)
    println("=== build_F ===")
    F = RotSW_CDKLM.build_F(hE, hW, uE, uW, vE, vW, g, bc)
    
    debug.F = copy(F)
    
    println("  F[1] (mass flux) range: $(minimum(F[1,:,:])) to $(maximum(F[1,:,:]))")
    println("  F[2] (x-momentum flux) range: $(minimum(F[2,:,:])) to $(maximum(F[2,:,:]))")
    println("  F[3] (transverse momentum) range: $(minimum(F[3,:,:])) to $(maximum(F[3,:,:]))")
    
    # Check boundary fluxes
    println("  Left boundary F[1,1,:] = $(F[1,1,1:3])...")
    println("  Right boundary F[1,end,:] = $(F[1,end,1:3])...")
    
    return F
end

function debug_build_G(hN, hS, uN, uS, vN, vS, g, bc, debug::DebugState)
    println("=== build_G ===")
    G = RotSW_CDKLM.build_G(hN, hS, uN, uS, vN, vS, g, bc)
    
    debug.G = copy(G)
    
    println("  G[1] (mass flux) range: $(minimum(G[1,:,:])) to $(maximum(G[1,:,:]))")
    println("  G[2] (transverse momentum) range: $(minimum(G[2,:,:])) to $(maximum(G[2,:,:]))")
    println("  G[3] (y-momentum flux) range: $(minimum(G[3,:,:])) to $(maximum(G[3,:,:]))")
    
    return G
end

function debug_build_S_B(h, Bfx, Bfy, g, dx, dy, debug::DebugState)
    println("=== build_S_B ===")
    SB = RotSW_CDKLM.build_S_B(h, Bfx, Bfy, g, dx, dy)
    
    debug.SB = copy(SB)
    
    println("  SB[2] (x-source) range: $(minimum(SB[2,:,:])) to $(maximum(SB[2,:,:]))")
    println("  SB[3] (y-source) range: $(minimum(SB[3,:,:])) to $(maximum(SB[3,:,:]))")
    
    return SB
end

function debug_build_S_C(h, u, v, f, debug::DebugState)
    println("=== build_S_C ===")
    SC = RotSW_CDKLM.build_S_C(h, u, v, f)
    
    debug.SC = copy(SC)
    
    println("  SC[2] (Coriolis x) range: $(minimum(SC[2,:,:])) to $(maximum(SC[2,:,:]))")
    println("  SC[3] (Coriolis y) range: $(minimum(SC[3,:,:])) to $(maximum(SC[3,:,:]))")
    
    # Check if Coriolis terms are balanced
    total_SC2 = sum(SC[2,:,:])
    total_SC3 = sum(SC[3,:,:])
    println("  Total SC[2] (sum over domain): $total_SC2")
    println("  Total SC[3] (sum over domain): $total_SC3")
    
    return SC
end

function debug_residual!(st, p, debug::DebugState)
    println("=== residual! ===")
    
    q = st.q
    dq = st.dq
    _, Nx, Ny = size(q)
    h  = @view q[1, :, :]
    hu = @view q[2, :, :]
    hv = @view q[3, :, :]
    
    debug.h = copy(h)
    
    # 1) velocities
    u, v = debug_build_velocities(p.x, p.y, h, hu, hv, p.Hmin, debug)
    
    # 2) UVKL
    Uface, Vface, Uc, Vc, K, L = debug_build_UV_KL(h, u, v, st.f, st.Bc, p.dx, p.dy, p.g, debug)
    
    # 3) reconstruct p = (u,v,K,L)
    uE,uW,uN,uS, vE,vW,vN,vS, KE,KW,KN,KS, LE,LW,LN,LS = 
        debug_reconstruct_p(u, v, K, L; limiter=p.limiter, bc=p.bc, debug=debug)
    
    # 4) reconstruct h
    hE,hW,hN,hS = debug_reconstruct_h(h, Uface, Vface, KE, KW, LN, LS, st.Bfx, st.Bfy, p.g, debug)
    
    # 5) Fluxes and sources
    st.F .= debug_build_F(hE, hW, uE, uW, vE, vW, p.g, p.bc, debug)
    st.G .= debug_build_G(hN, hS, uN, uS, vN, vS, p.g, p.bc, debug)
    st.SB .= debug_build_S_B(h, st.Bfx, st.Bfy, p.g, p.dx, p.dy, debug)
    st.SC .= debug_build_S_C(h, u, v, st.f, debug)
    
    F = st.F; G = st.G; SB = st.SB; SC = st.SC
    
    # 6) Compute residuals
    @inbounds for i in 1:Nx, j in 1:Ny
        dF1 = (F[1,i+1,j] - F[1,i,j]) / p.dx
        dF2 = (F[2,i+1,j] - F[2,i,j]) / p.dx
        dF3 = (F[3,i+1,j] - F[3,i,j]) / p.dx
        
        dG1 = (G[1,i,j+1] - G[1,i,j]) / p.dy
        dG2 = (G[2,i,j+1] - G[2,i,j]) / p.dy
        dG3 = (G[3,i,j+1] - G[3,i,j]) / p.dy
        
        dq[1,i,j]  = -dF1 - dG1 + SB[1,i,j] + SC[1,i,j]
        dq[2,i,j] = -dF2 - dG2 + SB[2,i,j] + SC[2,i,j]
        dq[3,i,j] = -dF3 - dG3 + SB[3,i,j] + SC[3,i,j]
    end
    
    debug.dq = copy(dq)
    
    println("  dq[1] (h residual) range: $(minimum(dq[1,:,:])) to $(maximum(dq[1,:,:]))")
    println("  dq[2] (hu residual) range: $(minimum(dq[2,:,:])) to $(maximum(dq[2,:,:]))")
    println("  dq[3] (hv residual) range: $(minimum(dq[3,:,:])) to $(maximum(dq[3,:,:]))")
    
    return nothing
end

function debug_step_RK2!(st, p, step::Int, debug::DebugState)
    println("\n" * "="^50)
    println("STEP $step - RK2 Stage 1")
    println("="^50)
    
    debug.step = step
    debug.stage = 1
    
    q = st.q
    dq = st.dq
    q1 = st.q_stage
    
    # Stage 1: q¹ = qⁿ + dt * R(qⁿ)
    debug_residual!(st, p, debug)
    @. q1 = q + p.dt * dq
    
    println("\n" * "="^50)
    println("STEP $step - RK2 Stage 2")
    println("="^50)
    
    debug.stage = 2
    
    # Temporarily swap q → q1 to compute R(q¹)
    q_orig = st.q
    st.q = q1
    debug_residual!(st, p, debug)
    st.q = q_orig
    
    # Stage 2: qⁿ⁺¹ = ½ ( qⁿ + q¹ + dt * R(q¹) )
    @. q = 0.5 * (q + q1 + p.dt * dq)
    
    # Sync scalar fields from q
    @views begin
        st.h  .= q[1, :, :]
        st.hu .= q[2, :, :]
        st.hv .= q[3, :, :]
    end
    
    return debug
end

# =========================
# Visualization Functions
# =========================
function plot_debug_step(debug::DebugState, p::Params, step::Int)
    xs = p.x
    ys = p.y
    
    # Create a comprehensive dashboard of plots
    plots = []
    
    # 1. Velocities
    if !isempty(debug.u)
        p1 = heatmap(xs, ys, debug.u', title="u velocity - Step $step", aspect_ratio=:equal)
        p2 = heatmap(xs, ys, debug.v', title="v velocity - Step $step", aspect_ratio=:equal)
        push!(plots, p1, p2)
    end
    
    # 2. K and L variables
    if !isempty(debug.K)
        p3 = heatmap(xs, ys, debug.K', title="K - Step $step", aspect_ratio=:equal)
        p4 = heatmap(xs, ys, debug.L', title="L - Step $step", aspect_ratio=:equal)
        push!(plots, p3, p4)
    end
    
    # 3. Reconstructed h at interfaces
    if !isempty(debug.hE)
        p5 = heatmap(xs, ys, debug.hE', title="hE - Step $step", aspect_ratio=:equal)
        p6 = heatmap(xs, ys, debug.hW', title="hW - Step $step", aspect_ratio=:equal)
        push!(plots, p5, p6)
    end
    
    # 4. Fluxes
    if !isempty(debug.F) && size(debug.F, 2) > 0
        p7 = heatmap(xs, 1:size(debug.F,3), debug.F[2,:,:]', title="F[2] (x-momentum flux)", aspect_ratio=:equal)
        p8 = heatmap(1:size(debug.G,2), ys, debug.G[3,:,:]', title="G[3] (y-momentum flux)", aspect_ratio=:equal)
        push!(plots, p7, p8)
    end
    
    # 5. Source terms
    if !isempty(debug.SC)
        p9 = heatmap(xs, ys, debug.SC[2,:,:]', title="SC[2] (Coriolis x-source)", aspect_ratio=:equal)
        p10 = heatmap(xs, ys, debug.SC[3,:,:]', title="SC[3] (Coriolis y-source)", aspect_ratio=:equal)
        push!(plots, p9, p10)
    end
    
    # 6. Residuals
    if !isempty(debug.dq)
        p11 = heatmap(xs, ys, debug.dq[2,:,:]', title="dq[2] (hu residual)", aspect_ratio=:equal)
        p12 = heatmap(xs, ys, debug.dq[3,:,:]', title="dq[3] (hv residual)", aspect_ratio=:equal)
        push!(plots, p11, p12)
    end
    
    # Arrange plots in a grid
    ncols = 2
    nrows = ceil(Int, length(plots) / ncols)
    layout = @layout [a b; c d; e f; g h; i j; k l][1:length(plots)]
    
    plot(plots..., layout=layout, size=(1200, 1800))
end

function analyze_balance(debug::DebugState, p::Params)
    println("\n" * "="^60)
    println("BALANCE ANALYSIS")
    println("="^60)
    
    if !isempty(debug.SC) && !isempty(debug.dq)
        # Check Coriolis source balance
        total_SC2 = sum(debug.SC[2,:,:])
        total_SC3 = sum(debug.SC[3,:,:])
        println("Coriolis source totals:")
        println("  SC[2] (x-direction): $total_SC2")
        println("  SC[3] (y-direction): $total_SC3")
        
        # Check if residuals are zero (for constant flow)
        max_dq2 = maximum(abs.(debug.dq[2,:,:]))
        max_dq3 = maximum(abs.(debug.dq[3,:,:]))
        println("Maximum residuals:")
        println("  dq[2] (hu): $max_dq2")
        println("  dq[3] (hv): $max_dq3")
        
        # Check UVKL consistency
        if !isempty(debug.K) && !isempty(debug.L)
            K_expected = p.g .* (debug.h .+ p.g*0)  # Assuming flat bottom
            K_error = maximum(abs.(debug.K .- K_expected))
            L_error = maximum(abs.(debug.L .- K_expected))
            println("UVKL consistency errors:")
            println("  K error: $K_error")
            println("  L error: $L_error")
        end
    end
end

# =========================
# Main Debug Function
# =========================
function run_debug_simulation(;steps=5, plot_every=1)
    # Set up the same test case
    limiter = :minmod
    nx, ny  = 20, 20  # Smaller for faster debugging
    dx, dy  = 1000, 1000
    g       = 9.81
    dt      = 0.1
    bc      = :periodic
    Hmin    = 1e-3
    
    # With Coriolis
    f0   = 1e-4  # Typical Coriolis parameter
    beta = 0.0
    bfun(x,y) = 0.0
    
    x = collect(range(dx/2, step=dx, length=nx))
    y = collect(range(dy/2, step=dy, length=ny))
    
    st, p = RotSW_CDKLM.init_state(x, y, bfun, f0, beta; g=g, dt=dt, Hmin=Hmin, limiter=limiter, bc=bc)
    
    # Initialize constant flow
    h_const = 10.0
    u_const = 0.1  
    v_const = 0.0
    
    @inbounds for i in 1:nx, j in 1:ny
        st.h[i,j] = h_const
        st.hu[i,j] = h_const * u_const
        st.hv[i,j] = h_const * v_const
    end
    
    @views begin
        st.q[1,:,:] .= st.h
        st.q[2,:,:] .= st.hu
        st.q[3,:,:] .= st.hv
    end
    
    # Initialize debug state
    debug_history = []
    
    println("Starting debug simulation with Coriolis")
    println("Initial conditions: h=$h_const, u=$u_const, v=$v_const, f0=$f0")
    
    for step in 1:steps
        debug = DebugState()
        debug_step_RK2!(st, p, step, debug)
        push!(debug_history, debug)
        
        analyze_balance(debug, p)
        
        if step % plot_every == 0
            plt = plot_debug_step(debug, p, step)
            display(plt)
            savefig(plt, "debug_step_$step.png")
        end
        
        # Check for problems
        u_current, v_current = RotSW_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
        min_u = minimum(u_current)
        min_v = minimum(v_current)
        println("After step $step: min(u)=$min_u, min(v)=$min_v")
        
        if min_u < 1e-3 || min_v < 1e-3
            println("WARNING: Velocities approaching zero!")
        end
    end
    
    return st, p, debug_history
end

# Run the debug simulation
if abspath(PROGRAM_FILE) == @__FILE__
    st, p, debug_history = run_debug_simulation(steps=3, plot_every=1)
end


run_debug_simulation(;steps=5, plot_every=1)