cd(@__DIR__)
include("x_test_boundaries.jl")
import .test_CDKLM
using Plots; theme(:default)


# Test constant flow with periodic BCs and zero bathymetry/Coriolis
limiter = :minmod
steps   = 1000
nx, ny  = 100, 100
dx, dy  = 1000, 1000
g       = 9.81
dt      = 0.1
bc      = :periodic
Hmin    = 1e-3

# Zero Coriolis and flat bottom
f0   = 0.0
beta = 0.0
x0 = x[end]/2
y0 = y[end]/2
Lx = x[end]/8
Ly = y[end]/8

bfun(x,y) = 2.0 * exp(-((x - x0)^2 / Lx^2 + (y - y0)^2 / Ly^2))


# Use cell centers that cover the entire domain properly
x = collect(range(dx/2, step=dx, length=nx))
y = collect(range(dy/2, step=dy, length=ny))

st, p = test_CDKLM.init_state(x, y, bfun, f0, beta; g=g, dt=dt, Hmin=Hmin, limiter=limiter, bc=bc)

# --- Constant flow to the right ---
h_const = 10.0
u_const = 0.1  
v_const = 0.0

# Initialize ALL physical cells including edges
println("Initializing state...")
@inbounds for i in 1:nx, j in 1:ny
    st.h[i,j] = h_const
    st.hu[i,j] = h_const * u_const
    st.hv[i,j] = h_const * v_const
end

# sync q - make sure ALL cells are set
@views begin
    st.q[1,:,:] .= st.h
    st.q[2,:,:] .= st.hu
    st.q[3,:,:] .= st.hv
end

# --- Enhanced diagnostics ---
function total_mass(st, p)
    p.dx * p.dy * sum(st.h)
end

function max_velocity(st, p)
    u, v = test_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
    max_u = maximum(abs.(u))
    max_v = maximum(abs.(v))
    return max_u, max_v
end

function min_velocity(st, p)
    u, v = test_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
    min_u = minimum(abs.(u))
    min_v = minimum(abs.(v))
    return min_u, min_v
end

function velocity_error(st, p, u_ref, v_ref)
    u, v = test_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
    err_u = maximum(abs.(u .- u_ref))
    err_v = maximum(abs.(v .- v_ref))
    return err_u, err_v
end

function check_edge_cells(st, p, u_ref, v_ref)
    u, v = test_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
    println("Edge cell check:")
    println("  Top-left corner (1,1): u = ", u[1,1], " (should be ", u_ref, ")")
    println("  Top-right corner (nx,1): u = ", u[p.nx,1], " (should be ", u_ref, ")")
    println("  Bottom-left corner (1,ny): u = ", u[1,p.ny], " (should be ", u_ref, ")")
    println("  Bottom-right corner (nx,ny): u = ", u[p.nx,p.ny], " (should be ", u_ref, ")")
    
    # Check if any edge cells are zero
    edge_zeros_u = 0
    edge_zeros_v = 0
    @inbounds for i in 1:p.nx
        if abs(u[i,1]) < 1e-10; edge_zeros_u += 1; end
        if abs(u[i,p.ny]) < 1e-10; edge_zeros_u += 1; end
        if abs(v[i,1]) < 1e-10; edge_zeros_v += 1; end
        if abs(v[i,p.ny]) < 1e-10; edge_zeros_v += 1; end
    end
    @inbounds for j in 1:p.ny
        if abs(u[1,j]) < 1e-10; edge_zeros_u += 1; end
        if abs(u[p.nx,j]) < 1e-10; edge_zeros_u += 1; end
        if abs(v[1,j]) < 1e-10; edge_zeros_v += 1; end
        if abs(v[p.nx,j]) < 1e-10; edge_zeros_v += 1; end
    end
    println("  Edge cells with near-zero u: ", edge_zeros_u)
    println("  Edge cells with near-zero v: ", edge_zeros_v)
end

println("Testing constant flow with periodic BCs")
println("Initial conditions: h = $h_const, u = $u_const, v = $v_const")
M0 = total_mass(st, p)
max_u0, max_v0 = max_velocity(st, p)
min_u0, min_v0 = min_velocity(st, p)
err_u0, err_v0 = velocity_error(st, p, u_const, v_const)
println("Initial: mass = $M0, max |u| = $max_u0, min |u| = $min_u0, max |v| = $max_v0, min |v| = $min_v0")
println("Initial velocity errors: u = $err_u0, v = $err_v0")
check_edge_cells(st, p, u_const, v_const)

# Track evolution
mass_evolution = Float64[]
u_error_evolution = Float64[]
v_error_evolution = Float64[]
min_u_evolution = Float64[]
min_v_evolution = Float64[]

push!(mass_evolution, M0)
push!(u_error_evolution, err_u0)
push!(v_error_evolution, err_v0)
push!(min_u_evolution, min_u0)
push!(min_v_evolution, min_v0)

for n in 1:steps
    test_CDKLM.step_RK2!(st, p)
    M  = total_mass(st, p)
    max_u, max_v = max_velocity(st, p)
    min_u, min_v = min_velocity(st, p)
    err_u, err_v = velocity_error(st, p, u_const, v_const)
    
    push!(mass_evolution, M)
    push!(u_error_evolution, err_u)
    push!(v_error_evolution, err_v)
    push!(min_u_evolution, min_u)
    push!(min_v_evolution, min_v)
    
    if n % 10 == 0
        println("step $n: mass = $M (ΔM = $(M - M0))")
        println("         max |u| = $max_u, min |u| = $min_u, max |v| = $max_v, min |v| = $min_v")
        println("         velocity errors: u = $err_u, v = $err_v")
        check_edge_cells(st, p, u_const, v_const)
    end
    
    # Check for NaNs
    if any(isnan.(st.h)) || any(isnan.(st.hu)) || any(isnan.(st.hv))
        println("NaN detected at step $n! Stopping.")
        break
    end
end

# Check momentum conservation specifically
function total_momentum(st, p)
    p.dx * p.dy * (sum(st.hu), sum(st.hv))
end

function momentum_error(st, p, hu_ref, hv_ref)
    err_hu = maximum(abs.(st.hu .- hu_ref))
    err_hv = maximum(abs.(st.hv .- hv_ref))
    return err_hu, err_hv
end

println("\n=== DETAILED DEBUG ANALYSIS ===")

# Check initial momentum
hu_ref = h_const * u_const
hv_ref = h_const * v_const
M0_hu, M0_hv = total_momentum(st, p)
println("Initial total hu = $M0_hu (should be $(hu_ref * p.dx * p.dy * nx * ny))")
println("Initial total hv = $M0_hv (should be $(hv_ref * p.dx * p.dy * nx * ny))")

# Check current momentum
M_hu, M_hv = total_momentum(st, p)
err_hu, err_hv = momentum_error(st, p, hu_ref, hv_ref)
println("Final total hu = $M_hu (error = $(M_hu - M0_hu))")
println("Final total hv = $M_hv (error = $(M_hv - M0_hv))")
println("Max momentum errors: hu = $err_hu, hv = $err_hv")

# Check specific problem areas
u_current, v_current = test_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
println("\nMinimum velocities in final state:")
println("min(u) = $(minimum(u_current)), max(u) = $(maximum(u_current))")
println("min(v) = $(minimum(v_current)), max(v) = $(maximum(v_current))")

# Find where u is becoming small
small_u_threshold = 0.05  # Half of initial u
small_u_cells = findall(x -> x < small_u_threshold, u_current)
println("Cells with u < $small_u_threshold: $(length(small_u_cells))")

if length(small_u_cells) > 0
    println("First 10 small-u cells:")
    for i in 1:min(10, length(small_u_cells))
        idx = small_u_cells[i]
        println("  Cell ($(idx[1]), $(idx[2])): u = $(u_current[idx]), hu = $(st.hu[idx]), h = $(st.h[idx])")
    end
end

# Check momentum conservation specifically
function total_momentum(st, p)
    p.dx * p.dy * (sum(st.hu), sum(st.hv))
end

function momentum_error(st, p, hu_ref, hv_ref)
    err_hu = maximum(abs.(st.hu .- hu_ref))
    err_hv = maximum(abs.(st.hv .- hv_ref))
    return err_hu, err_hv
end

println("\n=== DETAILED DEBUG ANALYSIS ===")

# Check initial momentum
hu_ref = h_const * u_const
hv_ref = h_const * v_const
M0_hu, M0_hv = total_momentum(st, p)
println("Initial total hu = $M0_hu (should be $(hu_ref * p.dx * p.dy * nx * ny))")
println("Initial total hv = $M0_hv (should be $(hv_ref * p.dx * p.dy * nx * ny))")

# Check current momentum
M_hu, M_hv = total_momentum(st, p)
err_hu, err_hv = momentum_error(st, p, hu_ref, hv_ref)
println("Final total hu = $M_hu (error = $(M_hu - M0_hu))")
println("Final total hv = $M_hv (error = $(M_hv - M0_hv))")
println("Max momentum errors: hu = $err_hu, hv = $err_hv")

# Check specific problem areas
u_current, v_current = test_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
println("\nMinimum velocities in final state:")
println("min(u) = $(minimum(u_current)), max(u) = $(maximum(u_current))")
println("min(v) = $(minimum(v_current)), max(v) = $(maximum(v_current))")

# Find where u is becoming small
small_u_threshold = 0.05  # Half of initial u
small_u_cells = findall(x -> x < small_u_threshold, u_current)
println("Cells with u < $small_u_threshold: $(length(small_u_cells))")

if length(small_u_cells) > 0
    println("First 10 small-u cells:")
    for i in 1:min(10, length(small_u_cells))
        idx = small_u_cells[i]
        println("  Cell ($(idx[1]), $(idx[2])): u = $(u_current[idx]), hu = $(st.hu[idx]), h = $(st.h[idx])")
    end
end

# ======================================================
# Plots: PHYSICAL CELLS ONLY
# ======================================================
println("\nFlux consistency check:")
for j in 1:min(3, ny)  # Check first few rows
    interior_flux = st.F[2, nx÷2, j]  # Middle of domain
    left_flux = st.F[2, 1, j]
    right_flux = st.F[2, nx+1, j]
    println("Row $j: interior=$interior_flux, left=$left_flux, right=$right_flux")
end
# Check if source terms are truly zero
max_SB = maximum(abs.(st.SB))
max_SC = maximum(abs.(st.SC))
println("\nSource term analysis:")
println("Max |SB| = $max_SB")
println("Max |SC| = $max_SC")

if max_SB > 1e-10
    println("WARNING: Non-zero bathymetry source terms!")
end
if max_SC > 1e-10  
    println("WARNING: Non-zero Coriolis source terms!")
end

# Debug the K and L reconstruction
u_debug, v_debug = test_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
Uface, Vface, Uc, Vc, K, L = test_CDKLM.build_UV_KL(st.h, u_debug, v_debug, st.f, st.Bc, p.dx, p.dy, p.g, p.bc)

println("\nK and L analysis:")
println("K range: $(minimum(K)) to $(maximum(K))")
println("L range: $(minimum(L)) to $(maximum(L))")
println("Theoretical K = g*h = $(p.g * h_const)")

# Check if K and L are consistent
K_error = maximum(abs.(K .- (p.g * h_const)))
L_error = maximum(abs.(L .- (p.g * h_const)))  
println("K error from theoretical: $K_error")
println("L error from theoretical: $L_error")
xs = p.x
ys = p.y

# Current velocity field
u, v = test_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
speed = @. sqrt(u^2 + v^2)
u_error = @. abs(u - u_const)
v_error = @. abs(v - v_const)
println("Actual speed values at corners:")
println("  (1,1): ", speed[1,1])
println("  (1,ny): ", speed[1,end])  
println("  (nx,1): ", speed[end,1])
println("  (nx,ny): ", speed[end,end])
# 1) Velocity magnitude
pltVel = contourf(xs, ys, permutedims(speed);
                 aspect_ratio = :equal,
                 xlabel = "x (m)", ylabel = "y (m)", 
                 title = "Velocity magnitude at t = $(steps*dt) s",
                 colorbar = true)

# 2) Velocity errors
pltUError = contourf(xs, ys, permutedims(u_error);
                    aspect_ratio = :equal,
                    xlabel = "x (m)", ylabel = "y (m)",
                    title = "u-velocity error",
                    colorbar = true)

pltVError = contourf(xs, ys, permutedims(v_error);
                    aspect_ratio = :equal, 
                    xlabel = "x (m)", ylabel = "y (m)",
                    title = "v-velocity error", 
                    colorbar = true)

# 3) Conservation plots
pltMass = plot(0:steps, mass_evolution .- M0, 
               xlabel="Time step", ylabel="Mass error (kg)",
               title="Mass conservation", legend=false)
pltUErrorEvol = plot(0:steps, u_error_evolution,
                    xlabel="Time step", ylabel="Max u-error (m/s)",
                    title="U-velocity error evolution", legend=false)
pltVErrorEvol = plot(0:steps, v_error_evolution,
                    xlabel="Time step", ylabel="Max v-error (m/s)", 
                    title="V-velocity error evolution", legend=false)
pltMinU = plot(0:steps, min_u_evolution,
              xlabel="Time step", ylabel="Min |u| (m/s)",
              title="Minimum u-velocity", legend=false)
pltMinV = plot(0:steps, min_v_evolution,
              xlabel="Time step", ylabel="Min |v| (m/s)",
              title="Minimum v-velocity", legend=false)

# Display plots
display(pltVel)
display(plot(pltUError, pltVError, layout=(1,2), size=(800,400)))
display(plot(pltMass, pltUErrorEvol, pltVErrorEvol, layout=(3,1), size=(800,600)))
display(plot(pltMinU, pltMinV, layout=(1,2), size=(800,400)))

# Create output directory
mkpath("Plots_CDKLM")

# Save plots
savefig(pltVel, "Plots_CDKLM/constant_flow_velocity.png")
savefig(pltUError, "Plots_CDKLM/constant_flow_u_error.png")
savefig(pltVError, "Plots_CDKLM/constant_flow_v_error.png")
savefig(plot(pltMass, pltUErrorEvol, pltVErrorEvol, layout=(3,1), size=(800,600)),
        "Plots_CDKLM/constant_flow_conservation.png")
savefig(plot(pltMinU, pltMinV, layout=(1,2), size=(800,400)),
        "Plots_CDKLM/constant_flow_min_velocity.png")

println("\nSimulation completed!")
println("Final mass error: $(mass_evolution[end] - M0)")
println("Final velocity errors: u = $(u_error_evolution[end]), v = $(v_error_evolution[end])")
println("Final min velocities: u = $(min_u_evolution[end]), v = $(min_v_evolution[end])")