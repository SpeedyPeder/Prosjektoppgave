cd(@__DIR__)
include("x_test_boundaries.jl")
import .test_CDKLM
using Plots; theme(:default)

# ---------------------------------------
# Parameters
# ---------------------------------------
limiter = :minmod
steps   = 1000
nx, ny  = 100, 100
dx, dy  = 1000.0, 1000.0
g       = 9.81
dt      = 0.1
bc      = :periodic
Hmin    = 1e-3

# Zero Coriolis and FLAT bottom
f0   = 0.0
beta = 0.0

# Cell centers covering the domain
x = collect(range(dx/2, step=dx, length=nx))
y = collect(range(dy/2, step=dy, length=ny))

# Flat bathymetry for this test
bfun(x,y) = 0.0

# ---------------------------------------
# Initialise state
# ---------------------------------------
st, p = test_CDKLM.init_state(x, y, bfun, f0, beta;
                              g=g, dt=dt, Hmin=Hmin,
                              limiter=limiter, bc=bc)

# Constant flow state
h_const = 10.0
u_const = 0.5
v_const = 0.0

@inbounds for i in 1:nx, j in 1:ny
    st.h[i,j]  = h_const
    st.hu[i,j] = h_const * u_const
    st.hv[i,j] = h_const * v_const
end

@views begin
    st.q[1,:,:] .= st.h
    st.q[2,:,:] .= st.hu
    st.q[3,:,:] .= st.hv
end

# ---------------------------------------
# Helper functions
# ---------------------------------------
total_mass(st, p) = p.dx * p.dy * sum(st.h)

function velocity_error(st, p, u_ref, v_ref)
    u, v = test_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
    err_u = maximum(abs.(u .- u_ref))
    err_v = maximum(abs.(v .- v_ref))
    return err_u, err_v
end

function min_velocity(st, p)
    u, v = test_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
    return minimum(abs.(u)), minimum(abs.(v))
end

# ---------------------------------------
# Time stepping + diagnostics
# ---------------------------------------
M0 = total_mass(st, p)
err_u0, err_v0 = velocity_error(st, p, u_const, v_const)
min_u0, min_v0 = min_velocity(st, p)

mass_evolution    = Float64[M0]
u_error_evolution = Float64[err_u0]
v_error_evolution = Float64[err_v0]
min_u_evolution   = Float64[min_u0]
min_v_evolution   = Float64[min_v0]

for n in 1:steps
    test_CDKLM.step_RK2!(st, p)

    M      = total_mass(st, p)
    err_u, err_v = velocity_error(st, p, u_const, v_const)
    min_u, min_v = min_velocity(st, p)

    push!(mass_evolution, M)
    push!(u_error_evolution, err_u)
    push!(v_error_evolution, err_v)
    push!(min_u_evolution, min_u)
    push!(min_v_evolution, min_v)
end

# ---------------------------------------
# Build plots
# ---------------------------------------
xs, ys = p.x, p.y
u, v   = test_CDKLM.build_velocities(p.x, p.y, st.h, st.hu, st.hv, p.Hmin)
speed  = @. sqrt(u^2 + v^2)
u_err  = @. abs(u - u_const)
v_err  = @. abs(v - v_const)

pltVel = contourf(xs, ys, permutedims(speed);
                  aspect_ratio = :equal,
                  xlabel = "x (m)", ylabel = "y (m)",
                  title  = "Velocity magnitude at t = $(steps*dt) s",
                  colorbar = true)

pltUError = contourf(xs, ys, permutedims(u_err);
                     aspect_ratio = :equal,
                     xlabel = "x (m)", ylabel = "y (m)",
                     title  = "u-velocity error",
                     colorbar = true)

pltVError = contourf(xs, ys, permutedims(v_err);
                     aspect_ratio = :equal,
                     xlabel = "x (m)", ylabel = "y (m)",
                     title  = "v-velocity error",
                     colorbar = true)

pltMass = plot(0:steps, mass_evolution .- M0,
               xlabel = "Time step", ylabel = "Mass error",
               title  = "Mass conservation", legend = false)

pltUErrorEvol = plot(0:steps, u_error_evolution,
                     xlabel = "Time step", ylabel = "Max u-error",
                     title  = "u-error evolution", legend = false)

pltVErrorEvol = plot(0:steps, v_error_evolution,
                     xlabel = "Time step", ylabel = "Max v-error",
                     title  = "v-error evolution", legend = false)

pltMinU = plot(0:steps, min_u_evolution,
               xlabel = "Time step", ylabel = "Min |u|",
               title  = "Minimum u-velocity", legend = false)

pltMinV = plot(0:steps, min_v_evolution,
               xlabel = "Time step", ylabel = "Min |v|",
               title  = "Minimum v-velocity", legend = false)

# Show plots
display(pltVel)
display(plot(pltUError, pltVError, layout=(1,2), size=(800,400)))
display(plot(pltMass, pltUErrorEvol, pltVErrorEvol, layout=(3,1), size=(800,600)))
display(plot(pltMinU, pltMinV, layout=(1,2), size=(800,400)))

# Save plots
mkpath("Plots_CDKLM")
savefig(pltVel, "Plots_CDKLM/constant_flow_velocity.png")
savefig(pltUError, "Plots_CDKLM/constant_flow_u_error.png")
savefig(pltVError, "Plots_CDKLM/constant_flow_v_error.png")
savefig(plot(pltMass, pltUErrorEvol, pltVErrorEvol, layout=(3,1), size=(800,600)),
        "Plots_CDKLM/constant_flow_conservation.png")
savefig(plot(pltMinU, pltMinV, layout=(1,2), size=(800,400)),
        "Plots_CDKLM/constant_flow_min_velocity.png")

println("Simulation completed.")
println("Final mass error      = ", mass_evolution[end] - M0)
println("Final max u-error     = ", u_error_evolution[end])
println("Final max v-error     = ", v_error_evolution[end])
println("Final min |u|, min |v| = ", min_u_evolution[end], ", ", min_v_evolution[end])
