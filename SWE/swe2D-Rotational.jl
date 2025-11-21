cd(@__DIR__)
include("sweSim2D-Rotational.jl")
import .RotSW_CDKLM
using Plots; theme(:default)

limiter = :minmod
steps   = 100
nx, ny  = 100, 100
dx, dy  = 10000, 10000
g       = 9.81
dt      = 0.1
bc      = :periodic
Hmin    = 1e-3

# --- Coriolis: β–plane -------------------------------------
f0   = 1e-4         # base Coriolis parameter (s⁻¹)
beta = 2e-6      

x = collect(range(0, step=dx, length=nx))
y = collect(range(0, step=dy, length=ny))

# --- Bathymetry ---------------------------------------------
# Example: a smooth Gaussian bump in the middle (in meters)
x0 = x[end]/2
y0 = y[end]/2
Lx = x[end]/3
Ly = y[end]/3

bfun(x,y) = 2.0 * exp(-((x - x0)^2 / Lx^2 + (y - y0)^2 / Ly^2))   # NEW

st, p = RotSW_CDKLM.init_state(x, y, bfun, f0, beta;
                               g=g, dt=dt, Hmin=Hmin,
                               limiter=limiter, bc=bc)

# --- Lake at rest IC: w = const ------------------------------
w0 = 10.0                      # equilibrium free surface level (m)  # NEW

# depth h = w0 - B(x,y)
st.h  .= w0 .- st.Bc           # NEW: lake-at-rest with topography
st.hu .= 0.0
st.hv .= 0.0

# sync q
@views begin
    st.q[1,:,:] .= st.h
    st.q[2,:,:] .= st.hu
    st.q[3,:,:] .= st.hv
end

# --- diagnostics helpers ---
function total_mass(st, p)
    p.dx * p.dy * sum(st.h)
end

function potential_energy(st, p; w0)
    dx, dy = p.dx, p.dy
    w = st.h .+ st.Bc         # free surface
    η = w .- w0
    0.5 * p.g * dx * dy * sum(η.^2)
end

println("Testing lake-at-rest initial state, w0 = $w0")
M0  = total_mass(st, p)
PE0 = potential_energy(st, p; w0=w0)
println("Initial: mass = $M0, PE = $PE0")
println("Coriolis: f ∈ [$(minimum(st.f)), $(maximum(st.f))]")

for n in 1:steps
    RotSW_CDKLM.step_RK2!(st, p)
    M  = total_mass(st, p)
    PE = potential_energy(st, p; w0=w0)
    println("step $n: hmin = $(minimum(st.h)), hmax = $(maximum(st.h)), " *
            "mass = $M (ΔM = $(M - M0)), PE = $PE")
end

# ======================================================
# Plots: bathymetry + free surface
# ======================================================

xs = p.x
ys = p.y

# Current free surface and perturbation
w = st.h .+ st.Bc
η = w .- w0

# 1) 2D contour of bathymetry
pltB = contourf(xs, ys, permutedims(st.Bc);
                aspect_ratio = :equal,
                xlabel = "x (m)", ylabel = "y (m)",
                title  = "Bathymetry B(x,y)",
                colorbar = true)

# 2) 2D contour of free-surface perturbation
pltEta = contourf(xs, ys, permutedims(η);
                  aspect_ratio = :equal,
                  xlabel = "x (m)", ylabel = "y (m)",
                  title  = "Free-surface perturbation η at t = $(steps*dt) s",
                  colorbar = true)

display(pltB)
display(pltEta)

# 3) 1D cross-section: bottom + free surface in the same plot (optional)
jmid = Int(cld(ny, 2))   # mid row in y
pltSection = plot(xs, st.Bc[:, jmid], label="bottom B(x, y_mid)", lw=2)
plot!(xs, w[:, jmid],    label="free surface w(x, y_mid)", lw=2,
      xlabel="x (m)", ylabel="z (m)",
      title="Cross-section at y = $(ys[jmid]) m")
display(pltSection)
