cd(@__DIR__)
include("sweSim2D-Rotational.jl")
import .RotSW_CDKLM
using Plots; theme(:default)

limiter = :minmod
steps   = 100000
nx, ny  = 100, 100
dx, dy  = 1000, 1000
g       = 9.81
dt      = 0.1
bc      = :periodic
Hmin    = 1e-3

# --- Coriolis: β–plane -------------------------------------
f0   = 1    # base Coriolis parameter (s⁻¹)
beta = 0     

x = collect(range(0, step=dx, length=nx))
y = collect(range(0, step=dy, length=ny))

# --- Bathymetry ---------------------------------------------
# Example: a smooth Gaussian bump in the middle (in meters)
x0 = x[end]/2
y0 = y[end]/2
Lx = x[end]/2
Ly = y[end]/2

bfun(x,y) = 2*sin(2π * x / Lx)# 


st, p = RotSW_CDKLM.init_state(x, y, bfun, f0, beta; g=g, dt=dt, Hmin=Hmin, limiter=limiter, bc=bc)


# --- Lake at rest IC: w = const ------------------------------
# --- Initial condition: lake-at-rest free surface + uniform velocity ----
w0 = 10.0                      # equilibrium free surface level (m)

# depth h = w0 - B(x,y)
st.h .= w0 .- st.Bc

# choose initial velocities (u in x, v in y)
u0 = 0.5    # m/s, 
v0 = 0.5    # m/s, 

# momentum = h * velocity
st.hu .= st.h .* u0    # x-momentum h*u
st.hv .= st.h .* v0    # y-momentum h*v

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

println("Testing lake-at-rest initial state, w0 = $w0")
M0  = total_mass(st, p)
println("Initial: mass = $M0")
println("Coriolis: f ∈ [$(minimum(st.f)), $(maximum(st.f))]")


for n in 1:steps
    RotSW_CDKLM.step_RK2!(st, p)
    M  = total_mass(st, p)
    println("step $n: wmin = $(minimum(st.h + st.Bc)), wmax = $(maximum(st.h + st.Bc)), mass = $M (ΔM = $(M - M0))")
end

# ======================================================
# Plots: bathymetry + free surface
# ======================================================

xs = p.x
ys = p.y

# Current free surface and perturbation
w = st.h .+ st.Bc
η = w .- w0
v = st.hv ./ st.h
u = st.hu ./ st.h

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

# 4) velocity
pltv = contourf(xs, ys, u;
                aspect_ratio = :equal,
                xlabel = "x (m)", ylabel = "y (m)",
                title  = "Velocity (u,v)",
                colorbar = true)
display(pltv)
print(maximum(u))

mkpath("Plots_CDKLM")   # create folder if it doesn't exist

savefig(pltB,       "Plots_CDKLM/bathymetry.png")
savefig(pltEta,     "Plots_CDKLM/free_surface_η_t$(steps*dt).png")
savefig(pltSection, "Plots_CDKLM/cross_section_t$(steps*dt).png")