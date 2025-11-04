cd(@__DIR__)
include("sweSim2D-Rotational.jl")
using .RotSW_CDKLM
using Plots; theme(:default)

# ---------------- Choose simulation parameters ----------------
limiter = :minmod           # :minmod or :vanalbada
steps   = 1200
nx, ny  = 102, 102
dx, dy  = 500.0, 500.0
g       = 9.81
dt      = 0.5
bc      = :outflow          # :periodic, :reflective, or :outflow

# ---------------- Create grid ----------------
x = range(0, step=dx, length=nx)
y = range(0, step=dy, length=ny)
ix = 2:nx-1;  iy = 2:ny-1
x_int, y_int = x[ix], y[iy]

# Output folder
const PLOTS_DIR = joinpath(@__DIR__, "plots"); mkpath(PLOTS_DIR)

# Compute velocities (interior)
vel_int(st) = begin
    ufull = ifelse.(st.h .> 0, st.qx ./ max.(st.h, 1e-12), 0.0)
    vfull = ifelse.(st.h .> 0, st.qy ./ max.(st.h, 1e-12), 0.0)
    ufull[ix,iy], vfull[ix,iy]
end

# ---------------- Initial condition: Gaussian bump, u=v=0 ----------------
η0 = 10.0
A  = 10.0
L  = 10_000.0
f0 = 1e-4

# Create parameter struct and initialize state
p  = RotSW_CDKLM.Params(nx, ny, dx, dy, g, dt, limiter, bc)
st = RotSW_CDKLM.initialize_state(p.nx, p.ny)
st.f .= f0
st.b .= 0.0
st.qx .= 0.0
st.qy .= 0.0
#Create Gaussian η bump
xc, yc = x[end]/2, y[end]/2
@inbounds for j in eachindex(y), i in eachindex(x)
    r2 = (x[i]-xc)^2 + (y[j]-yc)^2
    st.h[i,j] = η0 + A*exp(-r2/(L^2))   # b=0, thus h=η
end

# Fill halos according to chosen BC
if p.bc === :periodic
    RotSW_CDKLM.set_periodic!(st)
elseif p.bc === :reflective
    RotSW_CDKLM.set_reflective!(st)
elseif p.bc === :outflow
    RotSW_CDKLM.set_outflow!(st)
else
    error("Unknown bc=$(p.bc)")
end
#Printing the initial condition info
println("IC: Gaussian η bump (η0=$(η0), A=$(A), L=$(L)) with u=v=0, b=0, f=$(f0); limiter=$(limiter), bc=$(bc)")

# --------- Save & plot initial η for reference ----------
ηI0 = (st.h .+ st.b)[ix,iy]
plt0 = contourf(x_int, y_int, permutedims(ηI0);
                title = "η (initial) — Gaussian bump, u=v=0",
                xlabel="x (m)", ylabel="y (m)", aspect_ratio=:equal, colorbar=true)
display(plt0)
savefig(plt0, joinpath(PLOTS_DIR, "eta_initial.png"))

# ---------------- Time integration to update state ----------------
for n in 1:steps
    RotSW_CDKLM.step_rk2!(st, p)
end

# ---------------- Final plots, counturf plot with velocity arrows----------------
ηIf = (st.h .+ st.b)[ix,iy]
uI, vI = vel_int(st)
pltη = contourf(x_int, y_int, permutedims(ηIf);
                title = "η (final) — geostrophic adjustment, limiter=$(limiter)",
                xlabel="x (m)", ylabel="y (m)", aspect_ratio=:equal, colorbar=true)
display(pltη)
savefig(pltη, joinpath(PLOTS_DIR, "eta_final.png"))

# Velocity arrows (rescaled), since gravity waves are slow here
skip = 10
xs = x_int[1:skip:end]; ys = y_int[1:skip:end]
U = uI[1:skip:end, 1:skip:end]
V = vI[1:skip:end, 1:skip:end]
X = repeat(xs, 1, length(ys))
Y = repeat(permutedims(ys), length(xs), 1)
speed = sqrt.(U.^2 .+ V.^2)
maxspeed = maximum(speed)
Δ = 0.7 * min(skip*dx, skip*dy)               
scale_factor = Δ / max(maxspeed, 1e-12)
Us = U .* scale_factor
Vs = V .* scale_factor

# Plot with velocity field
pltV = contourf(x_int, y_int, permutedims(ηIf);
                aspect_ratio=:equal, colorbar=true,
                title="Velocity field at t=$(steps*dt) s",
                xlabel="x (m)", ylabel="y (m)")
quiver!(pltV, vec(X), vec(Y), quiver=(vec(Us), vec(Vs));
        scale=:none, arrowsize=0.5, lw=1.0, color=:green, label=false)
display(pltV)
savefig(pltV, joinpath(PLOTS_DIR, "velocity_final.png"))
