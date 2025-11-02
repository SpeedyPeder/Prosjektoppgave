cd(@__DIR__)
include("sweSim2D-Rotational.jl")
using .RotSW_CDKLM
using Plots; theme(:default)

# ---------------- Config (single limiter, single IC) ----------------
limiter = :vanalbada           # choose :minmod or :vanalbada
steps   = 1200                 # long enough to see adjustment
nx, ny  = 102, 102
dx, dy  = 500.0, 500.0
g       = 9.81            # Adjustet for waves moving much faster, gravity (m/s²)
dt      = 0.5
bc      = :outflow         # choose :periodic, :reflective, or :outflow

# grids
x = range(0, step=dx, length=nx)
y = range(0, step=dy, length=ny)

# interior (exclude ghost ring for diagnostics/plots)
ix = 2:nx-1;  iy = 2:ny-1
x_int, y_int = x[ix], y[iy]

# output folder
const PLOTS_DIR = joinpath(@__DIR__, "plots"); mkpath(PLOTS_DIR)

# ---------- helpers ----------
# robust contour of INTERIOR field (avoids ghost frame & NaNs)
function safe_contourf_int(x, y, Z; filename=nothing, title="")
    Zc = copy(Z); Zc[.!isfinite.(Zc)] .= 0.0
    v = vec(Zc[isfinite.(Zc)])
    lo, hi = extrema(v); if !(isfinite(lo)&&isfinite(hi)) || lo==hi; lo,hi=0.0,1.0; end
    plt = contourf(x, y, Zc'; clims=(lo,hi),
                   title=title, xlabel="x (m)", ylabel="y (m)",
                   aspect_ratio=:equal, colorbar=true)
    display(plt); if filename !== nothing; savefig(plt, filename); end
    return plt
end

# compute velocities safely on interior
function vel_int(st)
    ufull = ifelse.(st.h .> 0, st.qx ./ max.(st.h, 1e-12), 0.0)
    vfull = ifelse.(st.h .> 0, st.qy ./ max.(st.h, 1e-12), 0.0)
    return ufull[ix,iy], vfull[ix,iy]
end

# ---------------- Initial condition: geostrophic adjustment ----------------
# Free surface: Gaussian bump; zero initial velocity; flat bottom; constant f
η0 = 10.0           # background free surface (depth, since b=0)
A  = 10.0          # bump amplitude
L  = 10_000.0       # bump width (meters)
f0 = 1e-4           # Coriolis (1/s)

p  = RotSW_CDKLM.Params(nx, ny, dx, dy, g, dt, limiter, bc)
st = RotSW_CDKLM.initialize_state(p.nx, p.ny)

st.f .= f0
st.b .= 0.0
st.qx .= 0.0
st.qy .= 0.0

xc, yc = x[end]/2, y[end]/2
@inbounds for j in eachindex(y), i in eachindex(x)
    r2 = (x[i]-xc)^2 + (y[j]-yc)^2
    η   = η0 + A*exp(-r2/(L^2))   # not initially balanced with u=v=0
    st.h[i,j] = η                 # (since b=0, h=η)
end

# Fill halos to make ghosts consistent with IC
if p.bc == :periodic
    RotSW_CDKLM.set_periodic!(st)
else
    RotSW_CDKLM.set_reflective!(st)
end

println("IC: Gaussian η bump (η0=$(η0), A=$(A), L=$(L)) with u=v=0, b=0, f=$(f0); limiter=$(limiter)")

# --------- Save & plot initial η (interior) ----------
ηI0 = (st.h .+ st.b)[ix,iy]
safe_contourf_int(x_int, y_int, ηI0;
    filename=joinpath(PLOTS_DIR, "eta_initial.png"),
    title="η (initial) — Gaussian bump, u=v=0")

# ---------------- Time integration ----------------
for n in 1:steps
    RotSW_CDKLM.step_rk2!(st, p)
end

# ---------------- Final plots (only η and velocity) ----------------
ηIf = (st.h .+ st.b)[ix,iy]
uI, vI = vel_int(st)

# 1) Final η
safe_contourf_int(x_int, y_int, ηIf;
    filename=joinpath(PLOTS_DIR, "eta_final.png"),
    title="η (final) — geostrophic adjustment, limiter=$(limiter)")

# 2) Final velocity quiver
# -------- Velocity arrows with true lengths + a speed reference --------
skip = 10
xs = x_int[1:skip:end]
ys = y_int[1:skip:end]

# downsample velocities to the same coarse grid
U = uI[1:skip:end, 1:skip:end]
V = vI[1:skip:end, 1:skip:end]

# 2D coordinate grids matching U,V
X = repeat(xs, 1, length(ys))
Y = repeat(permutedims(ys), length(xs), 1)

# --- choose a scale so longest arrow ~ 70% of spacing between arrows ---
speed = sqrt.(U.^2 .+ V.^2)
maxspeed = maximum(speed)

# distance between neighboring arrows (in plot units)
Δ = 0.7 * min(skip*dx, skip*dy)         # target arrow length for max speed
scale_factor = Δ / max(maxspeed, 1e-12)  # avoid divide-by-zero

Us = U .* scale_factor
Vs = V .* scale_factor

# (optional) background η to mimic your example style; comment out if unwanted:
pltV = contourf(x_int, y_int, permutedims(ηIf);
                aspect_ratio=:equal, colorbar=true,
                title="Velocity field at time t=$(steps*dt) s",
                xlabel="x (m)", ylabel="y (m)")
quiver!(pltV, vec(X), vec(Y), quiver=(vec(Us), vec(Vs));
        scale=:none, arrowsize=0.5, lw=1.0, color=:green, label=false)


display(pltV)
savefig(pltV, joinpath(PLOTS_DIR, "velocity_final.png"))
