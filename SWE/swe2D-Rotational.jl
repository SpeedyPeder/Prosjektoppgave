cd(@__DIR__)
include("sweSim2D-Rotational.jl")
import .RotSW_CDKLM
using Plots; theme(:default)
using Printf   # <-- for scientific formatting

limiter = :minmod
steps   = 100
nx, ny  = 200, 200
dx, dy  = 1, 1
g       = 9.81
dt      = 0.1
bc      = :periodic
Hmin    = 1e-8

# --- Coriolis ---------------------------------------------
f0   = 1      # must be ≠ 0 for geostrophic v = g/f * h_x
beta = 0.0

x = collect(range(0, step=dx, length=nx))
y = collect(range(0, step=dy, length=ny))

# For this test: B = 0 (matches your derivation)
bfun(x,y) = 0.0

st, p = RotSW_CDKLM.init_state(x, y, bfun, f0, beta;
                               g=g, dt=dt, Hmin=Hmin,
                               limiter=limiter, bc=bc)

#############  Geostrophic IC: h(x), u=0, v(x)  #############

Lx = x[end] - x[1] + dx   # = nx*dx
h0 = 10
a  = 1e-3                 # bump amplitude (tune as you like)

# centers of left and right features
cL = Lx/4
cR = 3Lx/4
# half-width of each bump (so support is [c-w, c+w])
w  = Lx/6

# smooth cosine bump (positive) with compact support
function cos_bump(ξ, c, w)
    z = (ξ - c)/w
    if abs(z) <= 1
        return 0.5 * (1 + cos(π*z))   # 1 at center, 0 at |z|=1
    else
        return 0.0
    end
end

# derivative of the bump
function d_cos_bump(ξ, c, w)
    z = (ξ - c)/w
    if abs(z) <= 1
        return -0.5 * (π/w) * sin(π*z)
    else
        return 0.0
    end
end

# Geostrophic equilibrium state
h_profile(ξ) = h0 + a*cos_bump(ξ, cL, w) - a*cos_bump(ξ, cR, w)
dh_dx(ξ)     = a*d_cos_bump(ξ, cL, w)   - a*d_cos_bump(ξ, cR, w)
v_profile(ξ) = (g / f0) * dh_dx(ξ)

# fill grid
for (i, xi) in enumerate(x)
    hxi = h_profile(xi)
    vxi = v_profile(xi)
    for j in 1:ny
        st.h[i,j]  = hxi
        st.hu[i,j] = 0.0
        st.hv[i,j] = hxi * vxi
    end
end
@views begin
    st.q[1,:,:] .= st.h
    st.q[2,:,:] .= st.hu
    st.q[3,:,:] .= st.hv
end

w0 = copy(st.h)

# --- diagnostics helpers ---
function total_mass(st, p)
    p.dx * p.dy * sum(st.h)
end

println("Testing geostrophic-like initial state")
M0  = total_mass(st, p)
println("Initial: mass = $M0")
println("Coriolis: f ∈ [$(minimum(st.f)), $(maximum(st.f))]")

# (optional) check residual at t=0
RotSW_CDKLM.residual!(st,p)
println("max |dq| at t=0 = ", maximum(abs.(st.dq)))

###############################
# Helper for scientific colorbar ticks
###############################
function sci_cb_ticks(A; n=7)
    zmin, zmax = extrema(A)
    if zmin == zmax
        vals = [zmin]
    else
        vals = collect(range(zmin, zmax; length=n))
    end
    labels = [@sprintf("%.1e", v) for v in vals]
    return (vals, labels)
end

###############################
# Plot INITIAL CONDITIONS
###############################

xs = p.x
ys = p.y

w_init = st.h .+ st.Bc      # free surface at t = 0
u_init = st.hu ./ st.h
v_init = st.hv ./ st.h

# mid-line for 1D cross-sections
jmid = Int(cld(ny, 2))

########## HEIGHT ##########

# 1) 2D initial height (free surface) with scientific colorbar labels
plt_h2D = contourf(xs, ys, permutedims(w_init);
    aspect_ratio = :equal,
    xlabel = "x (m)", ylabel = "y (m)",
    title = "Initial free surface w(x,y)",
    colorbar = true,
    colorbar_ticks = sci_cb_ticks(w_init),
)

# 2) 1D cross-section of height (line)
plt_h1D = plot(xs, w_init[:, jmid];
    lw = 2,
    label="w(x, t=0)",
    xlabel="x (m)", ylabel="z (m)",
    title="Initial height cross-section at y = $(ys[jmid])",
)

########## VELOCITY ##########

# 3) 2D initial v-velocity with scientific colorbar labels
plt_v2D = contourf(xs, ys, permutedims(v_init);
    aspect_ratio = :equal,
    xlabel = "x (m)", ylabel = "y (m)",
    title = "Initial v-velocity v(x,y)",
    colorbar = true,
    colorbar_ticks = sci_cb_ticks(v_init),
)

# 4) 1D cross-section of v (line)
plt_v1D = plot(xs, v_init[:, jmid];
    lw = 2,
    label="v(x, t=0)",
    xlabel="x (m)", ylabel="v (m/s)",
    title="Initial v cross-section at y = $(ys[jmid])",
)

display(plt_h2D)
display(plt_h1D)
display(plt_v2D)
display(plt_v1D)

#######################

# =======================================================
# Run Simulation
# =======================================================
for n in 1:steps
    RotSW_CDKLM.step_RK2!(st, p)
    M  = total_mass(st, p)
    println("step $n: wmin = $(minimum(st.h)), wmax = $(maximum(st.h)), mass = $M (ΔM = $(M - M0))")
end

# ======================================================
# Plots at final time
# ======================================================

w = st.h .+ st.Bc      # free surface at final time
u = st.hu ./ st.h
v = st.hv ./ st.h
η = w .- w0            # error relative to equilibrium state

########## ERROR (w - w0) ##########

# 1) 2D error with scientific colorbar labels
pltEta = contourf(xs, ys, permutedims(η);
    aspect_ratio = :equal,
    xlabel = "x (m)", ylabel = "y (m)",
    title  = "Error w - w0 at t = $(steps*dt) s",
    colorbar = true,
    colorbar_ticks = sci_cb_ticks(η),
)
display(pltEta)

# 2) 1D error cross-section (line)
pltErr1D = plot(xs, η[:, jmid];
    lw = 2,
    label="w(x,t) - w0(x)",
    xlabel="x (m)", ylabel="error (m)",
    title="Error cross-section at y = $(ys[jmid]) m",
)
display(pltErr1D)

########## FINAL VS INITIAL (HEIGHT) ##########

# small horizontal shift for final curve

pltWSection = scatter(xs, w0[:, jmid];
    markersize = 3,
    marker = :circle,
    label="w0(x)",
    xlabel="x (m)", ylabel="z (m)",
    title="Free-surface cross-section at y = $(ys[jmid]) m",
)

scatter!(pltWSection, xs , w[:, jmid];
    markersize = 3,
    marker = :cross,
    label="w(x, t=$(steps*dt)s)",
)
display(pltWSection)

########## FINAL VS INITIAL (VELOCITY) ##########

pltVSection = scatter(xs, v_init[:, jmid];
    markersize = 3,
    marker = :circle,
    label="v(x,0)",
    xlabel="x (m)", ylabel="v (m/s)",
    title="Velocity cross-section at y = $(ys[jmid]) m",
)

scatter!(pltVSection, xs, v[:, jmid];
    markersize = 3,
    marker = :cross,
    label="v(x,t=$(steps*dt)s)",
)
display(pltVSection)


#########################
# SAVE FIGURES WITH PREFIX
#########################

mkpath("Plots_CDKLM")

# prefix such as "f0_1e-4" or "f0_1"
f_prefix = "f0_" * replace(string(f0), "." => "p")

# --- Initial condition plots ---
savefig(plt_h2D,   "Plots_CDKLM/$(f_prefix)_w_initial_2D.png")
savefig(plt_h1D,   "Plots_CDKLM/$(f_prefix)_w_initial_1D.png")
savefig(plt_v2D,   "Plots_CDKLM/$(f_prefix)_v_initial_2D.png")
savefig(plt_v1D,   "Plots_CDKLM/$(f_prefix)_v_initial_1D.png")

# --- Error and final state plots ---
savefig(pltEta,      "Plots_CDKLM/$(f_prefix)_eta_t$(steps*dt).png")
savefig(pltErr1D,    "Plots_CDKLM/$(f_prefix)_eta_cross_section_t$(steps*dt).png")
savefig(pltWSection, "Plots_CDKLM/$(f_prefix)_w_initial_vs_final_t$(steps*dt).png")
savefig(pltVSection, "Plots_CDKLM/$(f_prefix)_v_initial_vs_final_t$(steps*dt).png")
