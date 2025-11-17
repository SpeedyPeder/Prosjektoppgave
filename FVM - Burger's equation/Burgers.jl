cd(@__DIR__)                  # run from this script’s folder
include("BurgersSim.jl")
using .BurgersSim
using Plots; theme(:default)

# --- Create plots/ folder ---
const PLOTS_DIR = joinpath(@__DIR__, "plots")
mkpath(PLOTS_DIR)

# --- Parameters (edit here) ---
N, L = 200, 1.0
CFL  = 0.6
T    = 0.5
times = 0:0.1:T               # snapshots every 0.1 s

# --- Final state at T ---
x, uT = BurgersSim.burgers_muscl_godunov(N, L, T; CFL=CFL, limiter=:mc)
plt_final = plot(x, uT, xlabel="x", ylabel="u", label="t=$T", legend=:bottomright)
display(plt_final)
savefig(joinpath(PLOTS_DIR, "burgers_final.png"))

# --- Snapshots (overlay) ---
x, snaps = BurgersSim.burgers_snapshots(N, L, times; CFL=CFL, limiter=:mc)
plt_snaps = plot(xlabel="x", ylabel="u", legend=:bottomright,
                 title="Snapshots every 0.1 s")
for t in times
    plot!(plt_snaps, x, snaps[t], label="t=$(t)")
end
display(plt_snaps)
savefig(joinpath(PLOTS_DIR, "burgers_snapshots.png"))

# --- Compare limiters (all plotted on same figure) ---
BurgersSim.compare_limiters_zoom(
    50, 1, 0.8; CFL=0.6, zoom_halfwidth= L/20,
    limiters=[:minmod, :superbee],
    savepath=joinpath(PLOTS_DIR, "limiters_zoom.png")
)

# --- Animation (GIF) ---
gifpath = joinpath(PLOTS_DIR, "burgers.gif")
_ = BurgersSim.animate_burgers(N, L, T; CFL=CFL, limiter=:mc, fps=30, path=gifpath)

# --- Compare different FV methods (analytic vs numeric) ---
# smaller grid to highlight smearing/oscillations
N, L, CFL, T = 50, 1, 0.6, 1

BurgersSim.burgers_compare_at(N, L, T; CFL,
    method=:upwind, show_analytic=true,
    savepath=joinpath(PLOTS_DIR,"upwind.png"))
BurgersSim.burgers_compare_at(N, L, T; CFL,
    method=:laxfriedrichs, show_analytic=true,
    savepath=joinpath(PLOTS_DIR,"lax-friedrichs.png"))
BurgersSim.burgers_compare_at(N, L, T; CFL,
    method=:laxwendroff, show_analytic=true,
    savepath=joinpath(PLOTS_DIR,"lax-wendroff.png"))
BurgersSim.burgers_compare_at(N, L, T; CFL,
    method=:muscl, limiter=:minmod, show_analytic=true,
    savepath=joinpath(PLOTS_DIR,"muscl_minmod.png"))
BurgersSim.burgers_compare_at(N, L, T; CFL,
    method=:muscl, limiter=:superbee, show_analytic=true,
    savepath=joinpath(PLOTS_DIR,"muscl_superbee.png"))
BurgersSim.burgers_compare_at(N, L, T; CFL,
    method=:muscl, limiter=:vanleer, show_analytic=true,
    savepath=joinpath(PLOTS_DIR,"muscl_vanleer.png"))
BurgersSim.burgers_compare_at(N, L, T; CFL,
    method=:muscl, limiter=:mc, show_analytic=true,
    savepath=joinpath(PLOTS_DIR,"muscl_mc.png"))
