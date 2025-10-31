cd(@__DIR__)                  # save outputs in this script’s folder
include("BurgersSim.jl")
using .BurgersSim           

using Plots; theme(:default)
# --- Parameters (edit here) ---
N, L = 200, 1.0
CFL  = 0.6
T    = 0.5
times = 0:0.1:T               # snapshots every 0.1 s

# --- Final state at T ---
x, uT = BurgersSim.burgers_muscl_godunov(N, L, T; CFL=CFL, limiter=:mc)
plot(x, uT, xlabel="x", ylabel="u", label="t=$T", legend=:bottomright)
display(current())

# --- Snapshots (overlay) ---
x, snaps = BurgersSim.burgers_snapshots(N, L, times; CFL=CFL, limiter=:mc)
plt = plot(xlabel="x", ylabel="u", legend=:bottomright, title="Snapshots every 0.1 s")
for t in times
    plot!(plt, x, snaps[t], label="t=$(t)")
end

display(plt)
savefig(plt, "burgers_snapshots.png")

#------ Compare limiters ------
BurgersSim.compare_limiters_zoom(100, 1, 0.5; CFL=0.6,
                      limiters=[:minmod, :mc, :superbee, :vanleer])

# --- Animation (GIF) ---
gifpath = joinpath(@__DIR__, "burgers.gif")
_ = BurgersSim.animate_burgers(N, L, T; CFL=CFL, limiter=:mc, fps=30, path=gifpath)


# --- Compare different FV methods and compare with analytical solution ---
# N=50 grid to highlight smearing/oscillations
N, L, CFL, T = 50, 1.0, 0.6, 1

# One method at a time; numeric as dots, analytic as line
BurgersSim.burgers_compare_at(N, L, T; CFL, method=:upwind,      show_analytic=true, savepath="upwind.png")
BurgersSim.burgers_compare_at(N, L, T; CFL, method=:laxfriedrichs, show_analytic=true, savepath="lf.png")
BurgersSim.burgers_compare_at(N, L, T; CFL, method=:laxwendroff, show_analytic=true, savepath="lw.png")
BurgersSim.burgers_compare_at(N, L, T; CFL, method=:muscl, limiter=:minmod, show_analytic=true, savepath="muscl_minmod.png")
BurgersSim.burgers_compare_at(N, L, T; CFL, method=:muscl, limiter=:mc,     show_analytic=true, savepath="muscl_mc.png")
