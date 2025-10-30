cd(@__DIR__)                  # save outputs in this script’s folder
include("BurgersSim.jl")
using .BurgersSim           

using Plots

# --- Parameters (edit here) ---
N, L = 400, 1.0
CFL  = 0.45
T    = 1.0
lim  = :mc
times = 0:0.1:T               # snapshots every 0.1 s

# --- Final state at T ---
x, uT = BurgersSim.burgers_muscl_godunov(N, L, T; CFL=CFL, limiter=lim)
plot(x, uT, xlabel="x", ylabel="u", label="t=$T", legend=:bottomright)
display(current())

# --- Snapshots (overlay) ---
x, snaps = BurgersSim.burgers_snapshots(N, L, times; CFL=CFL, limiter=lim)
plt = plot(xlabel="x", ylabel="u", legend=:bottomright, title="Snapshots every 0.1 s")
for t in times
    plot!(plt, x, snaps[t], label="t=$(t)")
end

display(plt)
savefig(plt, "burgers_snapshots.png")

#------ Compare limiters vs Analytical ------
compare_limiters_vs_analytical(400, 1.0, 0.1; CFL=0.45, limiters=[:minmod, :mc])

# --- Animation (GIF) ---
gifpath = joinpath(@__DIR__, "burgers.gif")
_ = BurgersSim.animate_burgers(N, L, T; CFL=CFL, limiter=lim, fps=30, path=gifpath)

