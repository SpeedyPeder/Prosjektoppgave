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
BurgersSim.compare_limiters_zoom(50, 1, 0.5; CFL=0.6,
                      limiters=[:minmod, :mc, :superbee, :vanleer])

# --- Animation (GIF) ---
gifpath = joinpath(@__DIR__, "burgers.gif")
_ = BurgersSim.animate_burgers(N, L, T; CFL=CFL, limiter=:mc, fps=30, path=gifpath)

x,u_min = BurgersSim.burgers_muscl_godunov(N,L,T; limiter=:minmod)
_,u_sup = BurgersSim.burgers_muscl_godunov(N,L,T; limiter=:superbee)
println("max diff = ", maximum(abs.(u_min .- u_sup)))
