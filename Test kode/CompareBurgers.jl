cd(@__DIR__)

include("BurgersSim.jl")     # your FV MUSCL+Godunov module
using .BurgersSim            # FV functions
include("BurgersFDSim.jl")   # the FD module above
using .BurgersFDSim

using Plots

# --- Parameters ---
N  = 400
L  = 1.0
T  = 0.30               # > t_s = 1/(2π) ≈ 0.159 -> after shock
CFL_FV = 0.45
CFL_FD = 0.45          # for LF; if you switch to LW, you can use ~0.95

# --- Run FV (reference) ---
x_fv, u_fv = BurgersSim.burgers_muscl_godunov(N, L, T; CFL=CFL_FV, limiter=:mc)

# --- Run FD (choose scheme) ---
# Lax–Friedrichs (robust, more diffusive):
x_fd, u_fd = BurgersFDSim.burgers_fd_lf(N, L, T; CFL=CFL_FD)

# If you want Lax–Wendroff instead (sharper but can ring near shocks):
# x_fd, u_fd = BurgersFDSim.burgers_fd_lw(N, L, T; CFL=0.95)

# --- Plot snapshot comparison ---
plt = plot(x_fv, u_fv, label="FV MUSCL+Godunov @ T=$(T)", xlabel="x", ylabel="u",
           title="Burgers snapshot after shock (T=$(T))", legend=:bottomright)
plot!(plt, x_fd, u_fd, label="FD Lax–Friedrichs @ T=$(T)")
display(plt)
