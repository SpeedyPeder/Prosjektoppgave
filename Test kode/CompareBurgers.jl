cd(@__DIR__)

using .BurgersSim            # FV functions
using .BurgersFDSim          # FD functions

using Plots

# --- Parameters ---
N  = 400
L  = 1.0
T  = 1.0               #t_s = 1/(2π) ≈ 0.15915 (shock time)
CFL_FV = 0.45
CFL_FD = 0.45          

# --- Run FV  ---
x_fv, u_fv = BurgersSim.burgers_muscl_godunov(N, L, T; CFL=CFL_FV, limiter=:mc)

# --- Choose FD scheme!---
# Lax–Friedrichs:
x_fd, u_fd = BurgersFDSim.burgers_fd_lf(N, L, T; CFL=CFL_FD)

#Lax–Wendroff:
#x_fd, u_fd = BurgersFDSim.burgers_fd_lw(N, L, T; CFL=0.95)

# --- Choose comparison! ---
plt = plot(x_fv, u_fv, label="FV MUSCL+Godunov at T=$(T)", xlabel="x", ylabel="u",
           title="Burgers snapshot after shock (T=$(T))", legend=:bottomright)
#Lax–Friedrichs:
plot!(plt, x_fd, u_fd, label="FD Lax–Friedrichs at T=$(T)")
#Lax–Wendroff:
#plot!(plt, x_fd, u_fd, label="FD Lax–Wendroff at T=$(T)")
display(plt)
