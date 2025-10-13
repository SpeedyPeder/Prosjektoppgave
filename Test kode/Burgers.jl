"""
cd(@__DIR__)
include("BurgersSim.jl")
using .BurgersSim
using Plots

# --- Parameters ---
N, L = 100, 1.0
CFL  = 0.45
T    = 0.1
lim  = :mc

# Top-hat IC: u=1 on [0,0.1), u=0 elsewhere (periodic)
ic_box = x -> BurgersSim.initial_condition_tophat(x; a=0.0, b=0.1, u_in=1.0, u_out=0.0, L=L)

# --- Compare schemes at time T ---
x, sols = BurgersSim.burgers_compare_at(N, L, T; CFL=CFL, limiter=lim, ic=ic_box)

plt = plot(xlabel="x", ylabel="u", title="Burgers (box IC) at t=$(T)", legend=:bottomright)
for (lab, u) in sols
    scatter!(plt, x, u; label=lab, ms=3, m=:circle)
end
display(plt)
"""

cd(@__DIR__)
include("BurgersSim.jl")
using .BurgersSim
using Plots

# --- Parameters ---
N, L  = 400, 1.0
CFL   = 0.45
T     = 0.20                 # any t>0 works; periodic wrap is handled
uL,uR = 1.0, 0.0             # shock case; set uL<uR to test rarefaction
x0    = 0.35
lim   = :mc

# --- Run & compare ---
x, sols, u_exact = BurgersSim.burgers_compare_vs_exact(N, L, T;
    CFL=CFL, limiter=lim, uL=uL, uR=uR, x0=x0)

# Overlay: exact line + scheme dots
plt = plot(x, u_exact; lw=2, label="Exact", xlabel="x", ylabel="u",
           title="Burgers Riemann: t=$(T), uL=$(uL), uR=$(uR)")
for (lab, u) in sols
    scatter!(plt, x, u; label=lab, ms=3, m=:circle)
end
display(plt)

# Optional zoom near discontinuity (shock when uL>uR, rarefaction head/tail when uL<uR)
# if uL>uR
#   s = 0.5*(uL+uR); xshock = mod(x0 + s*T, L)
#   xlims!(plt, xshock-0.1, xshock+0.1); display(plt)
