cd(@__DIR__)

include("sweSim1D.jl")
using .sweSim1D
using Plots

# Params
N, L = 100, 1.0
CFL  = 0.4
T    = 2.0
lim  = :mc                 # or :minmod
solver = :hll              # or :rusanov
times = 0:0.2:T

# Choose initial condition & (optional) source
ic_fun     = x -> sweSim1D.default_ic_dambreak(x; hl=1.0, hr=0.1, ul=0.0, ur=0.0, xc=0.5)
source_fun = sweSim1D.default_source_zero
# Example simple constant forcing on momentum:
# source_fun = (h,m,x,t) -> (@inbounds for i in eachindex(h); m[i] += 0.0; end)

# Final state
x, h, m = sweSim1D.sw_muscl_hll(N, L, T; CFL=CFL, limiter=lim, solver=solver,
                                ic_fun=ic_fun, source_fun=source_fun)

# Elementwise u = m/h with dry-cell guard
HMIN = hasproperty(sweSim1D, :HMIN) ? sweSim1D.HMIN : 1e-8
u = ifelse.(h .> HMIN, m ./ h, 0.0)

p = plot(x, h, xlabel="x", ylabel="h", label="h(t=$T)")
plot!(p, x, u, ylabel="u", label="u(t=$T)", yaxis=:right)
display(p)

"""
# Snapshots of h
x, snaps = sweSim1D.sw_snapshots(N, L, times; CFL=CFL, limiter=lim, solver=solver,
                                 ic_fun=ic_fun, source_fun=source_fun)
plt = plot(xlabel="x", ylabel="h", legend=:topright, title="Shallow water: h snapshots")
for t in times
    hT, mT = snaps[t]
    plot!(plt, x, hT, label="t=")
end
display(plt)
"""
# Animation
gifpath = joinpath(@__DIR__, "shallow_water.gif")
_ = sweSim1D.animate_sw(N, L, T; CFL=CFL, limiter=lim, solver=solver,
                        ic_fun=ic_fun, source_fun=source_fun, path=gifpath)
