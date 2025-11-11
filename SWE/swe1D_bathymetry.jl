cd(@__DIR__)

include("sweSim1D.jl")
using .sweSim1D
using Plots
using Printf

# ---------------- Params ----------------
N, L  = 100, 5.0
CFL   = 0.45
T     = 10
lim   = :minmod
# times = 0.0:0.1:T   # (unused here)

# ---------------- Bathymetry (pick ONE) ----------------
bfun(x::AbstractVector) = 0.20 .* (x ./ L) .+ ifelse.(x .> 0.7L, 0.40, 0.0)    # linear + step
# bfun(x::AbstractVector) = 0.10 .* exp.(-((x .- 0.5L).^2) ./ (0.05L)^2)         # Gaussian bump
# bfun(x::AbstractVector) = ifelse.(x .> 0.55L, 0.07, 0.0) .+ ifelse.(x .> 0.82L, 0.05, 0.0) # double step
#bfun(x::AbstractVector) = zeros(length(x))                                       # flat bottom

# ---------------- Initial condition ----------------
# Lake-at-rest: η = η0 const, u = 0  (works for ANY bfun)
η0 = 1.0
ic_fun(x) = begin
    b = bfun(x)               # bfun is defined *above* now
    h = η0 .- b               # h = η - b
    u = zeros(length(x))
    return h, u
end

source_fun = sweSim1D.default_source_zero

# ---------------- Run ----------------
x, η, m = sweSim1D.sw_KP_upwind(N, L, T; CFL=CFL, limiter=lim,
                                ic_fun=ic_fun, bfun=bfun)

# ---------------- Plot ----------------
b = bfun(x)

# Plot free surface η and bottom b(x)
pη = plot(
    x, η,
    lw = 2,
    label = "η (free surface)",
    xlabel = "x",
    ylabel = "Height [m]",
    title = "Shallow Water with Bathymetry, T = $(T)",
)
plot!(x, b, lw = 2, ls = :dash, label = "b(x) (bathymetry)")
display(pη)


   
"""
# ---------------- Final state (numeric) ----------------
x, h, m = sweSim1D.(N, L, T; CFL=CFL, limiter=lim,
                                ic_fun=ic_fun, source_fun=source_fun,
                                bfun=bfun)

# Safe velocity (avoid divide-by-zero)
u = [h[i] > sweSim1D.HMIN ? m[i]/h[i] : 0.0 for i in eachindex(h)]
b = bfun(x)
η = h .+ b

# ---------------- Plot: η vs b and u ----------------
p1 = plot(x, η, lw=2, label="η = h + b", xlabel="x", ylabel="elevation",
          title="Shallow water with bathymetry, T=$(T)")
plot!(p1, x, b, lw=2, ls=:dash, label="b(x)")
p2 = plot(x, u, lw=2, label="u", xlabel="x", ylabel="u")
display(plot(p1, p2, layout=(2,1), size=(950,650)))

# ---------------- Snapshots (numeric only) ----------------
x_snap, snaps = sweSim1D.sw_snapshots(N, L, times; CFL=CFL, limiter=lim,
                                      ic_fun=ic_fun, source_fun=source_fun,
                                      bfun=bfun)
b_snap = bfun(x_snap)

plt = plot(xlabel="x", ylabel="elevation", legend=:topright,
           title="Snapshots: free surface and bathymetry")
for t in times
    hT, mT = snaps[t]
    ηT = hT .+ b_snap
    plot!(plt, x_snap, ηT, lw=1.6, label="η @ t=$(round(t,digits=2))")
end
plot!(plt, x_snap, b_snap, lw=2, ls=:dashdot, label="b(x)")
display(plt)

# ---------------- Animation (η & b overlay + velocity) ----------------
gifpath = joinpath(@__DIR__, "shallow_water_bathy.gif")
_ = sweSim1D.animate_sw(N, L, T; CFL=CFL, limiter=lim,
                        ic_fun=ic_fun, source_fun=source_fun,
                        bfun=bfun, path=gifpath)
println("Saved animation to: $gifpath")
"""