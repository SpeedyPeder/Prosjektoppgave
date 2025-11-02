cd(@__DIR__)

include("sweSim1D.jl")
using .sweSim1D
using Plots
using Printf

# ---------------- Params ----------------
N, L = 100, 5.0
CFL  = 0.45
T    = 1.0
lim  = :minmod
solver = :hll
times = 0.0:0.1:T   # snapshot times

# ---------------- Bathymetry (pick ONE) ----------------
bfun(x::AbstractVector) = 0.20 .* (x ./ L) .+ ifelse.(x .> 0.7L, 0.40, 0.0) # linear + step (asymmetric)
# bfun(x::AbstractVector) = 0.18 .* exp.(-((x .- 0.72L).^2) ./ (0.06L)^2)   # Gaussian bump
# bfun(x::AbstractVector) = ifelse.(x .> 0.55L, 0.07, 0.0) .+ ifelse.(x .> 0.82L, 0.05, 0.0) # double step
# bfun(x::AbstractVector) = zero.(x)                                         # flat (reference)

# ---------------- Initial condition ----------------
# Uniform depth (so η varies with b): this will generate motion.
h0 = 1.0
ic_fun(x) = (fill(h0, length(x)), zeros(length(x)))

#Lake-at-rest alternative (η = const, u=0). Uncomment to test steadiness with well_balanced=true:
η0 = 1.0
ic_fun(x) = begin
    b = bfun(x)
    h = max.(η0 .- b, 1e-8)
    u = zeros(length(x))
    return h, u
end

source_fun = sweSim1D.default_source_zero

# ---------------- Final state (numeric) ----------------
x, h, m = sweSim1D.sw_muscl_hll(N, L, T; CFL=CFL, limiter=lim, solver=solver,
                                ic_fun=ic_fun, source_fun=source_fun,
                                bfun=bfun, well_balanced=true)  # WB recommended with bathymetry

HMIN = hasproperty(sweSim1D, :HMIN) ? sweSim1D.HMIN : 1e-8
u = ifelse.(h .> HMIN, m ./ h, 0.0)
b = bfun(x)
η = h .+ b

# ---------------- Plot: η vs b and u ----------------
p1 = plot(x, η, lw=2, label="η=h+b", xlabel="x", ylabel="elevation",
          title="Shallow water with asymmetric bathymetry, T=$(T)")
plot!(p1, x, b, lw=2, ls=:dash, label="b(x)")
p2 = plot(x, u, lw=2, label="u", xlabel="x", ylabel="u")
display(plot(p1, p2, layout=(2,1), size=(950,650)))

# ---------------- Snapshots (numeric only) ----------------
x_snap, snaps = sweSim1D.sw_snapshots(N, L, times; CFL=CFL, limiter=lim, solver=solver,
                                      ic_fun=ic_fun, source_fun=source_fun,
                                      bfun=bfun, well_balanced= true)
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
_ = sweSim1D.animate_sw(N, L, T; CFL=CFL, limiter=lim, solver=solver,
                        ic_fun=ic_fun, source_fun=source_fun,
                        bfun=bfun, well_balanced=true,
                        path=gifpath)
println("Saved animation to: $gifpath")

