cd(@__DIR__)

include("sweSim1D.jl")
using .sweSim1D
using Plots
using Printf

# ---------------- Params ----------------
N, L  = 100, 5.0
CFL   = 0.45                 
T     = 0.01
lim   = :minmod
times = 0.0:0.1:T            # snapshot times

# ---------------- Bathymetry (pick ONE) ----------------
# bfun(x::AbstractVector) = 0.20 .* (x ./ L) .+ ifelse.(x .> 0.7L, 0.40, 0.0) # linear + step (asymmetric)
bfun = bfun(x::AbstractVector) = 0.10 .* exp.(-((x .- 0.5L).^2) ./ (0.05L)^2)       # Gaussian bump
# bfun(x::AbstractVector) = ifelse.(x .> 0.55L, 0.07, 0.0) .+ ifelse.(x .> 0.82L, 0.05, 0.0) # double step
# bfun(x::AbstractVector) = zero.(x)                                           # flat (reference)

# ---------------- Initial condition ----------------
# Uniform depth (so η varies with b) – generates motion
h0 = 1.0
ic_fun(x) = (fill(h0, length(x)), zeros(length(x)))

#Lake-at-rest alternative (η = const, u = 0)
η0 = 1.0
ic_fun(x) = begin
    b = bfun(x)
    h = η0 .- b
    u = zeros(length(x))
    return h, u
end
source_fun = sweSim1D.default_source_zero

x = @. (0.5:1:N-0.5) * (L/N)

# Create "face" x-positions
xf = similar(x, N+1)
xf[1] = x[1] - 0.5*dx
for i in 2:N
    xf[i] = 0.5*(x[i-1] + x[i])
end
xf[end] = x[end] + 0.5*dx

# Reconstruct faces and centers
Bf, Bc, dx = sweSim1D.build_Btilde_faces_centers(x, bfun)



x, η, m = sweSim1D.sw_KP_upwind(N, L, T; CFL = CFL , limiter= lim, ic_fun = ic_fun,
    bfun = bfun)
print(size(η))
print(η)
#pη = plot(x, η, lw=2, label="η at T=$(T)", xlabel="x", ylabel="η",
#    title="Shallow water with bathymetry, T=$(T)")
#    display(pη) 
#x, η, m = sweSim1D.kp_plot_final(x, η, m, bfun; ylim_η=((0,2)), ylim_u= ((0,1)), T=T)

   

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
