cd(@__DIR__)

include("sweSim1D.jl")
include("dambreak_analytic.jl")
using .DambreakAnalytic
using .sweSim1D
using Plots
using Printf

# ---------------- Params ----------------
N, L = 100, 5.0
CFL  = 0.45
T    = 1.0
lim  = :mc
solver = :hll
x0   = 0.5L

hl, hr = 1.0, 0.2
times = 0.0:0.1:T

# ---------------- Bathymetry ----------------
# bfun(x) = zeros(length(x))                            
# bfun(x) = @. 0.25 * exp(-((x - 0.7*L)^2) / (0.05*L)^2) 
#bfun(x) = [xi < 0.6L ? 0.0 : 0.15 for xi in x]        
bfun(x) = @. 0.12*(x/L) + (x > 0.7L ? 0.08 : 0.0)
# bfun(x) = @. 0.18 * exp(-((x - 0.72L)^2) / (0.06L)^2)
# bfun(x) = @. (x > 0.55L ? 0.07 : 0.0) + (x > 0.82L ? 0.05 : 0.0)


# ---------------- Initial condition ----------------
ic_fun(x) = ( [xi < x0 ? hl : hr for xi in x], zeros(length(x)) )
source_fun = sweSim1D.default_source_zero

# --------- Trim times before reflections ---------
g   = sweSim1D.g
cL  = sqrt(g*hl)
hm  = DambreakAnalytic.solve_hm(hl, hr; g=g)
cm  = sqrt(g*hm)
um  = 2*(cL - cm)
s   = (hm * um) / (hm - hr)
tmax = min(x0 / cL, (L - x0) / s)

valid_times = [t for t in times if t ≤ tmax - eps()]
if maximum(times) > tmax
    @warn "Some requested times exceed non-reflection window (tmax=$(round(tmax,digits=3))). Skipping analytic comparison after this."
end

# ---------------- Final state (numeric) ----------------
if @isdefined(bfun)
    x, h, m = sweSim1D.sw_muscl_hll(N, L, T; CFL=CFL, limiter=lim,
                                    solver=solver, ic_fun=ic_fun,
                                    source_fun=source_fun, bfun=bfun)
    b = bfun(x)
else
    x, h, m = sweSim1D.sw_muscl_hll(N, L, T; CFL=CFL, limiter=lim,
                                    solver=solver, ic_fun=ic_fun,
                                    source_fun=source_fun)
    b = zeros(length(x))
end

HMIN = hasproperty(sweSim1D, :HMIN) ? sweSim1D.HMIN : 1e-8
u = ifelse.(h .> HMIN, m ./ h, 0.0)

# ---------------- Plot water surface vs bathymetry ----------------
η = h .+ b   # free surface elevation
p1 = plot(x, η, lw=2, label="water surface", xlabel="x", ylabel="elevation")
plot!(p1, x, b, lw=2, ls=:dash, c=:brown, label="bathymetry")
p2 = plot(x, u, xlabel="x", ylabel="u", label="velocity u")
display(plot(p1, p2, layout=(2,1), size=(900,600)))

# ------------- Numeric vs Analytic (only if flat bottom) -------------
if !@isdefined(bfun) && T ≤ tmax - eps()
    h_ex, u_ex = DambreakAnalytic.stoker_solution(x, T; hl=hl, hr=hr, x0=x0, g=g)

    e_h = h .- h_ex
    e_u = u .- u_ex
    L1(v)  = sum(abs, v)/length(v)
    L2(v)  = sqrt(sum(abs2, v)/length(v))
    Linf(v)= maximum(abs.(v))

    println("\nErrors at T=$(T):")
    @printf("  h:  L1=%.4e  L2=%.4e  Linf=%.4e\n", L1(e_h), L2(e_h), Linf(e_h))
    @printf("  u:  L1=%.4e  L2=%.4e  Linf=%.4e\n", L1(e_u), L2(e_u), Linf(e_u))

    q1 = plot(x, h_ex, lw=3, label="analytic h", xlabel="x", ylabel="h", title="t=$(round(T,digits=3))")
    plot!(q1, x, h,   lw=1.8, ls=:dash, label="numeric h")
    q2 = plot(x, u_ex, lw=3, label="analytic u", xlabel="x", ylabel="u")
    plot!(q2, x, u,    lw=1.8, ls=:dash, label="numeric u")
    display(plot(q1, q2, layout=(2,1), size=(900,600)))
elseif @isdefined(bfun)
    @info "Analytic solution disabled (bathymetry present)."
else
    @info "Skipping analytic overlay at T=$T (waves reached boundaries)."
end

# ------------- Snapshots -------------
if @isdefined(bfun)
    x_snap, snaps = sweSim1D.sw_snapshots(N, L, valid_times; CFL=CFL,
                                          limiter=lim, solver=solver,
                                          ic_fun=ic_fun, source_fun=source_fun)
    b_snap = bfun(x_snap)
else
    x_snap, snaps = sweSim1D.sw_snapshots(N, L, valid_times; CFL=CFL,
                                          limiter=lim, solver=solver,
                                          ic_fun=ic_fun, source_fun=source_fun)
    b_snap = zeros(length(x_snap))
end

plt = plot(xlabel="x", ylabel="h", legend=:topright, title="Shallow water snapshots")
for t in valid_times
    hT, mT = snaps[t]
    ηT = hT .+ b_snap
    plot!(plt, x_snap, ηT,  lw=1.5, ls=:dash, label="η num t=$(round(t,digits=2))")
    if !@isdefined(bfun)
        h_ex, _ = DambreakAnalytic.stoker_solution(x_snap, t; hl=hl, hr=hr, x0=x0, g=g)
        plot!(plt, x_snap, h_ex, lw=2.5, label="h exact t=$(round(t,digits=2))")
    end
end
plot!(plt, x_snap, b_snap, lw=2, ls=:dashdot, c=:brown, label="bathymetry")
display(plt)

# ------------- Animation (numeric) -------------
gifpath = joinpath(@__DIR__, "shallow_water.gif")
if @isdefined(bfun)
    _ = sweSim1D.animate_sw(N, L, T; CFL=CFL, limiter=lim, solver=solver,
                            ic_fun=ic_fun, source_fun=source_fun, bfun=bfun, path=gifpath)
else
    _ = sweSim1D.animate_sw(N, L, T; CFL=CFL, limiter=lim, solver=solver,
                            ic_fun=ic_fun, source_fun=source_fun, path=gifpath)
end
println("Saved animation to: $gifpath")

