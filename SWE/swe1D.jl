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
lim  = :minmod2       # or :minmod
#solver = :hll        # or :rusanov
x0   = 0.5L
dx = L / N

hl, hr = 1.2, 0.4
times = 0.0:0.3:T

# ---------------- Initial condition (flat bottom, dam break) ----------------
ic_fun(x) = ( [xi < x0 ? hl : hr for xi in x], zeros(length(x)) )
source_fun = sweSim1D.default_source_zero   # no extra forcing

# --------- Non-reflection window for analytic comparison ---------
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
x, h, m = sweSim1D.sw_muscl_hll(N, L, T; CFL=CFL, limiter=lim,
                                solver=solver, ic_fun=ic_fun, source_fun=source_fun)  
u =  m ./ h                                # velocity u = m/h

# ---------------- Plot h and u ----------------
p1 = plot(x, h, lw=2, label="numeric h(t=$T)", xlabel="x", ylabel="h")
p2 = plot(x, u, lw=2, label="numeric u(t=$T)", xlabel="x", ylabel="u")
display(plot(p1, p2, layout=(2,1), size=(900,600)))

# ------------- Numeric vs Analytic (flat bottom only) -------------
if T ≤ tmax - eps()
    h_ex, u_ex = DambreakAnalytic.stoker_solution(x, T; hl=hl, hr=hr, x0=x0, g=g)

    e_h = h .- h_ex
    e_u = u .- u_ex
    L1(v)  = sum(abs, v)/length(v)
    L2(v)  = sqrt(sum(abs2, v)/length(v))
    Linf(v)= maximum(abs.(v))

    println("\nErrors at T=$(T):")
    @printf("  h:  L1=%.4e  L2=%.4e  Linf=%.4e\n", L1(e_h), L2(e_h), Linf(e_h))
    @printf("  u:  L1=%.4e  L2=%.4e  Linf=%.4e\n", L1(e_u), L2(e_u), Linf(e_u))

    q1 = plot(x, h_ex, lw=3, label="analytic h", xlabel="x", ylabel="h",
              title="Dambreak: t=$(round(T,digits=3)) (flat bottom)")
    plot!(q1, x, h,   lw=2, ls=:dash, label="numeric h")

    q2 = plot(x, u_ex, lw=3, label="analytic u", xlabel="x", ylabel="u")
    plot!(q2, x, u,    lw=2, ls=:dash, label="numeric u")

    display(plot(q1, q2, layout=(2,1), size=(900,600)))
else
    @info "Skipping analytic overlay at T=$T (waves reached boundaries, tmax=$(round(tmax,digits=3)))."
end

# ------------- Snapshots (with analytic overlays while valid) -------------
x_snap, snaps = sweSim1D.sw_snapshots(N, L, valid_times; CFL=CFL, limiter=lim,
                                      solver=solver, ic_fun=ic_fun, source_fun=source_fun)

plt = plot(xlabel="x", ylabel="h", legend=:bottomleft, title="Shallow water snapshots (flat bottom)")
for t in valid_times
    hT, mT = snaps[t]
    plot!(plt, x_snap, hT,  lw=1.6, ls=:dash, label="h num, at t=$(round(t,digits=2))")
    h_ex, _ = DambreakAnalytic.stoker_solution(x_snap, t; hl=hl, hr=hr, x0=x0, g=g)
    plot!(plt, x_snap, h_ex, lw=2.2, label="h exact, at t=$(round(t,digits=2))")
end
display(plt)
savefig(plt, joinpath(@__DIR__, "shallow_water_snapshots.png"))
println("Saved snapshots plot to: $(joinpath(@__DIR__, "shallow_water_snapshots.png"))")

# ------------- Animation (numeric) -------------
gifpath = joinpath(@__DIR__, "shallow_water.gif")
_ = sweSim1D.animate_sw(N, L, T; CFL=CFL, limiter=lim, solver=solver,
                        ic_fun=ic_fun, source_fun=source_fun, path=gifpath)
println("Saved animation to: $gifpath")