cd(@__DIR__)

include("sweSim1D.jl")
include("dambreak_analytic.jl")
using .DambreakAnalytic
using .sweSim1D
using Plots
using Printf

# ---------------- Params ----------------
N, L = 400, 5.0
CFL  = 0.45
T    = 1.0                  # used for static plots below; animation will stop earlier
lim  = :mc
solver = :hll
x0   = 0.5L

hl, hr = 1.0, 0.2
times = 0.0:0.1:T

# Initial condition (step at x0)
ic_fun(x) = ( [xi < x0 ? hl : hr for xi in x], zeros(length(x)) )
source_fun = sweSim1D.default_source_zero

# --------- Non-reflection window ---------
g   = sweSim1D.g
cL  = sqrt(g*hl)
hm  = DambreakAnalytic.solve_hm(hl, hr; g=g)
cm  = sqrt(g*hm)
um  = 2*(cL - cm)
s   = (hm * um) / (hm - hr)
tmax = min(x0 / cL, (L - x0) / s)  # valid while waves haven't hit walls

valid_times = [t for t in times if t ≤ tmax - eps()]
if maximum(times) > tmax
    @warn "Some requested times exceed non-reflection window (tmax=$(round(tmax,digits=3))). They will be skipped for analytic comparison."
end

# ---------------- Final state (numeric) ----------------
x, h, m = sweSim1D.sw_muscl_hll(N, L, T; CFL=CFL, limiter=lim, solver=solver,
                                ic_fun=ic_fun, source_fun=source_fun)

HMIN = hasproperty(sweSim1D, :HMIN) ? sweSim1D.HMIN : 1e-8
u = ifelse.(h .> HMIN, m ./ h, 0.0)

# Two-panel plot instead of yaxis=:right (avoids backend warning)
p1 = plot(x, h, xlabel="x", ylabel="h", label="numeric h(t=$T)")
p2 = plot(x, u, xlabel="x", ylabel="u", label="numeric u(t=$T)")
display(plot(p1, p2, layout=(2,1), size=(900,600)))

# ------------- Numeric vs Analytic @ T (if valid) -------------
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

    q1 = plot(x, h_ex, lw=3, label="analytic h", xlabel="x", ylabel="h", title="t=$(round(T,digits=3))")
    plot!(q1, x, h,   lw=1.8, ls=:dash, label="numeric h")
    q2 = plot(x, u_ex, lw=3, label="analytic u", xlabel="x", ylabel="u")
    plot!(q2, x, u,    lw=1.8, ls=:dash, label="numeric u")
    display(plot(q1, q2, layout=(2,1), size=(900,600)))
else
    @info "Skipping analytic overlay at T=$T because waves have already hit the boundaries (tmax=$(round(tmax,digits=3)))."
end

# ------------- Snapshots + Analytic overlays -------------
x_snap, snaps = sweSim1D.sw_snapshots(N, L, valid_times; CFL=CFL, limiter=lim, solver=solver,
                                      ic_fun=ic_fun, source_fun=source_fun)

plt = plot(xlabel="x", ylabel="h", legend=:topright, title="Shallow water: h snapshots (numeric vs analytic)")
for t in valid_times
    hT, mT = snaps[t]
    h_ex, _ = DambreakAnalytic.stoker_solution(x_snap, t; hl=hl, hr=hr, x0=x0, g=g)
    plot!(plt, x_snap, h_ex, lw=2.5, label="h exact t=$(round(t,digits=2))")
    plot!(plt, x_snap, hT,  lw=1.5, ls=:dash, label="h num   t=$(round(t,digits=2))")
end
display(plt)

# ------------- Animation (numeric + analytic, up to 0.5 s and before walls) -------------
function animate_compare(N, L; x0, hl, hr, CFL=0.45, limiter=:mc, solver=:hll,
                         Tstop=0.5, fps=30, path="compare_dambreak.gif")
    # stop before reflections and at requested cap
    g   = sweSim1D.g
    cL  = sqrt(g*hl)
    hm  = DambreakAnalytic.solve_hm(hl, hr; g=g)
    cm  = sqrt(g*hm)
    um  = 2*(cL - cm)
    s   = (hm * um) / (hm - hr)
    tmax = min(x0 / cL, (L - x0) / s)
    Tend = min(Tstop, tmax - 1e-9)           # safety margin

    # grid and initial state at t=0
    ic(x) = ( [xi < x0 ? hl : hr for xi in x], zeros(length(x)) )
    x, h, m = sweSim1D.sw_muscl_hll(N, L, 0.0; CFL=CFL, limiter=limiter, solver=solver,
                                    ic_fun=ic, source_fun=sweSim1D.default_source_zero)

    # storage for fluxes and RK stages
    Fhat = zeros(eltype(h), length(h)+1, 2)
    h1 = similar(h); m1 = similar(h)
    h2 = similar(h); m2 = similar(h)

    anim = Animation()
    t = 0.0
    dx = L / N
    while t < Tend - eps()
        # --- draw frame (numeric vs analytic)
        u = similar(h); @inbounds for i in eachindex(h); u[i] = (h[i] > sweSim1D.HMIN) ? m[i]/h[i] : 0.0; end
        h_ex, u_ex = DambreakAnalytic.stoker_solution(x, t; hl=hl, hr=hr, x0=x0, g=g)

        p1 = plot(x, h_ex, lw=3, label="analytic h", xlabel="x", ylabel="h",
                  title="t=$(round(t, digits=3))  (stops at $(round(Tend,digits=3)) s)")
        plot!(p1, x, h, lw=1.8, ls=:dash, label="numeric h")

        p2 = plot(x, u_ex, lw=3, label="analytic u", xlabel="x", ylabel="u")
        plot!(p2, x, u, lw=1.8, ls=:dash, label="numeric u")

        frame(anim, plot(p1, p2, layout=(2,1), size=(900,600)))

        # --- advance one SSPRK2 step (same numerics as your animate_sw)
        amax = sweSim1D.build_fluxes_reflective!(Fhat, h, m; limiter=limiter, solver=solver)
        dt = (amax > 0) ? min(CFL*dx/amax, Tend - t) : (Tend - t)
        dt_dx = dt/dx

        sweSim1D.euler_step_reflective!(h1,m1, h,m, Fhat, dt_dx, x, t, sweSim1D.default_source_zero)
        _ = sweSim1D.build_fluxes_reflective!(Fhat, h1, m1; limiter=limiter, solver=solver)
        sweSim1D.euler_step_reflective!(h2,m2, h1,m1, Fhat, dt_dx, x, t+dt, sweSim1D.default_source_zero)

        @inbounds for i in eachindex(h)
            h[i] = 0.5*(h[i] + h2[i])
            m[i] = 0.5*(m[i] + m2[i])
        end
        t += dt
    end
    gif(anim, path, fps=fps)
    return path
end

gif_comp = animate_compare(N, L; x0=x0, hl=hl, hr=hr, CFL=CFL, limiter=lim, solver=solver,
                           Tstop=0.5, fps=30, path=joinpath(@__DIR__, "compare_dambreak.gif"))
println("Saved comparison animation to: $gif_comp")

# ------------- Animation (numeric only, full domain) -------------
gifpath = joinpath(@__DIR__, "shallow_water.gif")
_ = sweSim1D.animate_sw(N, L, T; CFL=CFL, limiter=lim, solver=solver,
                        ic_fun=ic_fun, source_fun=source_fun, path=gifpath)
println("Saved numeric animation to: $gifpath")
