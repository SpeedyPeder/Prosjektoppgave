cd(@__DIR__)

include("sweSim1D.jl")
include("dambreak_analytic.jl")

using .sweSim1D
using .DambreakAnalytic
using Plots
using Printf
save_dir = joinpath(@__DIR__, "Plots_Compare")
isdir(save_dir) || mkdir(save_dir)

# ---------------- Params ----------------
N, L = 100, 5.0
CFL  = 0.45
T    = 0.6                  # final time for comparison
lim  = :mc
η0   = 0.0                  # not used directly here
x0   = 0.5L                 # dam position at middle

# Dry dam-break: left wet, right dry (or nearly dry)
hl, hr = 1.0, 0.2          # hr ~ 0 to approximate dry bed
times = 0.0:0.2:T

# ---------------- Bathymetry ----------------
# Dry dam-break on a flat bottom
bfun(x::AbstractVector) = zeros(length(x))

# ---------------- Initial condition ----------------
# Dam break in terms of depth h (right side almost dry)
function ic_dambreak(x; hl=hl, hr=hr, x0=x0)
    h = [xi < x0 ? hl : hr for xi in x]
    u = zeros(length(x))
    return h, u
end

# ---------------- Non-reflection window (same logic as old file) ----------------
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

# ---------------- Final state (numeric, KP scheme) ----------------
x, η, m = sweSim1D.sw_KP_upwind(N, L, T; CFL = CFL,limiter = lim,ic_fun = ic_dambreak,bfun = bfun)

# Convert (η, m) -> (h, u)
b = bfun(x)
h = @. η - b
HMIN = sweSim1D.HMIN
u = similar(h)
@inbounds for i in eachindex(h)
    u[i] = (h[i] > HMIN) ? (m[i]/h[i]) : 0.0
end

# ------------- Numeric vs Analytic @ T (if valid) -------------
if T ≤ tmax - eps()
    h_ex, u_ex = DambreakAnalytic.stoker_solution(x, T; hl=hl, hr=hr, x0=x0, g=g)

    e_h = h .- h_ex
    e_u = u .- u_ex

    L1(v)   = sum(abs, v)/length(v)
    L2(v)   = sqrt(sum(abs2, v)/length(v))
    Linf(v) = maximum(abs.(v))

    println("\nErrors at T=$(T): (KP dry dam-break)")
    @printf("  h:  L1=%.4e  L2=%.4e  Linf=%.4e\n", L1(e_h), L2(e_h), Linf(e_h))
    @printf("  u:  L1=%.4e  L2=%.4e  Linf=%.4e\n", L1(e_u), L2(e_u), Linf(e_u))

    q1 = plot(x, h_ex, lw=3, label="analytic h", xlabel="x", ylabel="h",
              title="t=$(round(T,digits=3))  (KP vs analytic, dry dam-break)")
    plot!(q1, x, h,   lw=1.8, ls=:dash, label="numeric h (KP)")

    q2 = plot(x, u_ex, lw=3, label="analytic u", xlabel="x", ylabel="u")
    plot!(q2, x, u,    lw=1.8, ls=:dash, label="numeric u (KP)")

    display(plot(q1, q2, layout=(2,1), size=(900,600)))
    savefig(joinpath(save_dir,"KP_compare_numeric_vs_analytic_T=$(round(T,digits=2)).png"))
else
    @info "Skipping analytic overlay at T=$T because waves have already hit the boundaries (tmax=$(round(tmax,digits=3)))."
end

# ------------- Snapshots + Analytic overlays (depth h) -------------
x_snap, snaps = sweSim1D.sw_KP_snapshots(N, L, valid_times; CFL = CFL,limiter = lim,ic_fun  = ic_dambreak,bfun  = bfun)

plt = plot(xlabel="x", ylabel="h", legend=:topright,title="KP dam-break: h snapshots (numeric vs analytic)")
for t in valid_times
    ηT, mT = snaps[t]
    bT = bfun(x_snap)
    hT = @. ηT - bT            # numeric depth at time t
    h_ex, _ = DambreakAnalytic.stoker_solution(x_snap, t; hl=hl, hr=hr, x0=x0, g=g)
    plot!(plt, x_snap, h_ex, lw=2.5, label="h exact, t=$(round(t,digits=2))")
    plot!(plt, x_snap, hT,  lw=1.5, ls=:dash, label="h num, t=$(round(t,digits=2))")
end
display(plt)
savefig(joinpath(save_dir, "KP_snapshots.png"))

# ------------- Animation (numeric + analytic, up to non-reflection Tend) -------------
function animate_compare_KP_dry(N, L; x0, hl, hr, CFL=0.45, limiter=:mc, Tstop=0.5, fps=30, path = joinpath(save_dir, "KP_compare_dambreak.gif"))
    g   = sweSim1D.g
    cL  = sqrt(g*hl)
    hm  = DambreakAnalytic.solve_hm(hl, hr; g=g)
    cm  = sqrt(g*hm)
    um  = 2*(cL - cm)
    s   = (hm * um) / (hm - hr)
    tmax = min(x0 / cL, (L - x0) / s)
    Tend = min(Tstop, tmax - 1e-9)
    # grid
    dx = L / N
    x  = @. (0.5:1:N-0.5) * dx

    # initial state (η,m)
    h0, u0 = ic_dambreak(x; hl=hl, hr=hr, x0=x0)
    b  = bfun(x)
    η  = h0 .+ b
    m  = h0 .* u0

    H   = zeros(eltype(η), N+1, 2)
    η1  = similar(η);  m1 = similar(m)
    η2  = similar(η);  m2 = similar(m)

    anim = Animation()
    t = 0.0

    while t < Tend - eps()
        # numeric h,u
        h = @. η - b
        u = similar(h)
        @inbounds for i in eachindex(h)
            u[i] = (h[i] > sweSim1D.HMIN) ? (m[i]/h[i]) : 0.0
        end

        # analytic h,u
        h_ex, u_ex = DambreakAnalytic.stoker_solution(x, t; hl=hl, hr=hr, x0=x0, g=g)

        p1 = plot(x, h_ex, lw=3, label="analytic h", xlabel="x", ylabel="h",
                  title="t=$(round(t,digits=3))  (stops at $(round(Tend,digits=3)) s)")
        plot!(p1, x, h, lw=1.8, ls=:dash, label="numeric h (KP)")

        p2 = plot(x, u_ex, lw=3, label="analytic u", xlabel="x", ylabel="u")
        plot!(p2, x, u, lw=1.8, ls=:dash, label="numeric u (KP)")

        frame(anim, plot(p1, p2, layout=(2,1), size=(900,600)))

        # advance one KP Heun step
        Bf, Bc, _dx = sweSim1D.build_Btilde_faces_centers(x, bfun)
        amax = sweSim1D.build_fluxes_reflective!(H, η, m, dx; Bf=Bf, limiter=limiter)
        dt   = (amax > 0) ? min(CFL*dx/(2*amax), Tend - t) : (Tend - t)

        sweSim1D.euler_step!(η1, m1, η, m, H, dt, dx, Bf, Bc)
        _ = sweSim1D.build_fluxes_reflective!(H, η1, m1, dx; Bf=Bf, limiter=limiter)
        sweSim1D.euler_step!(η2, m2, η1, m1, H, dt, dx, Bf, Bc)

        @inbounds for j in eachindex(η)
            η[j] = 0.5*(η[j] + η2[j])
            m[j] = 0.5*(m[j] + m2[j])
        end
        t += dt
    end

    gif(anim, path, fps=fps)
    return path
end

gif_comp = animate_compare_KP_dry(N, L;x0 = x0,hl = hl,hr = hr,CFL = CFL,limiter = lim,Tstop = 0.5,fps = 60,path = joinpath(save_dir, "KP_compare_dambreak.gif"))
println("Saved KP comparison animation to: $gif_comp")
