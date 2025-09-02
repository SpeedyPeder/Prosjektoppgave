module sweSim1D

using Plots
using StaticArrays
export sw_muscl_hll, sw_snapshots, animate_sw,
       default_ic_dambreak, default_ic_sine, default_source_zero

# ------------------ Physics & constants ------------------
const g = 9.81
const HMIN = 1e-8        # dry tolerance
const UMAXCAP = 1e6      # safety cap for u

@inline cons(h,u) = SVector(h, h*u)

@inline function sw_flux(h, m)
    u = (h > HMIN) ? (m/h) : 0.0
    return SVector(m, m*u + 0.5*g*h*h)
end

# ------------------ Limiters & helpers -------------------
@inline minmod2(a, b) = (a*b <= 0) ? 0.0 : sign(a)*min(abs(a), abs(b))
@inline function minmod3(a, b, c)
    s = sign(a)
    (sign(b) != s || sign(c) != s) && return 0.0
    return s * min(abs(a), abs(b), abs(c))
end

@inline function slope_limited(um, u0, up; limiter::Symbol=:mc)
    a = u0 - um
    b = up - u0
    limiter === :minmod && return minmod2(a,b)
    limiter === :mc     && return minmod3(0.5*(a+b), 2a, 2b)
    error("Unknown limiter $limiter")
end

@inline pos(x) = x < HMIN ? HMIN : x
@inline capu(u) = abs(u) > UMAXCAP ? sign(u)*UMAXCAP : u

# ------------------ Riemann solvers ----------------------
@inline function rusanov_flux(hL,uL,hR,uR)
    mL = hL*uL; mR = hR*uR
    FL = sw_flux(hL, mL)
    FR = sw_flux(hR, mR)
    cL = sqrt(g*hL); cR = sqrt(g*hR)
    smax = max(abs(uL)+cL, abs(uR)+cR)
    return 0.5*(FL + FR) - 0.5*smax*(SVector(hR,mR) - SVector(hL,mL))
end

@inline function hll_flux(hL,uL,hR,uR)
    mL = hL*uL; mR = hR*uR
    FL = sw_flux(hL, mL); FR = sw_flux(hR, mR)
    cL = sqrt(g*hL); cR = sqrt(g*hR)
    sL = min(uL - cL, uR - cR)
    sR = max(uL + cL, uR + cR)
    if sL >= 0
        return FL
    elseif sR <= 0
        return FR
    else
        return (sR*FL - sL*FR + sL*sR*(SVector(hR,mR) - SVector(hL,mL))) / (sR - sL)
    end
end

# ------------------ Reflective flux builder ---------------
# Build F̂ on N+1 interfaces with reflective (wall) BCs:
# interface 1:   ghost|cell 1     (u flips, h mirrors)
# interfaces 2..N: between cells
# interface N+1: cell N|ghost
function build_fluxes_reflective!(Fhat, h, m; limiter::Symbol=:mc, solver::Symbol=:hll)
    inds = axes(h,1); fi, li = firstindex(h), lastindex(h)
    N = length(h)

    # 1) primitives (h,u) with positivity/caps
    hpr = similar(h); upr = similar(h)
    @inbounds for i in inds
        hi = pos(h[i])
        ui = (hi > HMIN) ? m[i]/hi : 0.0
        hpr[i] = hi
        upr[i] = capu(ui)
    end

    # 2) limited slopes for primitives (reflect at boundaries)
    sh = similar(hpr); su = similar(upr)
    @inbounds for i in inds
        if i == fi
            hm, um = hpr[fi], -upr[fi]          # left ghost: mirror h, flip u
            hp, up = hpr[fi+1], upr[fi+1]
            sh[i] = slope_limited(hm, hpr[i], hp; limiter=limiter)
            su[i] = slope_limited(um, upr[i], up; limiter=limiter)
        elseif i == li
            hm, um = hpr[li-1], upr[li-1]
            hp, up = hpr[li],   -upr[li]        # right ghost
            sh[i] = slope_limited(hm, hpr[i], hp; limiter=limiter)
            su[i] = slope_limited(um, upr[i], up; limiter=limiter)
        else
            im, ip = i-1, i+1
            sh[i] = slope_limited(hpr[im], hpr[i], hpr[ip]; limiter=limiter)
            su[i] = slope_limited(upr[im], upr[i], upr[ip]; limiter=limiter)
        end
    end

    # 3) interface fluxes (N+1)
    amax = 0.0
    # indices for Fhat dimension 1
    If = axes(Fhat,1); iF1, iFN = first(If), last(If)

    # left boundary: ghost | cell fi  → interface iF1
    let
        hL = hpr[fi]; uL = -upr[fi]                       # ghost
        hR = max(HMIN, hpr[fi] - 0.5*sh[fi])              # cell face (right side)
        uR = capu(upr[fi] - 0.5*su[fi])
        f = (solver === :hll) ? hll_flux(hL,uL,hR,uR) : rusanov_flux(hL,uL,hR,uR)
        Fhat[iF1,1] = f[1]; Fhat[iF1,2] = f[2]
        amax = max(amax, abs(uL)+sqrt(g*hL), abs(uR)+sqrt(g*hR))
    end

    # interior interfaces k = 2..N  map to between cell (k-1) and k
    @inbounds for k in (iF1+1):(iFN-1)
        il = fi + (k - iF1) - 1    # left cell index
        ir = il + 1                # right cell index
        hL = max(HMIN, hpr[il] + 0.5*sh[il])
        uL = capu(upr[il] + 0.5*su[il])
        hR = max(HMIN, hpr[ir] - 0.5*sh[ir])
        uR = capu(upr[ir] - 0.5*su[ir])
        f = (solver === :hll) ? hll_flux(hL,uL,hR,uR) : rusanov_flux(hL,uL,hR,uR)
        Fhat[k,1] = f[1]; Fhat[k,2] = f[2]
        amax = max(amax, abs(uL)+sqrt(g*hL), abs(uR)+sqrt(g*hR))
    end

    # right boundary: cell li | ghost  → interface iFN
    let
        hL = max(HMIN, hpr[li] + 0.5*sh[li])              # cell face (left side)
        uL = capu(upr[li] + 0.5*su[li])
        hR = hpr[li]; uR = -upr[li]                       # ghost
        f = (solver === :hll) ? hll_flux(hL,uL,hR,uR) : rusanov_flux(hL,uL,hR,uR)
        Fhat[iFN,1] = f[1]; Fhat[iFN,2] = f[2]
        amax = max(amax, abs(uL)+sqrt(g*hL), abs(uR)+sqrt(g*hR))
    end

    return amax
end

# ------------------ Euler step (reflective) ----------------
@inline function euler_step_reflective!(hout, mout, h, m, Fhat, dt_dx, x, t, source_fun)
    inds = axes(h,1); fi = firstindex(h)
    # interface indexing: for cell i, left interface kL = (i - fi) + first(Fhat axes), right = kL+1
    offset = first(axes(Fhat,1)) - fi
    @inbounds for i in inds
        kL = i + offset
        kR = kL + 1
        hout[i] = h[i] - dt_dx*(Fhat[kR,1] - Fhat[kL,1])
        mout[i] = m[i] - dt_dx*(Fhat[kR,2] - Fhat[kL,2])
    end
    # source term (in-place)
    source_fun(hout, mout, x, t)
    # positivity / dry fix
    @inbounds for i in inds
        if hout[i] < HMIN
            hout[i] = HMIN
            mout[i] = 0.0
        end
    end
end

# ------------------ Public API -----------------------------
function default_ic_dambreak(x; hl=1.0, hr=0.0, ul=0.0, ur=0.0, xc=0.5)
    h = map(xi-> (xi < xc ? hl : hr), x)
    u = map(xi-> (xi < xc ? ul : ur), x)
    return h, u
end

function default_ic_sine(x; h0=1.0, amp=0.1, u0=0.0)
    h = @. h0 + amp*sin(2π*x)
    u = fill(u0, length(x))
    return h, u
end

default_source_zero(h, m, x, t) = nothing

"""
    sw_muscl_hll(N, L, T; CFL=0.4, limiter=:mc, solver=:hll,
                 ic_fun=default_ic_sine, source_fun=default_source_zero)

1D shallow water with MUSCL + HLL/Rusanov, SSPRK2, **reflective boundaries**.
Returns (x, h, m) at time T.
"""
function sw_muscl_hll(N, L, T; CFL=0.4, limiter::Symbol=:mc, solver::Symbol=:hll,
                      ic_fun = default_ic_sine, source_fun = default_source_zero)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx

    hvec, uvec = ic_fun(x)
    h = collect(hvec)
    m = similar(h); @inbounds for i in eachindex(h); m[i] = h[i]*uvec[i]; end

    Fhat = zeros(eltype(h), length(h)+1, 2)  # N+1 interfaces for reflective BCs
    h1 = similar(h); m1 = similar(h)
    h2 = similar(h); m2 = similar(h)

    t = 0.0
    while t < T - eps()
        amax = build_fluxes_reflective!(Fhat, h, m; limiter=limiter, solver=solver)
        dt = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        dt_dx = dt/dx

        euler_step_reflective!(h1,m1, h,m, Fhat, dt_dx, x, t, source_fun)
        _ = build_fluxes_reflective!(Fhat, h1, m1; limiter=limiter, solver=solver)
        euler_step_reflective!(h2,m2, h1,m1, Fhat, dt_dx, x, t+dt, source_fun)

        @inbounds for i in eachindex(h)
            h[i] = 0.5*(h[i] + h2[i])
            m[i] = 0.5*(m[i] + m2[i])
        end
        t += dt
    end
    return x, h, m
end

function sw_snapshots(N, L, times; CFL=0.4, limiter::Symbol=:mc, solver::Symbol=:hll,
                      ic_fun = default_ic_sine, source_fun = default_source_zero)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx

    hvec, uvec = ic_fun(x)
    h = collect(hvec)
    m = similar(h); @inbounds for i in eachindex(h); m[i] = h[i]*uvec[i]; end

    Fhat = zeros(eltype(h), length(h)+1, 2)
    h1 = similar(h); m1 = similar(h)
    h2 = similar(h); m2 = similar(h)

    times = sort(collect(times))
    results = Dict{Float64, Tuple{Vector{Float64},Vector{Float64}}}()
    t = 0.0
    if !isempty(times) && isapprox(times[1], 0.0; atol=1e-15)
        results[0.0] = (copy(h), copy(m))
        times = times[2:end]
    end

    while !isempty(times)
        target = first(times)
        while t < target - eps()
            amax = build_fluxes_reflective!(Fhat, h, m; limiter=limiter, solver=solver)
            dt = (amax > 0) ? min(CFL*dx/amax, target - t) : (target - t)
            dt_dx = dt/dx

            euler_step_reflective!(h1,m1, h,m, Fhat, dt_dx, x, t, source_fun)
            _ = build_fluxes_reflective!(Fhat, h1, m1; limiter=limiter, solver=solver)
            euler_step_reflective!(h2,m2, h1,m1, Fhat, dt_dx, x, t+dt, source_fun)

            @inbounds for i in eachindex(h)
                h[i] = 0.5*(h[i] + h2[i])
                m[i] = 0.5*(m[i] + m2[i])
            end
            t += dt
        end
        results[target] = (copy(h), copy(m))
        times = times[2:end]
    end
    return x, results
end

function animate_sw(N, L, T; CFL=0.4, limiter::Symbol=:mc, solver::Symbol=:hll,
                    ic_fun = default_ic_sine, source_fun = default_source_zero,
                    fps::Integer=30, ylim_h=(0.0,2.0), ylim_u=(-2.0,2.0),
                    path::AbstractString="shallow_water.gif")
    x, h, m = sw_muscl_hll(N, L, 0.0; CFL=CFL, limiter=limiter, solver=solver,
                           ic_fun=ic_fun, source_fun=source_fun)
    anim = Animation()

    t = 0.0
    dx = L/N
    Fhat = zeros(eltype(h), length(h)+1, 2)
    h1 = similar(h); m1 = similar(h)
    h2 = similar(h); m2 = similar(h)

    while t < T - eps()
        # plot frame
        u = similar(h); @inbounds for i in eachindex(h); u[i] = (h[i] > HMIN) ? m[i]/h[i] : 0.0; end
        p1 = plot(x, h, xlabel="x", ylabel="h", ylim=ylim_h, label=false, title="t=$(round(t,digits=3))")
        p2 = plot(x, u, xlabel="x", ylabel="u", ylim=ylim_u, label=false)
        p  = plot(p1, p2, layout=(2,1))
        frame(anim, p)

        # advance one SSPRK2 step
        amax = build_fluxes_reflective!(Fhat, h, m; limiter=limiter, solver=solver)
        dt = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        dt_dx = dt/dx

        euler_step_reflective!(h1,m1, h,m, Fhat, dt_dx, x, t, source_fun)
        _ = build_fluxes_reflective!(Fhat, h1, m1; limiter=limiter, solver=solver)
        euler_step_reflective!(h2,m2, h1,m1, Fhat, dt_dx, x, t+dt, source_fun)

        @inbounds for i in eachindex(h)
            h[i] = 0.5*(h[i] + h2[i])
            m[i] = 0.5*(m[i] + m2[i])
        end
        t += dt
    end

    gif(anim, path, fps=fps)
    return x, h, m
end

end # module
