
#################### Bruker ikke disse fluxene lenger ##########################
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
######################################################################################

####################################################################################################################################################################
######################## Gammel kode for MUSCL-HLL eller MUSCL-Rusanov. Har byttet ut med KP-Central Upwind flux over ###############################################

"""
    sw_muscl_hll(N, L, T; CFL=0.4, limiter=:mc, solver=:hll,
                 ic_fun=default_ic_sine, source_fun=default_source_zero,
                 bfun=default_bathymetry)

Advance to time T and return (x, h, m).
"""
function sw_muscl_hll(N, L, T; CFL=0.4, limiter::Symbol=:mc, solver::Symbol=:hll,
                      ic_fun=default_ic_sine, source_fun=default_source_zero,
                      bfun=default_bathymetry)
    dx = L/N
    x  = @. (0.5:1:N-0.5) * dx

    hvec, uvec = ic_fun(x)
    h = collect(hvec)
    m = similar(h); @inbounds for i in eachindex(h); m[i] = h[i]*uvec[i]; end

    Fhat = zeros(eltype(h), length(h)+1, 2)
    h1 = similar(h); m1 = similar(h)
    h2 = similar(h); m2 = similar(h)

    t = 0.0
    while t < T - eps()
        # --- Stage A: build bathymetry once and use everywhere
        Bf, Bc, _ = build_Btilde_faces_centers(x, bfun)

        amax = build_fluxes_reflective!(Fhat, h, m, dx; Bf=Bf, Bc=Bc, limiter=limiter)
        dt = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        dt_dx = dt/dx

        # Euler step + KP source using SAME (Bf,Bc)
        @inbounds for i in eachindex(h)
            h1[i] = h[i] - dt_dx*(Fhat[i+1,1] - Fhat[i,1])
            m1[i] = m[i] - dt_dx*(Fhat[i+1,2] - Fhat[i,2])
        end
        source_fun(h1, m1, x, t)
        bathy_source_kp07!(h1, m1, dt_dx, Bf, Bc)

        # --- Stage B: rebuild fluxes at (h1,m1) with SAME rule
        Bf, Bc, _ = build_Btilde_faces_centers(x, bfun)
        _ = build_fluxes_reflective!(Fhat, h1, m1, dx; Bf=Bf, Bc=Bc, limiter=limiter)

        # Second Euler + KP source
        @inbounds for i in eachindex(h)
            h2[i] = h1[i] - dt_dx*(Fhat[i+1,1] - Fhat[i,1])
            m2[i] = m1[i] - dt_dx*(Fhat[i+1,2] - Fhat[i,2])
        end
        source_fun(h2, m2, x, t+dt)
        bathy_source_kp07!(h2, m2, dt_dx, Bf, Bc)

        # Heun average
        @inbounds for i in eachindex(h)
            h[i] = 0.5*(h[i] + h2[i])
            m[i] = 0.5*(m[i] + m2[i])
        end
        t += dt
    end
    return x, h, m
end


# ------------------ Snapshot function ----------------------
function sw_snapshots(N, L, times; CFL=0.4, limiter::Symbol=:mc, solver::Symbol=:hll,
                      ic_fun=default_ic_sine, source_fun=default_source_zero,
                      bfun=default_bathymetry)
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
            b = bfun(x)
            Bf, Bc, _ = build_Btilde_faces_centers(x, bfun)
            amax = build_fluxes_reflective!(Fhat, h, m, dx; Bf=Bf, Bc=Bc, limiter=limiter)
            dt = (amax > 0) ? min(CFL*dx/amax, target - t) : (target - t)
            dt_dx = dt/dx

            euler_step_reflective!(h1,m1, h,m,   Fhat, dt_dx, x, t,    source_fun, Bf, Bc)
            _ = build_fluxes_reflective!(Fhat, h1, m1, dx; Bf=Bf, Bc=Bc, limiter=limiter)
            euler_step_reflective!(h2,m2, h1,m1, Fhat, dt_dx, x, t+dt, source_fun, Bf, Bc)
        

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

# ------------------ Animation function ----------------------
function animate_sw(N, L, T; CFL=0.4, limiter::Symbol=:mc, solver::Symbol=:hll,
                    ic_fun=default_ic_sine, source_fun=default_source_zero, bfun=default_bathymetry,
                    fps::Integer=30, ylim_h=(0.0,2.0), ylim_u=(-2.0,2.0),
                    path::AbstractString="shallow_water.gif")
    @eval using Plots
    x, h, m = sw_muscl_hll(N, L, 0.0; CFL=CFL, limiter=limiter,
                           ic_fun=ic_fun, source_fun=source_fun, bfun=bfun)
    b = bfun(x)

    anim = Animation()
    t = 0.0
    dx = L/N
    Fhat = zeros(eltype(h), length(h)+1, 2)
    h1 = similar(h); m1 = similar(h)
    h2 = similar(h); m2 = similar(h)

    while t < T - eps()
        # frame
        u = similar(h); @inbounds for i in eachindex(h); u[i] = m[i]/h[i] ; end
        η = h .+ b
        p1 = plot(x, η, xlabel="x", ylabel="elevation", ylim=ylim_h, label="η=h+b",
                  title="t=$(round(t,digits=3))")
        plot!(p1, x, b, label="b(x)", ls=:dash)
        p2 = plot(x, u, xlabel="x", ylabel="u", ylim=ylim_u, label=false)
        frame(anim, plot(p1, p2, layout=(2,1)))

        # step
        Bf, Bc, _ = build_Btilde_faces_centers(x, bfun)
        amax = build_fluxes_reflective!(Fhat, h, m, dx; Bf=Bf, Bc=Bc, limiter=limiter)
        dt = (amax > 0) ? min(CFL*dx/amax, T - t) : (T - t)
        dt_dx = dt/dx

        euler_step_reflective!(h1,m1, h,m,   Fhat, dt_dx, x, t,    source_fun, Bf, Bc)
        _ = build_fluxes_reflective!(Fhat, h1, m1, dx; Bf=Bf, Bc=Bc, limiter=limiter)
        euler_step_reflective!(h2,m2, h1,m1, Fhat, dt_dx, x, t+dt, source_fun, Bf, Bc)

        @inbounds for i in eachindex(h)
            h[i] = 0.5*(h[i] + h2[i])
            m[i] = 0.5*(m[i] + m2[i])
        end
        t += dt
    end

    gif(anim, path, fps=fps)
    return x, h, m
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