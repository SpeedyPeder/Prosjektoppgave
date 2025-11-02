# -------- Stoker wet–wet dam-break analytic solution (no extra packages) --------
module DambreakAnalytic
export solve_hm, stoker_solution

const g = 9.81

"""
    solve_hm(hl, hr; g=9.81)

Solve for hm in (hr, hl] from:
    2(√(g hl) - √(g hm)) = (hm - hr) * sqrt( g(hm + hr)/(2 hm hr) )
Uses robust bisection.
"""
function solve_hm(hl, hr; g=9.81, itmax=100, tol=1e-12)
    @assert hl > 0 && hr > 0 && hl >= hr "Need hl >= hr > 0"
    cL = sqrt(g*hl)
    f(hm) = 2*(cL - sqrt(g*hm)) - (hm - hr) * sqrt( g*(hm + hr)/(2*hm*hr) )

    a, b = max(hr*1.0000001, hr + eps(hr)), hl
    fa, fb = f(a), f(b)
    # If numerical issues near hr, expand a bit
    if sign(fa) == sign(fb)
        a = hr*0.999999 + 1e-12; fa = f(a)
    end
    @assert sign(fa) != sign(fb) "Root not bracketed. Check hl, hr."

    for _ in 1:itmax
        m = 0.5*(a+b)
        fm = f(m)
        if abs(fm) < tol || 0.5*(b-a) < tol*max(1.0,abs(m))
            return m
        end
        if sign(fm) == sign(fa)
            a, fa = m, fm
        else
            b, fb = m, fm
        end
    end
    return 0.5*(a+b)
end

"""
    stoker_solution(x, t; hl, hr, x0=0.5, g=9.81)

Return (h, u) arrays at time t>0 for the wet–wet dam-break with initial step at x0.
"""
function stoker_solution(x::AbstractVector, t::Real; hl::Real, hr::Real, x0::Real=0.5, g::Real=9.81)
    @assert t ≥ 0
    h  = similar(x, Float64)
    u  = similar(x, Float64)

    if t == 0
        @inbounds for i in eachindex(x)
            h[i] = x[i] < x0 ? hl : hr
            u[i] = 0.0
        end
        return h, u
    end

    cL = sqrt(g*hl)
    hm = solve_hm(hl, hr; g=g)
    cm = sqrt(g*hm)
    um = 2*(cL - cm)                  # from rarefaction
    s  = (hm * um) / (hm - hr)        # shock speed

    xA = x0 - cL * t
    xB = x0 + (um - cm) * t           # = x0 + (2cL - 3cm) t
    xC = x0 + s * t
    
    @inbounds for i in eachindex(x)
        ξ = (x[i] - x0)/t
        if x[i] <= xA
            h[i] = hl; u[i] = 0.0
        elseif x[i] <= xB
            cξ   = (2cL - ξ)/3
            h[i] = (cξ*cξ)/g
            u[i] = 2*(cL - cξ)
        elseif x[i] <= xC
            h[i] = hm; u[i] = um
        else
            h[i] = hr; u[i] = 0.0
        end
    end
    return h, u
end

end # module
