cd(@__DIR__)

include("sweSim1D.jl")
using .sweSim1D
using Plots
using Printf


# ---------------- Params ----------------
N, L  = 100, 5.0
CFL   = 0.45
T     = 2
lim   = :minmod
# times = 0.0:0.1:T   # (unused here)

# ---------------- Bathymetry (pick ONE) ----------------
#bfun(x::AbstractVector) = 0.20 .* (x ./ L) .+ ifelse.(x .> 0.7L, 0.40, 0.0)    # linear + step
# bfun(x::AbstractVector) = 0.10 .* exp.(-((x .- 0.5L).^2) ./ (0.05L)^2)         # Gaussian bump
bfun(x::AbstractVector) = ifelse.(x .> 0.55L, 0.07, 0.0) .+ ifelse.(x .> 0.82L, 0.05, 0.0) # double step
#bfun(x::AbstractVector) = zeros(length(x))                                       # flat bottom

# ---------------- Initial condition ----------------
# Lake-at-rest: η = η0 const, u = 0  (works for ANY bfun)
η0 = 0.3
function ic_fun(x)
    b = bfun(x)               # bfun is defined *above* now
    h = η0 .- b               # h = η - b
    u = zeros(length(x))
    return h, u
end


function ic_gaussian_surface(x; A=0.2, x0=0.5*(minimum(x)+maximum(x)), σ=0.05)
    b = bfun(x)
    η = η0 .+ A .* exp.(-((x .- x0).^2) ./ (2σ^2))
    h = max.(η .- b, 1e-12)       # enforce positivity: NOT needed here
    u = zeros(length(x))
    return h, u
end

source_fun = sweSim1D.default_source_zero

# Final plot autosaves to KP/final_T=1.0.png
sweSim1D.kp_plot_final(x, η, m, bfun; T= T, filename="run1_final.png")

# Animation autosaves to KP/shallow_water_KP.gif
sweSim1D.animate_sw_KP(N, L, T; ic_fun=ic_gaussian_surface, bfun=bfun, path="run1.gif")


   
