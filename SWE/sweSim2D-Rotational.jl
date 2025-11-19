module RotSW_CDKLM

export Params, State,
       initialize_state, set_periodic!, set_reflective!, set_outflow!,
       step_rk2!

# =========================
# Parameters & State
# =========================
struct Params
    nx::Int; ny::Int                 # number of cell centers including 1-cell halo on each side
    dx::Float64; dy::Float64         # grid spacings
    g::Float64                       # gravity
    dt::Float64                      # time step
    limiter::Symbol                  # slope limiter: :minmod or :vanalbada
    bc::Symbol                       # :periodic or :reflective
end

"Conservative variables and work arrays."
mutable struct State
    h::Array{Float64,2}                  # depth
    qx::Array{Float64,2}                 # x-momentum = h*u
    qy::Array{Float64,2}                 # y-momentum = h*v
    b::Array{Float64,2}                  # bathymetry
    f::Array{Float64,2}                  # Coriolis parameter
    η::Array{Float64,2}                  # helper for h+b (reconstruction variable)
    Fx::Array{NTuple{3,Float64},2}       # x-face fluxes, size (nx-1, ny)
    Gy::Array{NTuple{3,Float64},2}       # y-face fluxes, size (nx, ny-1)
end

# =========================
# Limiters
# =========================
@inline function _minmod(a,b)
    (a*b <= 0.0) && return 0.0
    return copysign(min(abs(a),abs(b)), a)
end

@inline function _vanalbada(a,b; eps=1e-12)
    num = (a^2 + eps)*b + (b^2 + eps)*a
    den = a^2 + b^2 + 2eps
    return den == 0 ? 0.0 : num/den
end

@inline function slope_limited(vL, vC, vR, limiter::Symbol)
    dl = vC - vL
    dr = vR - vC
    limiter === :minmod    && return _minmod(dl,dr)
    limiter === :vanalbada && return _vanalbada(dl,dr)
    error("Unknown limiter $(limiter). Use :minmod or :vanalbada.")
end


# =========================
# Reconstruction
# =========================
#Bathymetry
function build_Btilde(x, y, bfun)
    Nx, Ny = length(x), length(y)
    dx = x[2]-x[1]
    dy = y[2]-y[1]
    #1) build bathymetry at corners
    xC = range(x[1]-dx/2, x[end]+dx/2, length=Nx+1)
    yC = range(y[1]-dy/2, y[end]+dy/2, length=Ny+1)
    Bcorner = [bfun(xC[j], yC[k]) for j in 1:Nx+1, k in 1:Ny+1 ]
    #2) build bathymetry at face midpoints
    Bfx = Array{Float64}(undef, Nx+1, Ny)
    for j in 1:Nx+1, k in 1:Ny
        Bfx[j,k] = 0.5*(Bcorner[j,k+1] + Bcorner[j,k]) 
    end
    Bfy = Array{Float64}(undef, Nx, Ny+1)
    for j in 1:Nx, k in 1:Ny+1
        Bfy[j,k] = 0.5*(Bcorner[j+1,k] + Bcorner[j,k]) 
    end
    #3) build bathymetry at cell centers
    Bc = Array{Float64}(undef, Nx, Ny)
    for j in 1:Nx, k in 1:Ny
        Bc[j,k] = 0.25*(Bfx[j,k] + Bfx[j+1,k] + Bfy[j,k] + Bfy[j,k+1]) #(j,k) → (j-1/2,k-1/2)
    end
    return Bc, Bfx, Bfy, dx, dy
end

#Velocities
function build_velocities(x, y, h, qx, qy, Hmin)
    Nx, Ny = length(x), length(y)
    u = Array{Float64}(undef, Nx, Ny)
    v = Array{Float64}(undef, Nx, Ny)
    for j in 1:Nx, k in 1:Ny
        if h[j,k] < Hmin #Desingularize
            dx = x[2]-x[1]
            epsilon = (dx)^4
            u[j,k] = (sqrt(2)*h[j,k]*qx[j,k]) / sqrt(h[j,k]^4 + max(h[j,k]^4, epsilon))
            v[j,k] = (sqrt(2)*h[j,k]*qy[j,k]) / sqrt(h[j,k]^4 + max(h[j,k]^4, epsilon))
            #Recalculate discharge
            qx[j,k] = h[j,k]*u[j,k]
            qy[j,k] = h[j,k]*v[j,k]
        else
            u[j,k] = qx[j,k] / (h[j,k])
            v[j,k] = qy[j,k] / (h[j,k])
    end
    return u, v
end

#Build Coriolis parameter f = f_hat + \beta y_k
function build_f(x,y,f_hat, beta)
    Nx, Ny = length(x), length(y)
    f = Array{Float64}(undef, Nx, Ny)
    for j in 1:Nx, k in 1:Ny
        f[j,k] = f_hat + beta * y[k]
    end
    return f
end

# Primitives U_y = f/g * u and V_x = f/g * v 
function build_UV_KL(u::AbstractMatrix,
                     v::AbstractMatrix,
                     f::AbstractMatrix,
                     Bc::AbstractMatrix,
                     h::AbstractMatrix,
                     dx::Real, dy::Real, g::Real)

    Nx, Ny = size(u)
    @assert size(v)  == (Nx,Ny)
    @assert size(f)  == (Nx,Ny)
    @assert size(Bc) == (Nx,Ny)
    @assert size(h) == (Nx,Ny)
    
    Uface = zeros(Float64, Nx, Ny+1)  
    for j in 1:Nx
        for k in 1:Ny
            Uface[j,k+1] = Uface[j,k] + (f[j,k]/g) * u[j,k] * dy
        end
    end

    Vface = zeros(Float64, Nx+1, Ny)
    for j in 1:Nx
        for k in 1:Ny
            Vface[j+1,k] = Vface[j,k] + (f[j,k]/g) * v[j,k] * dx
        end
    end

    #Cell-centered U and V:
    Uc = zeros(Float64, Nx, Ny)
    Vc = zeros(Float64, Nx, Ny)
    for j in 1:Nx, k in 1:Ny
        Uc[j,k] = 0.5 * (Uface[j,k]   + Uface[j,k+1])
        Vc[j,k] = 0.5 * (Vface[j,k]   + Vface[j+1,k])
    end

    K = similar(Bc)
    L = similar(Bc)
    for j in 1:Nx, k in 1:Ny
        K[j,k] = g * (h[j,k] + Bc[j,k] - Vc[j,k])
        L[j,k] = g * (h[j,k] + Bc[j,k] + Uc[j,k])
    end
    return Uface, Vface, Uc, Vc, K, L
end

# Compute limited slopes for u, v, K, and L in x- and y-directions
function slopes_p2D!(
    σx_u, σx_v, σx_K, σx_L,
    σy_u, σy_v, σy_K, σy_L,
    ug, vg, Kg, Lg;
    limiter::Symbol = :minmod,
)
    Nx2, Ny2 = size(ug)
    @assert size(vg) == (Nx2, Ny2)
    @assert size(Kg) == (Nx2, Ny2)
    @assert size(Lg) == (Nx2, Ny2)

    Nx = Nx2 - 2          # number of interior cells in x
    Ny = Ny2 - 2          # number of interior cells in y
    @inbounds for i in 2:Nx+1, j in 2:Ny+1
        # x-direction slopes
        σx_u[i,j] = slope_limited(ug[i-1,j], ug[i,j], ug[i+1,j], limiter)
        σx_v[i,j] = slope_limited(vg[i-1,j], vg[i,j], vg[i+1,j], limiter)
        σx_K[i,j] = slope_limited(Kg[i-1,j], Kg[i,j], Kg[i+1,j], limiter)
        σx_L[i,j] = slope_limited(Lg[i-1,j], Lg[i,j], Lg[i+1,j], limiter)

        # y-direction slopes
        σy_u[i,j] = slope_limited(ug[i,j-1], ug[i,j], ug[i,j+1], limiter)
        σy_v[i,j] = slope_limited(vg[i,j-1], vg[i,j], vg[i,j+1], limiter)
        σy_K[i,j] = slope_limited(Kg[i,j-1], Kg[i,j], Kg[i,j+1], limiter)
        σy_L[i,j] = slope_limited(Lg[i,j-1], Lg[i,j], Lg[i,j+1], limiter)
    end
    return nothing
end


##########################
# 2D slopes for u,v,K,L  #
##########################
function slopes_p2D!(
    σx_u, σx_v, σx_K, σx_L,
    σy_u, σy_v, σy_K, σy_L,
    ug, vg, Kg, Lg;
    limiter::Symbol = :minmod,
)
    Nx2, Ny2 = size(ug)
    @assert size(vg) == (Nx2, Ny2)
    @assert size(Kg) == (Nx2, Ny2)
    @assert size(Lg) == (Nx2, Ny2)

    Nx = Nx2 - 2          # number of interior cells in x
    Ny = Ny2 - 2          # number of interior cells in y

    @inbounds for i in 2:Nx+1, j in 2:Ny+1
        # x-direction slopes
        σx_u[i,j] = slope_limited(ug[i-1,j], ug[i,j], ug[i+1,j], limiter)
        σx_v[i,j] = slope_limited(vg[i-1,j], vg[i,j], vg[i+1,j], limiter)
        σx_K[i,j] = slope_limited(Kg[i-1,j], Kg[i,j], Kg[i+1,j], limiter)
        σx_L[i,j] = slope_limited(Lg[i-1,j], Lg[i,j], Lg[i+1,j], limiter)

        # y-direction slopes
        σy_u[i,j] = slope_limited(ug[i,j-1], ug[i,j], ug[i,j+1], limiter)
        σy_v[i,j] = slope_limited(vg[i,j-1], vg[i,j], vg[i,j+1], limiter)
        σy_K[i,j] = slope_limited(Kg[i,j-1], Kg[i,j], Kg[i,j+1], limiter)
        σy_L[i,j] = slope_limited(Lg[i,j-1], Lg[i,j], Lg[i,j+1], limiter)
    end

    return nothing
end

# Reconstruct using piecewise-linear reconstruction with slope limiting
function reconstruct_p(u, v, K, L; limiter::Symbol = :minmod)
    Nx, Ny = size(u)
    @assert size(v) == (Nx,Ny)
    @assert size(K) == (Nx,Ny)
    @assert size(L) == (Nx,Ny)

    # Allocate ghosts and set reflective BCs
    ug = zeros(Float64, Nx+2, Ny+2)
    vg = zeros(Float64, Nx+2, Ny+2)
    Kg = zeros(Float64, Nx+2, Ny+2)
    Lg = zeros(Float64, Nx+2, Ny+2)
    @inbounds begin
        ug[2:Nx+1, 2:Ny+1] .= u
        vg[2:Nx+1, 2:Ny+1] .= v
        Kg[2:Nx+1, 2:Ny+1] .= K
        Lg[2:Nx+1, 2:Ny+1] .= L
    end

    # Set reflective BCs
    @inbounds begin
        # left/right boundaries
        ug[1,  :] .= -ug[2,  :]     
        ug[end,:] .= -ug[end-1, :] 
        vg[1,  :] .=  vg[2,  :]      
        vg[end,:] .=  vg[end-1, :]
        Kg[1,  :] .=  Kg[2,  :]    
        Kg[end,:] .=  Kg[end-1, :]
        Lg[1,  :] .=  Lg[2,  :]
        Lg[end,:] .=  Lg[end-1, :]

        # bottom/top boundaries
        ug[:, 1]  .=  ug[:, 2]     
        ug[:, end].=  ug[:, end-1]
        vg[:, 1]  .= -vg[:, 2]      
        vg[:, end].= -vg[:, end-1]
        Kg[:, 1]  .=  Kg[:, 2]
        Kg[:, end].=  Kg[:, end-1]
        Lg[:, 1]  .=  Lg[:, 2]
        Lg[:, end].=  Lg[:, end-1]
    end

    # Allocate slope arrays
    σx_u = zeros(Float64, Nx+2, Ny+2); σx_v = zeros(Float64, Nx+2, Ny+2); σx_K = zeros(Float64, Nx+2, Ny+2); σx_L = zeros(Float64, Nx+2, Ny+2)
    σy_u = zeros(Float64, Nx+2, Ny+2); σy_v = zeros(Float64, Nx+2, Ny+2); σy_K = zeros(Float64, Nx+2, Ny+2); σy_L = zeros(Float64, Nx+2, Ny+2)

    #Compute slopes
    slopes_p2D!(σx_u, σx_v, σx_K, σx_L,
                σy_u, σy_v, σy_K, σy_L,
                ug, vg, Kg, Lg; limiter=limiter)

    # Allocate interface arrays 
    uE = similar(u); uW = similar(u); uN = similar(u); uS = similar(u)
    vE = similar(v); vW = similar(v); vN = similar(v); vS = similar(v)
    KE = similar(K); KW = similar(K); KN = similar(K); KS = similar(K)
    LE = similar(L); LW = similar(L); LN = similar(L); LS = similar(L)

    # Reconstruct
    @inbounds for i in 1:Nx, j in 1:Ny
        #Shifted indices for ghosts
        I = i + 1
        J = j + 1

        # x-faces
        uE[i,j] = ug[I,J] + 0.5*σx_u[I,J]
        uW[i,j] = ug[I,J] - 0.5*σx_u[I,J]
        vE[i,j] = vg[I,J] + 0.5*σx_v[I,J]
        vW[i,j] = vg[I,J] - 0.5*σx_v[I,J]
        KE[i,j] = Kg[I,J] + 0.5*σx_K[I,J]
        KW[i,j] = Kg[I,J] - 0.5*σx_K[I,J]
        LE[i,j] = Lg[I,J] + 0.5*σx_L[I,J]
        LW[i,j] = Lg[I,J] - 0.5*σx_L[I,J]

        # y-faces
        uN[i,j] = ug[I,J] + 0.5*σy_u[I,J]
        uS[i,j] = ug[I,J] - 0.5*σy_u[I,J]
        vN[i,j] = vg[I,J] + 0.5*σy_v[I,J]
        vS[i,j] = vg[I,J] - 0.5*σy_v[I,J]
        KN[i,j] = Kg[I,J] + 0.5*σy_K[I,J]
        KS[i,j] = Kg[I,J] - 0.5*σy_K[I,J]
        LN[i,j] = Lg[I,J] + 0.5*σy_L[I,J]
        LS[i,j] = Lg[I,J] - 0.5*σy_L[I,J]
    end

    return uE,uW,uN,uS,
           vE,vW,vN,vS,
           KE,KW,KN,KS,
           LE,LW,LN,LS
end

function reconstruct_h(h, Uf, Vf, KE, KW, LN, LS, Bfx, Bfy)
    Nx, Ny = size(h)
    @assert size(Uf) == (Nx, Ny+1)
    @assert size(Vf) == (Nx+1, Ny)
    @assert size(KE)  == (Nx, Ny)
    @assert size(KW)  == (Nx, Ny)
    @assert size(LN)  == (Nx, Ny)
    @assert size(LS)  == (Nx, Ny)
    @assert size(Bfx) == (Nx+1, Ny)
    @assert size(Bfy) == (Nx, Ny+1)
    #Should not need ghosts here because h are reconstructed using K and L which have already been reconstructed with ghosts
    hE = similar(h); hW = similar(h); hN = similar(h); hS = similar(h)
    @inbounds for j in 1:Nx, k in 1:Ny
        hE[j,k] = KE[j,k]/g + Vf[j+1,k] - Bfx[j+1,k]
        hW[j,k] = KW[j,k]/g + Vf[j,k]   - Bfx[j,k]
        hN[j,k] = LN[j,k]/g - Uf[j,k+1] - Bfy[j,k+1]
        hS[j,k] = LS[j,k]/g - Uf[j,k]   - Bfy[j,k]
    end
    return hE,hW,hN,hS
end

# =========================
# Fluxes (central-upwind)
# =========================
#Build F for the reconstructed variables
"""
    flux_x_cu!(F1, F2, F3,
               hE, hW, uE, uW, vE, vW, g)

Compute x-direction CU fluxes F = (F^(1),F^(2),F^(3)) at faces x_{i+1/2,j},
using reconstructed interface states from the KP07 reconstruction:

  left state  at i+1/2,j : (hE[i,j], uE[i,j], vE[i,j])
  right state at i+1/2,j : (hW[i+1,j], uW[i+1,j], vW[i+1,j])

Array sizes:
  hE,hW,uE,uW,vE,vW :: (Nx, Ny)
  F1,F2,F3          :: (Nx+1, Ny)   (faces i = 1..Nx+1, between cells)

We only fill interior faces i = 2..Nx (between cell i-1 and i).
Boundary faces (i=1 and i=Nx+1) should be set by BCs.
"""
function build_F!(F1, F2, F3,
                    hE, hW, uE, uW, vE, vW, g)
    Nx, Ny = size(hE)
    @assert size(hW) == (Nx,Ny)
    @assert size(uE) == (Nx,Ny) == size(uW) == size(vE) == size(vW)
    @assert size(F1) == (Nx+1,Ny)
    @assert size(F2) == (Nx+1,Ny)
    @assert size(F3) == (Nx+1,Ny)
    @inbounds for i in 1:Nx-1, j in 1:Ny
        #Calculate left and right states
        hL = hE[i,j]; uL = uE[i,j]; vL = vE[i,j]
        hR = hW[i+1,j]; uR = uW[i+1,j]; vR = vW[i+1,j]
        cL = sqrt(g*max(hL,0.0))
        cR = sqrt(g*max(hR,0.0))
        #One sided speeds
        ap = max(0.0, uL + cL, uR + cR)
        am = min(0.0, uL - cL, uR - cR)
        denom = ap - am

        fi = i+1
        if denom <= 1e-14 #If difference in wave speeds is too small, set fluxes to zero
            F1[fi,j] = 0.0
            F2[fi,j] = 0.0
            F3[fi,j] = 0.0
            continue
        end
        #Calculate discharges
        qxL = hL*uL; qxR = hR*uR
        FL1 = qxL
        FL2 = qxL*uL + 0.5*g*hL*hL
        FR1 = qxR
        FR2 = qxR*uR + 0.5*g*hR*hR

        # F^(1), F^(2): CU (3.2)–(3.3)
        F1[fi,j] = (ap*FL1 - am*FR1)/denom + (ap*am/denom)*(hR  - hL)
        F2[fi,j] = (ap*FL2 - am*FR2)/denom + (ap*am/denom)*(qxR - qxL)

        # F^(3): upwind hvu (3.4)
        if uL + uR > 0.0
            F3[fi,j] = hL*uL*vL
        else
            F3[fi,j] = hR*uR*vR
        end
    end

    return nothing
end

"""
Build y-face fluxes Gy[i,j] for the face between cells j and j+1 (at fixed i).
Same reconstruction logic as before, but in y-direction.
"""
function _flux_y!(Gy, h,qx,qy,b,η, p::Params)
    nx,ny = size(h); g = p.g
    if p.bc === :periodic
        for i in 1:nx, j in 1:ny-1
            jm1 = wrap(j-1, ny)
            jp1 = wrap(j+1, ny)
            jp2 = wrap(j+2, ny)
            # Reconstruction in up- and down directions
            sD     = slope_limited(η[i,jm1], η[i,j],   η[i,jp1], p.limiter)
            sU     = slope_limited(η[i,j],   η[i,jp1], η[i,jp2], p.limiter)
            ηDface = η[i,j]   + 0.5*sD
            ηUface = η[i,jp1] - 0.5*sU
            hD = max(ηDface - b[i,j],    0.0)
            hU = max(ηUface - b[i,jp1],  0.0)
            qxD, qxU = qx[i,j],   qx[i,jp1]
            qyD, qyU = qy[i,j],   qy[i,jp1]
            # Velocities in y-direction
            uD = hD>0 ? qxD/hD : 0.0;  vD = hD>0 ? qyD/hD : 0.0
            uU = hU>0 ? qxU/hU : 0.0;  vU = hU>0 ? qyU/hU : 0.0
            cD = _c(g,hD); cU = _c(g,hU)
            bp = max(0.0, max(vD + cD, vU + cU))
            bm = min(0.0, min(vD - cD, vU - cU))
            denom = bp - bm
            if denom <= 1e-14
                Gy[i,j] = (0.0,0.0,0.0)
                continue
            end
            # Physical fluxes G(U) in y-direction
            GL1, GL2, GL3 = qyD, (hD>0 ? qxD*vD : 0.0), (hD>0 ? qyD*vD + 0.5*g*hD*hD : 0.0)
            GR1, GR2, GR3 = qyU, (hU>0 ? qxU*vU : 0.0), (hU>0 ? qyU*vU + 0.5*g*hU*hU : 0.0)
            g1 = (bp*GL1 - bm*GR1)/denom + (bp*bm/denom)*(hU  - hD)
            g2 = (bp*GL2 - bm*GR2)/denom + (bp*bm/denom)*(qxU - qxD)
            g3 = (bp*GL3 - bm*GR3)/denom + (bp*bm/denom)*(qyU - qyD)
            Gy[i,j] = (g1,g2,g3)  # face j+1/2 stored at index j
        end
    else
        # Outflow/reflective: compute ALL faces 1..ny-1 with clamped stencil as in x-direction
        for i in 1:nx, j in 1:ny-1
            jm1 = max(j-1, 1)
            jp1 = j+1            
            jp2 = min(j+2, ny)
            sD     = slope_limited(η[i,jm1], η[i,j],   η[i,jp1], p.limiter)
            sU     = slope_limited(η[i,j],   η[i,jp1], η[i,jp2], p.limiter)
            ηDface = η[i,j]   + 0.5*sD
            ηUface = η[i,jp1] - 0.5*sU
            hD = max(ηDface - b[i,j],    0.0)
            hU = max(ηUface - b[i,jp1],  0.0)
            qxD, qxU = qx[i,j],   qx[i,jp1]
            qyD, qyU = qy[i,j],   qy[i,jp1]
            uD = hD>0 ? qxD/hD : 0.0;  vD = hD>0 ? qyD/hD : 0.0
            uU = hU>0 ? qxU/hU : 0.0;  vU = hU>0 ? qyU/hU : 0.0
            cD = _c(p.g,hD); cU = _c(p.g,hU)
            bp = max(0.0, max(vD + cD, vU + cU))
            bm = min(0.0, min(vD - cD, vU - cU))
            denom = bp - bm
            if denom <= 1e-14
                Gy[i,j] = (0.0,0.0,0.0)
                continue
            end
            # Fluxes G(U) in y-direction
            GL1, GL2, GL3 = qyD, (hD>0 ? qxD*vD : 0.0), (hD>0 ? qyD*vD + 0.5*p.g*hD*hD : 0.0)
            GR1, GR2, GR3 = qyU, (hU>0 ? qxU*vU : 0.0), (hU>0 ? qyU*vU + 0.5*p.g*hU*hU : 0.0)
            g1 = (bp*GL1 - bm*GR1)/denom + (bp*bm/denom)*(hU  - hD)
            g2 = (bp*GL2 - bm*GR2)/denom + (bp*bm/denom)*(qxU - qxD)
            g3 = (bp*GL3 - bm*GR3)/denom + (bp*bm/denom)*(qyU - qyD)
            Gy[i,j] = (g1,g2,g3)
        end
    end
    return
end

# =========================
# Residual: flux divergence + sources
# =========================
function residual!(dh, dqx, dqy, st::State, p::Params;
                   stageq::Union{Nothing,Tuple{AbstractMatrix{<:Real},AbstractMatrix{<:Real}}}=nothing)
    h,qx,qy,b,f,η = st.h, st.qx, st.qy, st.b, st.f, st.η
    nx,ny = size(h)
    g,dx,dy = p.g, p.dx, p.dy
    #1) Apply BCs
    if p.bc === :periodic
        set_periodic!(st)
    elseif p.bc === :reflective
        set_reflective!(st)
    elseif p.bc === :outflow
        set_outflow!(st) 
    else
        error("Unknown bc=$(p.bc). Use :periodic, :reflective, or :outflow.")
    end
    #2) Recompute η = h + b for reconstruction
    for j=1:ny, i=1:nx
        η[i,j] = h[i,j] + b[i,j]
    end
    #3) Build face fluxes via central-upwind scheme
    _flux_x!(st.Fx, h,qx,qy,b,η,p)  # Fx[i,j] 
    _flux_y!(st.Gy, h,qx,qy,b,η,p)  # Gy[i,j] 

    #4)Flux divergence for interior cells
    for j=2:ny-1, i=2:nx-1
        FxL = st.Fx[i-1,j]; FxR = st.Fx[i,j]
        GyD = st.Gy[i,j-1]; GyU = st.Gy[i,j]
        dh[i,j]  = - (FxR[1]-FxL[1])/dx - (GyU[1]-GyD[1])/dy
        dqx[i,j] = - (FxR[2]-FxL[2])/dx - (GyU[2]-GyD[2])/dy
        dqy[i,j] = - (FxR[3]-FxL[3])/dx - (GyU[3]-GyD[3])/dy
    end

    #5) Add in source terms for bathymetry and Coriolis
    for j=2:ny-1, i=2:nx-1
        dbdx = (b[i+1,j]-b[i-1,j])/(2*dx)
        dbdy = (b[i,j+1]-b[i,j-1])/(2*dy)
        dqx[i,j] += - g*h[i,j]*dbdx
        dqy[i,j] += - g*h[i,j]*dbdy
        # Coriolis (skew-symmetric avg if in RK2 stage)
        if stageq === nothing
            qxavg = qx[i,j];  qyavg = qy[i,j]
        else
            qx_stage, qy_stage = stageq
            qxavg = 0.5*(qx[i,j] + qx_stage[i,j])
            qyavg = 0.5*(qy[i,j] + qy_stage[i,j])
        end
        #Update momentum from Coriolis
        dqx[i,j] +=  + f[i,j]*qyavg
        dqy[i,j] +=  - f[i,j]*qxavg
    end
    return
end

# =========================
# Time integration: SSP-RK2 (Heun)
# =========================
function step_rk2!(st::State, p::Params)
    nx,ny = size(st.h)
    dh  = zeros(nx,ny); dqx = zeros(nx,ny); dqy = zeros(nx,ny)
    h1  = similar(st.h); qx1 = similar(st.qx); qy1 = similar(st.qy)

    # Stage 1: U¹ = Uⁿ + dt * R(Uⁿ)
    residual!(dh,dqx,dqy, st,p)
    for j=2:ny-1, i=2:nx-1
        h1[i,j]  = st.h[i,j]  + p.dt*dh[i,j]
        qx1[i,j] = st.qx[i,j] + p.dt*dqx[i,j]
        qy1[i,j] = st.qy[i,j] + p.dt*dqy[i,j]
    end
    tmp = State(h1,qx1,qy1, st.b, st.f, st.η, st.Fx, st.Gy)

    # Stage 2: Uⁿ⁺¹ = ½( Uⁿ + U¹ + dt * R(U¹) ), with skew-sym avg in Coriolis
    dh .= 0; dqx .= 0; dqy .= 0
    stageq = (st.qx, st.qy)         # pass original momenta for skew-sym split
    residual!(dh,dqx,dqy, tmp,p; stageq=stageq)
    for j=2:ny-1, i=2:nx-1
        st.h[i,j]  = 0.5*(st.h[i,j]  + h1[i,j]  + p.dt*dh[i,j])
        st.qx[i,j] = 0.5*(st.qx[i,j] + qx1[i,j] + p.dt*dqx[i,j])
        st.qy[i,j] = 0.5*(st.qy[i,j] + qy1[i,j] + p.dt*dqy[i,j])
    end
    return
end

end # module
