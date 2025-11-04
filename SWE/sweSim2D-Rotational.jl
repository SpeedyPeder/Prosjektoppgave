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
# Boundary helpers
# =========================
function set_periodic!(A::AbstractMatrix)
    nx,ny = size(A)
    A[1,:]  .= A[nx-1,:];   A[nx,:] .= A[2,:]
    A[:,1]  .= A[:,ny-1];   A[:,ny] .= A[:,2]
    return A
end

function set_reflective!(h, qx, qy, b, f)
    nx,ny = size(h)
    # Left/right
    h[1,:]   .=  h[2,:];          h[nx,:]  .=  h[nx-1,:]
    qx[1,:]  .= -qx[2,:];         qx[nx,:] .= -qx[nx-1,:]
    qy[1,:]  .=  qy[2,:];         qy[nx,:] .=  qy[nx-1,:]
    b[1,:]   .=  b[2,:];          b[nx,:]  .=  b[nx-1,:]
    f[1,:]   .=  f[2,:];          f[nx,:]  .=  f[nx-1,:]
    # Bottom/top
    h[:,1]   .=  h[:,2];          h[:,ny]  .=  h[:,ny-1]
    qy[:,1]  .= -qy[:,2];         qy[:,ny] .= -qy[:,ny-1]
    qx[:,1]  .=  qx[:,2];         qx[:,ny] .=  qx[:,ny-1]
    b[:,1]   .=  b[:,2];          b[:,ny]  .=  b[:,ny-1]
    f[:,1]   .=  f[:,2];          f[:,ny]  .=  f[:,ny-1]
    return
end

function set_outflow!(h, qx, qy, b, f; no_inflow::Bool=true)
    nx, ny = size(h)
    # Left/right: copy values
    h[1,:]  .= h[2,:];       h[nx,:] .= h[nx-1,:]
    qx[1,:] .= qx[2,:];      qx[nx,:] .= qx[nx-1,:]
    qy[1,:] .= qy[2,:];      qy[nx,:] .= qy[nx-1,:]
    b[1,:]  .= b[2,:];       b[nx,:]  .= b[nx-1,:]
    f[1,:]  .= f[2,:];       f[nx,:]  .= f[nx-1,:]
    # Bottom/top: copy values
    h[:,1]  .= h[:,2];       h[:,ny] .= h[:,ny-1]
    qy[:,1] .= qy[:,2];      qy[:,ny] .= qy[:,ny-1]
    qx[:,1] .= qx[:,2];      qx[:,ny] .= qx[:,ny-1]
    b[:,1]  .= b[:,2];       b[:,ny]  .= b[:,ny-1]
    f[:,1]  .= f[:,2];       f[:,ny]  .= f[:,ny-1]

    if no_inflow
        @. qx[1,:] = min(qx[1,:], 0.0)
        @. qx[nx,:] = max(qx[nx,:], 0.0)
        @. qy[:,1] = min(qy[:,1], 0.0)
        @. qy[:,ny] = max(qy[:,ny], 0.0)
    end
    return
end


set_periodic!(st::State)   = (set_periodic!(st.h); set_periodic!(st.qx); set_periodic!(st.qy);
                             set_periodic!(st.b); set_periodic!(st.f); set_periodic!(st.η))
set_reflective!(st::State) = set_reflective!(st.h, st.qx, st.qy, st.b, st.f)
set_outflow!(st::State, kwargs...) = set_outflow!(st.h, st.qx, st.qy, st.b, st.f, kwargs...)

# =========================
# Allocation / init
# =========================
function initialize_state(nx,ny)
    H   = zeros(nx,ny); QX = zeros(nx,ny); QY = zeros(nx,ny)
    B   = zeros(nx,ny); F  = zeros(nx,ny); ETA = zeros(nx,ny)
    Fx  = fill((0.0,0.0,0.0), nx-1, ny)   # faces i=1..nx-1
    Gy  = fill((0.0,0.0,0.0), nx,   ny-1) # faces j=1..ny-1
    return State(H,QX,QY,B,F,ETA,Fx,Gy)
end

# =========================
# Fluxes (central-upwind)
# =========================
@inline _c(g,h) = sqrt(g*max(h,0.0))

# periodic index wrap to 1..n (SAFE; mod not %)
@inline wrap(i, n) = mod(i - 1, n) + 1

"""
Build x-face fluxes Fx[i,j] for the face between cells i and i+1.

Reconstruction variable is η = h + b (well-balanced friendly).
Left state at i+1/2  uses slope at cell i   from (i-1, i, i+1).
Right state at i+1/2 uses slope at cell i+1 from (i, i+1, i+2).

Speeds:
  a⁺ = max(0, u_L + c_L, u_R + c_R),
  a⁻ = min(0, u_L - c_L, u_R - c_R)

Central-upwind flux (KT/KP):
  F̂ = (a⁺ F(U_L) - a⁻ F(U_R)) / (a⁺-a⁻) + (a⁺ a⁻)/(a⁺-a⁻) * (U_R - U_L)
"""
function _flux_x!(Fx, h,qx,qy,b,η, p::Params)
    nx,ny = size(h); g = p.g
    if p.bc === :periodic
        for j in 1:ny, i in 1:nx-1
            im1 = wrap(i-1, nx)    # i-1
            ip1 = wrap(i+1, nx)    # i+1
            ip2 = wrap(i+2, nx)    # i+2
            # Reconstruct η at face i+1/2
            sL     = slope_limited(η[im1,j], η[i,j],   η[ip1,j], p.limiter)
            sR     = slope_limited(η[i,j],   η[ip1,j], η[ip2,j], p.limiter)
            # η at face
            ηLface = η[i,j]   + 0.5*sL
            ηRface = η[ip1,j] - 0.5*sR
            # height
            hL = max(ηLface - b[i,j],    0.0)
            hR = max(ηRface - b[ip1,j],  0.0)
            # Momentum in x and y directions
            qxL, qxR = qx[i,j],   qx[ip1,j]
            qyL, qyR = qy[i,j],   qy[ip1,j]
            # Velocities
            uL = hL>0 ? qxL/hL : 0.0;  vL = hL>0 ? qyL/hL : 0.0
            uR = hR>0 ? qxR/hR : 0.0;  vR = hR>0 ? qyR/hR : 0.0
            cL = _c(g,hL); cR = _c(g,hR)
            # Coefficients for flux update
            ap = max(0.0, max(uL + cL, uR + cR))
            am = min(0.0, min(uL - cL, uR - cR))
            denom = ap - am
            if denom <= 1e-14
                Fx[i,j] = (0.0,0.0,0.0)
                continue
            end

            # Physical fluxes F(U) in x-direction 
            FL1, FL2, FL3 = qxL, (hL>0 ? qxL*uL + 0.5*g*hL*hL : 0.0), (hL>0 ? qxL*vL : 0.0)
            FR1, FR2, FR3 = qxR, (hR>0 ? qxR*uR + 0.5*g*hR*hR : 0.0), (hR>0 ? qxR*vR : 0.0)
            f1 = (ap*FL1 - am*FR1)/denom + (ap*am/denom)*(hR  - hL)
            f2 = (ap*FL2 - am*FR2)/denom + (ap*am/denom)*(qxR - qxL)
            f3 = (ap*FL3 - am*FR3)/denom + (ap*am/denom)*(qyR - qyL)
            Fx[i,j] = (f1,f2,f3)  # Stored i+1/2 at index i
        end
    else
        # Outflow/reflective: compute ALL faces 1..nx-1 with clamped stencil
        for j in 1:ny, i in 1:nx-1
            im1 = max(i-1, 1)
            ip1 = i+1            # in 1..nx
            ip2 = min(i+2, nx)

            # Reconstruct η at face i+1/2 with clamped neighbors
            sL     = slope_limited(η[im1,j], η[i,j],   η[ip1,j], p.limiter)
            sR     = slope_limited(η[i,j],   η[ip1,j], η[ip2,j], p.limiter)
            ηLface = η[i,j]   + 0.5*sL
            ηRface = η[ip1,j] - 0.5*sR
            hL = max(ηLface - b[i,j],    0.0)
            hR = max(ηRface - b[ip1,j],  0.0)
            qxL, qxR = qx[i,j],   qx[ip1,j]
            qyL, qyR = qy[i,j],   qy[ip1,j]
            #Velocities
            uL = hL>0 ? qxL/hL : 0.0;  vL = hL>0 ? qyL/hL : 0.0
            uR = hR>0 ? qxR/hR : 0.0;  vR = hR>0 ? qyR/hR : 0.0
            cL = _c(p.g,hL); cR = _c(p.g,hR)
            ap = max(0.0, max(uL + cL, uR + cR))
            am = min(0.0, min(uL - cL, uR - cR))
            denom = ap - am
            if denom <= 1e-14
                Fx[i,j] = (0.0,0.0,0.0)
                continue
            end
            # Fluxes F(U) in x-direction
            FL1, FL2, FL3 = qxL, (hL>0 ? qxL*uL + 0.5*p.g*hL*hL : 0.0), (hL>0 ? qxL*vL : 0.0)
            FR1, FR2, FR3 = qxR, (hR>0 ? qxR*uR + 0.5*p.g*hR*hR : 0.0), (hR>0 ? qxR*vR : 0.0)
            f1 = (ap*FL1 - am*FR1)/denom + (ap*am/denom)*(hR  - hL)
            f2 = (ap*FL2 - am*FR2)/denom + (ap*am/denom)*(qxR - qxL)
            f3 = (ap*FL3 - am*FR3)/denom + (ap*am/denom)*(qyR - qyL)
            Fx[i,j] = (f1,f2,f3)
        end
    end
    return
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
