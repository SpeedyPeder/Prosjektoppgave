module RotSW_CDKLM

export Params, State, step_RK2!, init_state

# =========================
# Parameters & State
# =========================
struct Params
    x::Vector{Float64}  # x-grid points
    y::Vector{Float64}  # y-grid points
    nx::Int          # number of physical cells in x
    ny::Int          # number of physical cells in y
    dx::Float64      # grid spacing in x
    dy::Float64      # grid spacing in y
    g::Float64       # gravity
    dt::Float64      # time step
    Hmin::Float64    # desingularisation depth
    limiter::Symbol  # :minmod, :vanalbada, ...
    bc::Symbol       # :reflective, :periodic (This is not yet implemented)
end

mutable struct State
    # --- conservative vars ---
    h::Array{Float64,2}   # depth h(i,j)
    hu::Array{Float64,2}  # x-momentum h*u
    hv::Array{Float64,2}  # y-momentum h*v
    q::Array{Float64,3}   # combined state array (3, nx, ny)

    # --- bathymetry ---
    Bc::Array{Float64,2}        # bottom at cell centres    (nx,   ny)
    Bfx::Array{Float64,2}       # bottom at x-faces         (nx+1, ny)
    Bfy::Array{Float64,2}       # bottom at y-faces         (nx,   ny+1)
    Bcorner::Array{Float64,2}   # bottom at cell corners    (nx+1, ny+1)

    # --- Coriolis ---
    f::Array{Float64,2}         # f(i,j) at cell centres    (nx,   ny)

    # --- work arrays (reuse every RHS/step to avoid allocations) ---
    F::Array{Float64,3}   # x-fluxes  F[c,i,j], size (3, nx+1, ny)
    G::Array{Float64,3}   # y-fluxes  G[c,i,j], size (3, nx,   ny+1)

    SB::Array{Float64,3}  # bathymetry source terms  (3, nx, ny)
    SC::Array{Float64,3}  # Coriolis source terms    (3, nx, ny)

    dq::Array{Float64,3}      # RHS q_t = (h_t, (hu)_t, (hv)_t)
    q_stage::Array{Float64,3} # temporary stage for RK2 (same layout)
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
    return Bc, Bfx, Bfy, Bcorner
end

function build_velocities(x, y, h, hu, hv, Hmin)
    nx, ny = size(h)
    u = zeros(nx, ny)
    v = zeros(nx, ny)
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    eps = max(dx^4, dy^4)
    @inbounds for i in 1:nx, j in 1:ny
        hij = h[i,j]
        if hij < Hmin
            h4 = hij^4
            denom = h4 + max(h4, eps)
            s = sqrt(denom)
            uij = (sqrt(2)*hij*hu[i,j]) / s
            vij = (sqrt(2)*hij*hv[i,j]) / s
            u[i,j] = uij
            v[i,j] = vij
            hu[i,j] = hij * uij
            hv[i,j] = hij * vij
        else
            u[i,j] = hu[i,j] / hij
            v[i,j] = hv[i,j] / hij
        end
    end
    return u, v
end


#Build Coriolis parameter f = f_hat + \beta y_k
function build_f(x, y, f_hat, beta)
    Nx, Ny = length(x), length(y)
    f = Array{Float64}(undef, Nx, Ny)
    for j in 1:Nx, k in 1:Ny
        f[j,k] = f_hat + beta * y[k]
    end
    return f
end

# Primitives U_y = f/g * u and V_x = f/g * v 
function build_UV_KL(h, u, v, f, Bc, dx, dy, g)
    Nx, Ny = size(u)
    Uface = zeros(Float64, Nx, Ny+1)
    @inbounds for i in 1:Nx, j in 1:Ny
        Uface[i,j+1] = Uface[i,j] + (f[i,j]/g) * u[i,j] * dy
    end
    Vface = zeros(Float64, Nx+1, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        Vface[i+1,j] = Vface[i,j] + (f[i,j]/g) * v[i,j] * dx
    end
    Uc = @. 0.5 * (Uface[:,1:end-1] + Uface[:,2:end])
    Vc = @. 0.5 * (Vface[1:end-1,:] + Vface[2:end,:])

    K = similar(h)
    L = similar(h)
    @inbounds for i in 1:Nx, j in 1:Ny
        K[i,j] = g*(h[i,j] + Bc[i,j] - Vc[i,j])
        L[i,j] = g*(h[i,j] + Bc[i,j] + Uc[i,j])
    end

    return Uface, Vface, Uc, Vc, K, L
end

#================================#
# Ghost cell handling
#================================#
function fill_ghosts_uvKL!(ug, vg, Kg, Lg; bc::Symbol)
    Nx2, Ny2 = size(ug)      # = (Nx+2, Ny+2)
    Nx = Nx2 - 2
    Ny = Ny2 - 2
    if bc === :reflective
        @inbounds begin
            # left/right boundaries: u normal, v tangential
            ug[1,  :] .= -ug[2,    :]
            ug[end,:] .= -ug[end-1,:]
            vg[1,  :] .=  vg[2,    :]
            vg[end,:] .=  vg[end-1,:]
            Kg[1,  :] .=  Kg[2,    :]
            Kg[end,:] .=  Kg[end-1,:]
            Lg[1,  :] .=  Lg[2,    :]
            Lg[end,:] .=  Lg[end-1,:]

            # bottom/top boundaries: v normal, u tangential
            ug[:, 1]   .=  ug[:, 2   ]
            ug[:, end] .=  ug[:, end-1]
            vg[:, 1]   .= -vg[:, 2   ]
            vg[:, end] .= -vg[:, end-1]
            Kg[:, 1]   .=  Kg[:, 2   ]
            Kg[:, end] .=  Kg[:, end-1]
            Lg[:, 1]   .=  Lg[:, 2   ]
            Lg[:, end] .=  Lg[:, end-1]
        end
    elseif bc === :periodic
        @inbounds begin
            # left/right periodic
            ug[1,    :] .= ug[Nx+1, :]   # Left ghost = last physical
            ug[Nx+2, :] .= ug[2,    :]   # Right ghost = first physical
            vg[1,    :] .= vg[Nx+1, :]
            vg[Nx+2, :] .= vg[2,    :]
            Kg[1,    :] .= Kg[Nx+1, :]
            Kg[Nx+2, :] .= Kg[2,    :]
            Lg[1,    :] .= Lg[Nx+1, :]
            Lg[Nx+2, :] .= Lg[2,    :]

            # CORRECTED: bottom/top periodic
            # Bottom ghost (index 1) = top interior (index Ny+1)
            # Top ghost (index Ny+2) = bottom interior (index 2)
            ug[:, 1]    .= ug[:, Ny+1]   # Bottom ghost = last physical
            ug[:, Ny+2] .= ug[:, 2]      # Top ghost = first physical
            vg[:, 1]    .= vg[:, Ny+1]
            vg[:, Ny+2] .= vg[:, 2]
            Kg[:, 1]    .= Kg[:, Ny+1]
            Kg[:, Ny+2] .= Kg[:, 2]
            Lg[:, 1]    .= Lg[:, Ny+1]
            Lg[:, Ny+2] .= Lg[:, 2]
        end

    elseif bc === :outflow
        # Simple zero-gradient (Neumann) outflow BC
        @inbounds begin
            # left/right: copy interior values
            ug[1,  :] .= ug[2,    :]
            ug[end,:] .= ug[end-1,:]
            vg[1,  :] .= vg[2,    :]
            vg[end,:] .= vg[end-1,:]
            Kg[1,  :] .= Kg[2,    :]
            Kg[end,:] .= Kg[end-1,:]
            Lg[1,  :] .= Lg[2,    :]
            Lg[end,:] .= Lg[end-1,:]

            # bottom/top: copy interior values
            ug[:, 1]   .= ug[:, 2   ]
            ug[:, end] .= ug[:, end-1]
            vg[:, 1]   .= vg[:, 2   ]
            vg[:, end] .= vg[:, end-1]
            Kg[:, 1]   .= Kg[:, 2   ]
            Kg[:, end] .= Kg[:, end-1]
            Lg[:, 1]   .= Lg[:, 2   ]
            Lg[:, end] .= Lg[:, end-1]
        end

    else
        error("Unknown bc = $bc. Use :reflective, :periodic or :outflow.")
    end
    return nothing
end



##########################
# 2D slopes for u,v,K,L  #
##########################
function slopes_p2D!(σx_u, σx_v, σx_K, σx_L,
    σy_u, σy_v, σy_K, σy_L,
    ug, vg, Kg, Lg;
    limiter::Symbol = :minmod)
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
function reconstruct_p(u, v, K, L; limiter::Symbol = :minmod, bc::Symbol = :reflective)
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
    # Fill ghosts according to boundary condition
    fill_ghosts_uvKL!(ug, vg, Kg, Lg; bc = bc)

    # Allocate slope arrays
    σx_u = zeros(Float64, Nx+2, Ny+2); σx_v = zeros(Float64, Nx+2, Ny+2); σx_K = zeros(Float64, Nx+2, Ny+2); σx_L = zeros(Float64, Nx+2, Ny+2)
    σy_u = zeros(Float64, Nx+2, Ny+2); σy_v = zeros(Float64, Nx+2, Ny+2); σy_K = zeros(Float64, Nx+2, Ny+2); σy_L = zeros(Float64, Nx+2, Ny+2)

    #Compute slopes
    slopes_p2D!(σx_u, σx_v, σx_K, σx_L, σy_u, σy_v, σy_K, σy_L, ug, vg, Kg, Lg; limiter=limiter)

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

function reconstruct_h(h, Uf, Vf, KE, KW, LN, LS, Bfx, Bfy, g)
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
function build_F(hE, hW, uE, uW, vE, vW, g, bc::Symbol)
    Nx, Ny = size(hE)
    @assert size(hW) == (Nx,Ny)
    @assert size(uE) == (Nx,Ny) == size(uW) == size(vE) == size(vW)
    F = zeros(Float64, 3, Nx+1, Ny)
    @inbounds for i in 1:Nx-1, j in 1:Ny
        # face between cells i and i+1 → index fi = i+1
        hL = hE[i,j]; uL = uE[i,j]; vL = vE[i,j]
        hR = hW[i+1,j]; uR = uW[i+1,j]; vR = vW[i+1,j]
        cL = sqrt(g*max(hL,0.0)); cR = sqrt(g*max(hR,0.0))
        ap = max(0.0, uL + cL, uR + cR); am = min(0.0, uL - cL, uR - cR)
        denom = ap - am
        fi = i + 1
        if denom <= 1e-14
            F[1,fi,j] = 0.0; F[2,fi,j] = 0.0; F[3,fi,j] = 0.0
            continue
        end
        qL = hL*uL; qR = hR*uR
        F[1,fi,j] = (ap*qL - am*qR)/denom + (ap*am/denom)*(hR - hL)
        FL2 = hL*uL*uL + 0.5*g*hL*hL
        FR2 = hR*uR*uR + 0.5*g*hR*hR
        F[2,fi,j] = (ap*FL2 - am*FR2)/denom + (ap*am/denom)*(qR - qL)
        if (uL + uR) > 0.0
            F[3,fi,j] = qL*vL
        else
            F[3,fi,j] = qR*vR
        end
    end

    # ---------- boundary faces: depend on bc ----------
    if bc === :reflective
        @inbounds for j in 1:Ny
            hR = hW[1,j]; uR = uW[1,j]; vR = vW[1,j]
            hL = hR;      uL = -uR;     vL =  vR

            cL = sqrt(g*max(hL,0.0)); cR = sqrt(g*max(hR,0.0))
            ap = max(0.0, uL + cL, uR + cR); am = min(0.0, uL - cL, uR - cR)
            denom = ap - am

            if denom <= 1e-14
                F[1,1,j] = 0.0; F[2,1,j] = 0.0; F[3,1,j] = 0.0
            else
                qL = hL*uL
                qR = hR*uR
                F[1,1,j] = (ap*qL - am*qR)/denom + (ap*am/denom)*(hR - hL)
                FL2 = hL*uL*uL + 0.5*g*hL*hL
                FR2 = hR*uR*uR + 0.5*g*hR*hR
                F[2,1,j] = (ap*FL2 - am*FR2)/denom + (ap*am/denom)*(qR - qL)
                if (uL + uR) > 0.0
                    F[3,1,j] = qL*vL
                else
                    F[3,1,j] = qR*vR
                end
            end
            # RIGHT boundary: ghost cell (i=Nx+1) mirrored from cell Nx
            hL = hE[Nx,j]; uL = uE[Nx,j]; vL = vE[Nx,j]
            hR = hL;       uR = -uL;      vR =  vL
            cL = sqrt(g*max(hL,0.0))
            cR = sqrt(g*max(hR,0.0))
            ap = max(0.0, uL + cL, uR + cR)
            am = min(0.0, uL - cL, uR - cR)
            denom = ap - am
            if denom <= 1e-14
                F[1,Nx+1,j] = 0.0; F[2,Nx+1,j] = 0.0; F[3,Nx+1,j] = 0.0
            else
                qL = hL*uL
                qR = hR*uR
                F[1,Nx+1,j] = (ap*qL - am*qR)/denom + (ap*am/denom)*(hR - hL)
                FL2 = hL*uL*uL + 0.5*g*hL*hL
                FR2 = hR*uR*uR + 0.5*g*hR*hR
                F[2,Nx+1,j] = (ap*FL2 - am*FR2)/denom + (ap*am/denom)*(qR - qL)
                if (uL + uR) > 0.0
                    F[3,Nx+1,j] = qL*vL
                else
                    F[3,Nx+1,j] = qR*vR
                end
            end
        end
    elseif bc === :outflow
        @inbounds for j in 1:Ny
            # zero-gradient: copy neighbouring interior face
            F[:,1,j]    .= F[:,2,j]
            F[:,Nx+1,j] .= F[:,Nx,j]
        end

    elseif bc === :periodic
        @inbounds for j in 1:Ny
            # periodic face between cell Nx and cell 1
            hL = hE[Nx,j]; uL = uE[Nx,j]; vL = vE[Nx,j]
            hR = hW[1,j];  uR = uW[1,j];  vR = vW[1,j]
            cL = sqrt(g*max(hL,0.0))
            cR = sqrt(g*max(hR,0.0))
            ap = max(0.0, uL + cL, uR + cR)
            am = min(0.0, uL - cL, uR - cR)
            denom = ap - am

            if denom <= 1e-14
                F[1,1,j]    = 0.0
                F[2,1,j]    = 0.0
                F[3,1,j]    = 0.0
                F[1,Nx+1,j] = 0.0
                F[2,Nx+1,j] = 0.0
                F[3,Nx+1,j] = 0.0
                continue
            end

            qL = hL*uL
            qR = hR*uR

            Fh = (ap*qL - am*qR)/denom + (ap*am/denom)*(hR - hL)
            FL2 = hL*uL*uL + 0.5*g*hL*hL
            FR2 = hR*uR*uR + 0.5*g*hR*hR
            Fhu = (ap*FL2 - am*FR2)/denom + (ap*am/denom)*(qR - qL)
            Fhv = (uL + uR) > 0.0 ? qL*vL : qR*vR

            # same flux enters at left and leaves at right
            F[1,1,j]    = Fh
            F[2,1,j]    = Fhu
            F[3,1,j]    = Fhv
            F[1,Nx+1,j] = Fh
            F[2,Nx+1,j] = Fhu
            F[3,Nx+1,j] = Fhv
        end
    else
        error("Unknown bc = $bc in build_F")
    end

    return F
end


# Build G fluxes in y-direction
function build_G(hN, hS, uN, uS, vN, vS, g, bc::Symbol)
    Nx, Ny = size(hN)
    @assert size(hS) == (Nx,Ny)
    @assert size(uN) == (Nx,Ny) == size(uS) == size(vN) == size(vS)

    G = zeros(Float64, 3, Nx, Ny+1)

    # ---------- interior faces ----------
    @inbounds for i in 1:Nx, j in 1:Ny-1
        hD = hN[i,j];   uD = uN[i,j];   vD = vN[i,j]
        hU = hS[i,j+1]; uU = uS[i,j+1]; vU = vS[i,j+1]
        cD = sqrt(g*max(hD,0.0))
        cU = sqrt(g*max(hU,0.0))
        bp = max(0.0, vD + cD, vU + cU)
        bm = min(0.0, vD - cD, vU - cU)
        denom = bp - bm

        fj = j + 1
        if denom <= 1e-14
            G[1,i,fj] = 0.0
            G[2,i,fj] = 0.0
            G[3,i,fj] = 0.0
            continue
        end
        qyD = hD*vD
        qyU = hU*vU
        G[1,i,fj] = (bp*qyD - bm*qyU)/denom + (bp*bm/denom)*(hU - hD)
        GL3 = hD*vD*vD + 0.5*g*hD*hD
        GR3 = hU*vU*vU + 0.5*g*hU*hU
        G[3,i,fj] = (bp*GL3 - bm*GR3)/denom + (bp*bm/denom)*(qyU - qyD)
        if (vD + vU) > 0.0
            G[2,i,fj] = hD*uD*vD
        else
            G[2,i,fj] = hU*uU*vU
        end
    end

    # ---------- boundary faces ----------
     if bc === :reflective
        @inbounds for i in 1:Nx
            hU = hS[i,1]; uU = uS[i,1]; vU = vS[i,1]
            hD = hU;      uD = uU;      vD = -vU
            cD = sqrt(g*max(hD,0.0))
            cU = sqrt(g*max(hU,0.0))
            bp = max(0.0, vD + cD, vU + cU)
            bm = min(0.0, vD - cD, vU - cU)
            denom = bp - bm
            if denom <= 1e-14
                G[1,i,1] = 0.0
                G[2,i,1] = 0.0
                G[3,i,1] = 0.0
            else
                qyD = hD*vD
                qyU = hU*vU
                G[1,i,1] = (bp*qyD - bm*qyU)/denom + (bp*bm/denom)*(hU - hD)
                GL3 = hD*vD*vD + 0.5*g*hD*hD
                GR3 = hU*vU*vU + 0.5*g*hU*hU
                G[3,i,1] = (bp*GL3 - bm*GR3)/denom + (bp*bm/denom)*(qyU - qyD)
                if (vD + vU) > 0.0
                    G[2,i,1] = hD*uD*vD
                else
                    G[2,i,1] = hU*uU*vU
                end
            end

            # TOP boundary: ghost above (j=Ny+1) mirrors cell j=Ny
            hD = hN[i,Ny]; uD = uN[i,Ny]; vD = vN[i,Ny]
            hU = hD;       uU = uD;       vU = -vD
            cD = sqrt(g*max(hD,0.0))
            cU = sqrt(g*max(hU,0.0))
            bp = max(0.0, vD + cD, vU + cU)
            bm = min(0.0, vD - cD, vU - cU)
            denom = bp - bm
            if denom <= 1e-14
                G[1,i,Ny+1] = 0.0
                G[2,i,Ny+1] = 0.0
                G[3,i,Ny+1] = 0.0
            else
                qyD = hD*vD
                qyU = hU*vU
                G[1,i,Ny+1] = (bp*qyD - bm*qyU)/denom + (bp*bm/denom)*(hU - hD)
                GL3 = hD*vD*vD + 0.5*g*hD*hD
                GR3 = hU*vU*vU + 0.5*g*hU*hU
                G[3,i,Ny+1] = (bp*GL3 - bm*GR3)/denom + (bp*bm/denom)*(qyU - qyD)
                if (vD + vU) > 0.0
                    G[2,i,Ny+1] = hD*uD*vD
                else
                    G[2,i,Ny+1] = hU*uU*vU
                end
            end
        end
    elseif bc === :outflow
        @inbounds for i in 1:Nx
            G[:,i,1]     .= G[:,i,2]
            G[:,i,Ny+1]  .= G[:,i,Ny]
        end
    elseif bc === :periodic
        @inbounds for i in 1:Nx
            # periodic face between cell Ny and cell 1
            hD = hN[i,Ny];   uD = uN[i,Ny];   vD = vN[i,Ny]
            hU = hS[i,1];    uU = uS[i,1];    vU = vS[i,1]

            cD = sqrt(g*max(hD,0.0))
            cU = sqrt(g*max(hU,0.0))
            bp = max(0.0, vD + cD, vU + cU)
            bm = min(0.0, vD - cD, vU - cU)
            denom = bp - bm
            if denom <= 1e-14
                G[1,i,1]     = 0.0
                G[2,i,1]     = 0.0
                G[3,i,1]     = 0.0
                G[1,i,Ny+1]  = 0.0
                G[2,i,Ny+1]  = 0.0
                G[3,i,Ny+1]  = 0.0
                continue
            end
            qyD = hD*vD
            qyU = hU*vU
            Gh = (bp*qyD - bm*qyU)/denom + (bp*bm/denom)*(hU - hD)
            GL3 = hD*vD*vD + 0.5*g*hD*hD
            GR3 = hU*vU*vU + 0.5*g*hU*hU
            Ghv = (bp*GL3 - bm*GR3)/denom + (bp*bm/denom)*(qyU - qyD)
            Ghu = (vD + vU) > 0.0 ? hD*uD*vD : hU*uU*vU
            G[1,i,1]     = Gh
            G[2,i,1]     = Ghu
            G[3,i,1]     = Ghv
            G[1,i,Ny+1]  = Gh
            G[2,i,Ny+1]  = Ghu
            G[3,i,Ny+1]  = Ghv
        end
    else
        error("Unknown bc = $bc in build_G")
    end
    return G
end


# Bathymetry source S^B
function build_S_B(h, Bfx, Bfy, g, dx, dy)
    Nx, Ny = size(h)
    @assert size(Bfx) == (Nx+1, Ny)
    @assert size(Bfy) == (Nx, Ny+1)
    SB = zeros(Float64, 3, Nx, Ny)
    @inbounds for i in 1:Nx, j in 1:Ny
        hij = h[i,j]
        dBdx = (Bfx[i+1,j] - Bfx[i,j]) / dx
        dBdy = (Bfy[i,j+1] - Bfy[i,j]) / dy
        SB[2,i,j] = -g * hij * dBdx
        SB[3,i,j] = -g * hij * dBdy
    end
    return SB
end


# Coriolis source S^C
function build_S_C(h, u, v, f)
    Nx, Ny = size(h)
    @assert size(u) == (Nx,Ny)
    @assert size(v) == (Nx,Ny)
    @assert size(f) == (Nx,Ny)
    SC = zeros(Float64, 3, Nx, Ny)
    @inbounds for i in 1:Nx, j in 1:Ny
        fij = f[i,j]
        hu  = h[i,j] * u[i,j]
        hv  = h[i,j] * v[i,j]
        SC[2,i,j] =  fij * hv
        SC[3,i,j] = -fij * hu
    end
    return SC
end


# Calculate RHS residual by central-upwind scheme
function residual!(st::State, p::Params)
    q  = st.q
    dq = st.dq
    _, Nx, Ny = size(q)
    h  = @view q[1, :, :]
    hu = @view q[2, :, :]
    hv = @view q[3, :, :]
    dh  = @view dq[1, :, :]
    dhu = @view dq[2, :, :]
    dhv = @view dq[3, :, :]

    # 1) velocities
    u, v = build_velocities(p.x, p.y, h, hu, hv, p.Hmin)

    # 2) UVKL
    Uface, Vface, Uc, Vc, K, L =
    build_UV_KL(h, u, v, st.f, st.Bc,
                p.dx, p.dy, p.g)
   

    # 3) reconstruct p = (u,v,K,L)
    uE,uW,uN,uS,
    vE,vW,vN,vS,
    KE,KW,KN,KS,
    LE,LW,LN,LS = reconstruct_p(u, v, K, L; limiter=p.limiter, bc = p.bc)

    # 4) reconstruct h
    hE,hW,hN,hS = reconstruct_h(h, Uface, Vface, KE, KW, LN, LS, st.Bfx, st.Bfy, p.g)

    # 5) Fluxes and sources
    st.F .= build_F(hE, hW, uE, uW, vE, vW, p.g, p.bc)        # (3,Nx+1,Ny)
    st.G .= build_G(hN, hS, uN, uS, vN, vS, p.g, p.bc)        # (3,Nx,Ny+1)
    st.SB .= build_S_B(h, st.Bfx, st.Bfy, p.g, p.dx, p.dy)       # (3,Nx,Ny)
    st.SC .= build_S_C(h, u, v, st.f)                        # (3,Nx,Ny)
    F  = st.F; G  = st.G; SB = st.SB; SC = st.SC

    # 6) Compute residuals
    @inbounds for i in 1:Nx, j in 1:Ny
        dF1 = (F[1,i+1,j] - F[1,i,j]) / p.dx
        dF2 = (F[2,i+1,j] - F[2,i,j]) / p.dx
        dF3 = (F[3,i+1,j] - F[3,i,j]) / p.dx

        dG1 = (G[1,i,j+1] - G[1,i,j]) / p.dy
        dG2 = (G[2,i,j+1] - G[2,i,j]) / p.dy
        dG3 = (G[3,i,j+1] - G[3,i,j]) / p.dy

        dh[i,j]  = -dF1 - dG1 + SB[1,i,j] + SC[1,i,j]
        dhu[i,j] = -dF2 - dG2 + SB[2,i,j] + SC[2,i,j]
        dhv[i,j] = -dF3 - dG3 + SB[3,i,j] + SC[3,i,j]
    end
    return nothing
end

# =========================
# Time integration: SSP-RK2 (Heun)
# =========================
function step_RK2!(st::State, p::Params)
    q   = st.q
    dq  = st.dq
    q1  = st.q_stage

    # Stage 1: q¹ = qⁿ + dt * R(qⁿ)
    residual!(st, p)          # fills st.dq based on current st.q
    @. q1 = q + p.dt * dq     # q1 = q + dt*dq

    # Temporarily swap q → q1 to compute R(q¹)
    q_orig = st.q
    st.q = q1
    residual!(st, p)          # now dq = R(q1)
    st.q = q_orig             # restore

    # Stage 2: qⁿ⁺¹ = ½ ( qⁿ + q¹ + dt * R(q¹) )
    @. q = 0.5 * (q + q1 + p.dt * dq)

    # Sync scalar fields from q
    @views begin
        st.h  .= q[1, :, :]
        st.hu .= q[2, :, :]
        st.hv .= q[3, :, :]
    end

    return nothing
end

# Added function to initialize state and parameters (constructs bathymetry and coriolis)
function init_state(x, y, bfun, f_hat, beta;
                    g, dt, Hmin, limiter=:minmod, bc=:reflective)

    nx, ny = length(x), length(y)
    dx, dy = x[2]-x[1], y[2]-y[1]
    # Must build the bathymetry first
    Bc, Bfx, Bfy, Bcorner = build_Btilde(x, y, bfun)
    # Build coriolis parameter
    f = build_f(x, y, f_hat, beta)  

    # Conservative vars
    h  = zeros(nx, ny)
    hu = zeros(nx, ny)
    hv = zeros(nx, ny)
    q = zeros(3, nx, ny)
    @views begin 
        q[1, :, :] .= h
        q[2, :, :] .= hu
        q[3, :, :] .= hv
    end

    # Work arrays
    F  = zeros(3, nx+1, ny)
    G  = zeros(3, nx, ny+1)
    SB = zeros(3, nx, ny)
    SC = zeros(3, nx, ny)
    dq = zeros(3, nx, ny)
    q_stage = similar(dq)

    p  = Params(x, y, nx, ny, dx, dy, g, dt, Hmin, limiter, bc)
    st = State(h, hu, hv, q, Bc, Bfx, Bfy, Bcorner, f, F, G, SB, SC, dq, q_stage)
    return st, p
end


end # module
