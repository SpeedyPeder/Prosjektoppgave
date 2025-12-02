module test_CDKLM

export Params, State, step_RK2!, init_state

# =========================
# Parameters & State
# =========================
struct Params
    x::Vector{Float64}  # x-grid points (cell centers)
    y::Vector{Float64}  # y-grid points (cell centers)  
    nx::Int          # number of physical cells in x
    ny::Int          # number of physical cells in y
    dx::Float64      # grid spacing in x
    dy::Float64      # grid spacing in y
    g::Float64       # gravity
    dt::Float64      # time step
    Hmin::Float64    # desingularisation depth
    limiter::Symbol  # :minmod, :vanalbada, ...
    bc::Symbol       # :reflective, :periodic, :outflow
end

mutable struct State
    # --- conservative vars (PHYSICAL CELLS ONLY) ---
    h::Array{Float64,2}   # depth h(i,j) for i=1:nx, j=1:ny
    hu::Array{Float64,2}  # x-momentum h*u
    hv::Array{Float64,2}  # y-momentum h*v
    q::Array{Float64,3}   # combined state array (3, nx, ny) - PHYSICAL ONLY

    # --- bathymetry (includes ghost cells where needed for reconstruction) ---
    Bc::Array{Float64,2}        # bottom at cell centres    (nx,   ny) - PHYSICAL
    Bfx::Array{Float64,2}       # bottom at x-faces         (nx+1, ny) - INCLUDES BOUNDARIES
    Bfy::Array{Float64,2}       # bottom at y-faces         (nx,   ny+1) - INCLUDES BOUNDARIES
    Bcorner::Array{Float64,2}   # bottom at cell corners    (nx+1, ny+1) - FOR VISUALIZATION

    # --- Coriolis ---
    f::Array{Float64,2}         # f(i,j) at cell centres    (nx,   ny) - PHYSICAL

    # --- work arrays (include boundaries for fluxes) ---
    F::Array{Float64,3}   # x-fluxes  F[c,i,j], size (3, nx+1, ny) - INCLUDES BOUNDARY FLUXES
    G::Array{Float64,3}   # y-fluxes  G[c,i,j], size (3, nx,   ny+1) - INCLUDES BOUNDARY FLUXES

    SB::Array{Float64,3}  # bathymetry source terms  (3, nx, ny) - PHYSICAL
    SC::Array{Float64,3}  # Coriolis source terms    (3, nx, ny) - PHYSICAL

    dq::Array{Float64,3}      # RHS q_t = (h_t, (hu)_t, (hv)_t) (3, nx, ny) - PHYSICAL
    q_stage::Array{Float64,3} # temporary stage for RK2 (same layout) - PHYSICAL
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
function build_Btilde(x, y, bfun)
    Nx, Ny = length(x), length(y)
    dx = x[2]-x[1]
    dy = y[2]-y[1]
    
    # Cell centers (physical domain)
    x_centers = x
    y_centers = y
    
    # Face centers (including boundaries)
    x_faces = range(x[1]-dx/2, x[end]+dx/2, length=Nx+1)
    y_faces = range(y[1]-dy/2, y[end]+dy/2, length=Ny+1)
    
    # Corners (including boundaries)
    x_corners = range(x[1]-dx/2, x[end]+dx/2, length=Nx+1)
    y_corners = range(y[1]-dy/2, y[end]+dy/2, length=Ny+1)
    
    # Build arrays
    Bcorner = [bfun(x_corners[j], y_corners[k]) for j in 1:Nx+1, k in 1:Ny+1]
    
    Bfx = Array{Float64}(undef, Nx+1, Ny)
    for j in 1:Nx+1, k in 1:Ny
        Bfx[j,k] = 0.5*(Bcorner[j,k+1] + Bcorner[j,k]) 
    end
    
    Bfy = Array{Float64}(undef, Nx, Ny+1)
    for j in 1:Nx, k in 1:Ny+1
        Bfy[j,k] = 0.5*(Bcorner[j+1,k] + Bcorner[j,k]) 
    end
    
    Bc = Array{Float64}(undef, Nx, Ny)
    for j in 1:Nx, k in 1:Ny
        Bc[j,k] = 0.25*(Bfx[j,k] + Bfx[j+1,k] + Bfy[j,k] + Bfy[j,k+1])
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

function build_f(x, y, f_hat, beta)
    Nx, Ny = length(x), length(y)
    f = Array{Float64}(undef, Nx, Ny)
    for j in 1:Nx, k in 1:Ny
        f[j,k] = f_hat + beta * y[k]
    end
    return f
end

function build_UV_KL(h, u, v, f, Bc, dx, dy, g, bc::Symbol)
    Nx, Ny = size(u)
    
    Uface = zeros(Float64, Nx, Ny+1)
    Vface = zeros(Float64, Nx+1, Ny)
    
    # Integrate Uface in y-direction
    @inbounds for i in 1:Nx
        # Start from bottom and integrate upward
        for j in 1:Ny
            Uface[i,j+1] = Uface[i,j] + (f[i,j]/g) * u[i,j] * dy
        end
    end
    
    # Integrate Vface in x-direction  
    @inbounds for j in 1:Ny
        # Start from left and integrate rightward
        for i in 1:Nx
            Vface[i+1,j] = Vface[i,j] + (f[i,j]/g) * v[i,j] * dx
        end
    end

    Uc = @. 0.5 * (Uface[:,1:end-1] + Uface[:,2:end])
    Vc = @. 0.5 * (Vface[1:end-1,:] + Vface[2:end,:])

    K = similar(h)
    L = similar(h)
    @inbounds for i in 1:Nx, j in 1:Ny
        K[i,j] = g*(max(h[i,j], 0.0) + Bc[i,j] - Vc[i,j])
        L[i,j] = g*(max(h[i,j], 0.0) + Bc[i,j] + Uc[i,j])
    end

    return Uface, Vface, Uc, Vc, K, L
end

#================================#
# Ghost cell handling - CORRECTED
#================================#
function fill_ghosts!(ug, vg, Kg, Lg; bc::Symbol)
    Nx2, Ny2 = size(ug)  # Size including ghosts: (Nx+2, Ny+2)
    Nx = Nx2 - 2
    Ny = Ny2 - 2
    
    if bc === :reflective
        @inbounds begin
            # left/right boundaries
            for j in 1:Ny2
                ug[1, j] = -ug[2, j]         # u: reflective (normal component)
                ug[Nx2, j] = -ug[Nx2-1, j]   # u: reflective
                vg[1, j] = vg[2, j]          # v: no change (tangential)
                vg[Nx2, j] = vg[Nx2-1, j]    # v: no change
                Kg[1, j] = Kg[2, j]          # K: zero gradient
                Kg[Nx2, j] = Kg[Nx2-1, j]    # K: zero gradient
                Lg[1, j] = Lg[2, j]          # L: zero gradient  
                Lg[Nx2, j] = Lg[Nx2-1, j]    # L: zero gradient
            end
            
            # bottom/top boundaries
            for i in 1:Nx2
                ug[i, 1] = ug[i, 2]          # u: no change (tangential)
                ug[i, Ny2] = ug[i, Ny2-1]    # u: no change
                vg[i, 1] = -vg[i, 2]         # v: reflective (normal component)
                vg[i, Ny2] = -vg[i, Ny2-1]   # v: reflective
                Kg[i, 1] = Kg[i, 2]          # K: zero gradient
                Kg[i, Ny2] = Kg[i, Ny2-1]    # K: zero gradient
                Lg[i, 1] = Lg[i, 2]          # L: zero gradient
                Lg[i, Ny2] = Lg[i, Ny2-1]    # L: zero gradient
            end
        end
        
    elseif bc === :periodic
        @inbounds begin
            # left/right periodic - CORRECTED
            for j in 1:Ny2
                ug[1, j] = ug[Nx+1, j]       # Left ghost = right interior
                ug[Nx2, j] = ug[2, j]        # Right ghost = left interior
                vg[1, j] = vg[Nx+1, j]
                vg[Nx2, j] = vg[2, j]
                Kg[1, j] = Kg[Nx+1, j]
                Kg[Nx2, j] = Kg[2, j]
                Lg[1, j] = Lg[Nx+1, j]
                Lg[Nx2, j] = Lg[2, j]
            end
            
            # bottom/top periodic - CORRECTED
            for i in 1:Nx2
                ug[i, 1] = ug[i, Ny+1]       # Bottom ghost = top interior
                ug[i, Ny2] = ug[i, 2]        # Top ghost = bottom interior
                vg[i, 1] = vg[i, Ny+1]
                vg[i, Ny2] = vg[i, 2]
                Kg[i, 1] = Kg[i, Ny+1]
                Kg[i, Ny2] = Kg[i, 2]
                Lg[i, 1] = Lg[i, Ny+1]
                Lg[i, Ny2] = Lg[i, 2]
            end
        end
        
    elseif bc === :outflow
        @inbounds begin
            # Zero-gradient for all boundaries
            for j in 1:Ny2
                ug[1, j] = ug[2, j]
                ug[Nx2, j] = ug[Nx2-1, j]
                vg[1, j] = vg[2, j]
                vg[Nx2, j] = vg[Nx2-1, j]
                Kg[1, j] = Kg[2, j]
                Kg[Nx2, j] = Kg[Nx2-1, j]
                Lg[1, j] = Lg[2, j]
                Lg[Nx2, j] = Lg[Nx2-1, j]
            end
            
            for i in 1:Nx2
                ug[i, 1] = ug[i, 2]
                ug[i, Ny2] = ug[i, Ny2-1]
                vg[i, 1] = vg[i, 2]
                vg[i, Ny2] = vg[i, Ny2-1]
                Kg[i, 1] = Kg[i, 2]
                Kg[i, Ny2] = Kg[i, Ny2-1]
                Lg[i, 1] = Lg[i, 2]
                Lg[i, Ny2] = Lg[i, Ny2-1]
            end
        end
        
    else
        error("Unknown bc = $bc. Use :reflective, :periodic or :outflow.")
    end
    return nothing
end

function slopes_p2D!(σx_u, σx_v, σx_K, σx_L,
                    σy_u, σy_v, σy_K, σy_L,
                    ug, vg, Kg, Lg;
                    limiter::Symbol = :minmod)
    Nx2, Ny2 = size(ug)
    Nx = Nx2 - 2
    Ny = Ny2 - 2

    @inbounds for i in 2:Nx+1, j in 2:Ny+1
        # x-direction slopes (physical cells only)
        σx_u[i,j] = slope_limited(ug[i-1,j], ug[i,j], ug[i+1,j], limiter)
        σx_v[i,j] = slope_limited(vg[i-1,j], vg[i,j], vg[i+1,j], limiter)
        σx_K[i,j] = slope_limited(Kg[i-1,j], Kg[i,j], Kg[i+1,j], limiter)
        σx_L[i,j] = slope_limited(Lg[i-1,j], Lg[i,j], Lg[i+1,j], limiter)

        # y-direction slopes (physical cells only)
        σy_u[i,j] = slope_limited(ug[i,j-1], ug[i,j], ug[i,j+1], limiter)
        σy_v[i,j] = slope_limited(vg[i,j-1], vg[i,j], vg[i,j+1], limiter)
        σy_K[i,j] = slope_limited(Kg[i,j-1], Kg[i,j], Kg[i,j+1], limiter)
        σy_L[i,j] = slope_limited(Lg[i,j-1], Lg[i,j], Lg[i,j+1], limiter)
    end

    return nothing
end

function reconstruct_p(u, v, K, L; limiter::Symbol = :minmod, bc::Symbol = :reflective)
    Nx, Ny = size(u)

    # Create ghosted arrays (Nx+2, Ny+2)
    ug = zeros(Float64, Nx+2, Ny+2)
    vg = zeros(Float64, Nx+2, Ny+2)
    Kg = zeros(Float64, Nx+2, Ny+2)
    Lg = zeros(Float64, Nx+2, Ny+2)
    
    # Fill interior (physical cells) - indices 2:Nx+1, 2:Ny+1
    @inbounds for i in 1:Nx, j in 1:Ny
        ug[i+1, j+1] = u[i,j]
        vg[i+1, j+1] = v[i,j]
        Kg[i+1, j+1] = K[i,j]
        Lg[i+1, j+1] = L[i,j]
    end
    
    # Fill ghost cells
    fill_ghosts!(ug, vg, Kg, Lg; bc=bc)

    # Allocate slope arrays
    σx_u = zeros(Float64, Nx+2, Ny+2); σx_v = zeros(Float64, Nx+2, Ny+2)
    σx_K = zeros(Float64, Nx+2, Ny+2); σx_L = zeros(Float64, Nx+2, Ny+2)
    σy_u = zeros(Float64, Nx+2, Ny+2); σy_v = zeros(Float64, Nx+2, Ny+2)
    σy_K = zeros(Float64, Nx+2, Ny+2); σy_L = zeros(Float64, Nx+2, Ny+2)

    # Compute slopes (for physical cells only)
    slopes_p2D!(σx_u, σx_v, σx_K, σx_L, σy_u, σy_v, σy_K, σy_L, ug, vg, Kg, Lg; limiter=limiter)

    # Allocate interface values (for physical cells only)
    uE = similar(u); uW = similar(u); uN = similar(u); uS = similar(u)
    vE = similar(v); vW = similar(v); vN = similar(v); vS = similar(v)
    KE = similar(K); KW = similar(K); KN = similar(K); KS = similar(K)
    LE = similar(L); LW = similar(L); LN = similar(L); LS = similar(L)

    # Reconstruct to interfaces (physical cells only)
    @inbounds for i in 1:Nx, j in 1:Ny
        ig = i + 1  # Index in ghosted array
        jg = j + 1

        # x-faces (i+1/2, j) and (i-1/2, j)
        uE[i,j] = ug[ig,jg] + 0.5*σx_u[ig,jg]  # right face of cell i
        uW[i,j] = ug[ig,jg] - 0.5*σx_u[ig,jg]  # left face of cell i
        vE[i,j] = vg[ig,jg] + 0.5*σx_v[ig,jg]
        vW[i,j] = vg[ig,jg] - 0.5*σx_v[ig,jg]
        KE[i,j] = Kg[ig,jg] + 0.5*σx_K[ig,jg]
        KW[i,j] = Kg[ig,jg] - 0.5*σx_K[ig,jg]
        LE[i,j] = Lg[ig,jg] + 0.5*σx_L[ig,jg]
        LW[i,j] = Lg[ig,jg] - 0.5*σx_L[ig,jg]

        # y-faces (i, j+1/2) and (i, j-1/2)
        uN[i,j] = ug[ig,jg] + 0.5*σy_u[ig,jg]  # top face of cell j
        uS[i,j] = ug[ig,jg] - 0.5*σy_u[ig,jg]  # bottom face of cell j
        vN[i,j] = vg[ig,jg] + 0.5*σy_v[ig,jg]
        vS[i,j] = vg[ig,jg] - 0.5*σy_v[ig,jg]
        KN[i,j] = Kg[ig,jg] + 0.5*σy_K[ig,jg]
        KS[i,j] = Kg[ig,jg] - 0.5*σy_K[ig,jg]
        LN[i,j] = Lg[ig,jg] + 0.5*σy_L[ig,jg]
        LS[i,j] = Lg[ig,jg] - 0.5*σy_L[ig,jg]
    end

    return uE,uW,uN,uS, vE,vW,vN,vS, KE,KW,KN,KS, LE,LW,LN,LS
end

function reconstruct_h(h, Uf, Vf, KE, KW, LN, LS, Bfx, Bfy, g)
    Nx, Ny = size(h)
    hE = similar(h); hW = similar(h); hN = similar(h); hS = similar(h)
    eps_h = 1e-12
    
    @inbounds for i in 1:Nx, j in 1:Ny
        # x-face reconstructions
        hE_val = KE[i,j]/g + Vf[i+1,j] - Bfx[i+1,j]  # right face of cell i
        hW_val = KW[i,j]/g + Vf[i,j]   - Bfx[i,j]    # left face of cell i
        
        # y-face reconstructions  
        hN_val = LN[i,j]/g - Uf[i,j+1] - Bfy[i,j+1]  # top face of cell j
        hS_val = LS[i,j]/g - Uf[i,j]   - Bfy[i,j]    # bottom face of cell j
        
        hE[i,j] = max(hE_val, eps_h)
        hW[i,j] = max(hW_val, eps_h)
        hN[i,j] = max(hN_val, eps_h)
        hS[i,j] = max(hS_val, eps_h)
    end
    return hE,hW,hN,hS
end

# =========================
# Fluxes (central-upwind) - CORRECTED PERIODIC BOUNDARIES
# =========================
function build_F(hE, hW, uE, uW, vE, vW, g, bc::Symbol)
    Nx, Ny = size(hE)
    F = zeros(Float64, 3, Nx+1, Ny)
    
    eps_h = 1e-12
    
    # Interior faces (i = 1 to Nx-1 for right faces of cells)
    @inbounds for i in 1:Nx-1, j in 1:Ny
        # Face between cell i and i+1 (face i+1)
        hL = max(hE[i,j], eps_h); uL = uE[i,j]; vL = vE[i,j]
        hR = max(hW[i+1,j], eps_h); uR = uW[i+1,j]; vR = vW[i+1,j]
        
        cL = sqrt(g * hL); cR = sqrt(g * hR)
        ap = max(0.0, uL + cL, uR + cR)
        am = min(0.0, uL - cL, uR - cR)
        denom = ap - am
        
        face_idx = i + 1
        if denom <= 1e-14
            F[1,face_idx,j] = 0.0; F[2,face_idx,j] = 0.0; F[3,face_idx,j] = 0.0
            continue
        end
        
        qL = hL * uL; qR = hR * uR
        F[1,face_idx,j] = (ap * qL - am * qR) / denom + (ap * am / denom) * (hR - hL)
        
        pL = 0.5 * g * hL^2; pR = 0.5 * g * hR^2
        FL2 = hL * uL^2 + pL; FR2 = hR * uR^2 + pR
        F[2,face_idx,j] = (ap * FL2 - am * FR2) / denom + (ap * am / denom) * (qR - qL)
        
        # Transverse momentum
        if uL + uR > 0.0
            F[3,face_idx,j] = qL * vL
        else
            F[3,face_idx,j] = qR * vR
        end
    end

    # Boundary faces
    if bc === :reflective
        @inbounds for j in 1:Ny
            # Left boundary (face 1)
            hR = max(hW[1,j], eps_h); uR = uW[1,j]; vR = vW[1,j]
            hL = hR; uL = -uR; vL = vR  # Reflective: normal velocity reverses
            
            cL = sqrt(g * hL); cR = sqrt(g * hR)
            ap = max(0.0, uL + cL, uR + cR); am = min(0.0, uL - cL, uR - cR)
            denom = ap - am
            
            if denom > 1e-14
                qL = hL * uL; qR = hR * uR
                F[1,1,j] = (ap * qL - am * qR) / denom + (ap * am / denom) * (hR - hL)
                pL = 0.5 * g * hL^2; pR = 0.5 * g * hR^2
                FL2 = hL * uL^2 + pL; FR2 = hR * uR^2 + pR
                F[2,1,j] = (ap * FL2 - am * FR2) / denom + (ap * am / denom) * (qR - qL)
                F[3,1,j] = (uL + uR > 0.0) ? qL * vL : qR * vR
            else
                F[1,1,j] = 0.0; F[2,1,j] = 0.0; F[3,1,j] = 0.0
            end
            
            # Right boundary (face Nx+1)
            hL = max(hE[Nx,j], eps_h); uL = uE[Nx,j]; vL = vE[Nx,j]
            hR = hL; uR = -uL; vR = vL  # Reflective: normal velocity reverses
            
            cL = sqrt(g * hL); cR = sqrt(g * hR)
            ap = max(0.0, uL + cL, uR + cR); am = min(0.0, uL - cL, uR - cR)
            denom = ap - am
            
            if denom > 1e-14
                qL = hL * uL; qR = hR * uR
                F[1,Nx+1,j] = (ap * qL - am * qR) / denom + (ap * am / denom) * (hR - hL)
                pL = 0.5 * g * hL^2; pR = 0.5 * g * hR^2
                FL2 = hL * uL^2 + pL; FR2 = hR * uR^2 + pR
                F[2,Nx+1,j] = (ap * FL2 - am * FR2) / denom + (ap * am / denom) * (qR - qL)
                F[3,Nx+1,j] = (uL + uR > 0.0) ? qL * vL : qR * vR
            else
                F[1,Nx+1,j] = 0.0; F[2,Nx+1,j] = 0.0; F[3,Nx+1,j] = 0.0
            end
        end
        
    elseif bc === :periodic
        @inbounds for j in 1:Ny
            # Left boundary (face 1) - periodic with right boundary
            # Use right state from last cell and left state from first cell
            hL = max(hE[Nx,j], eps_h); uL = uE[Nx,j]; vL = vE[Nx,j]  # Right face of last cell
            hR = max(hW[1,j], eps_h); uR = uW[1,j]; vR = vW[1,j]     # Left face of first cell
            
            cL = sqrt(g * hL); cR = sqrt(g * hR)
            ap = max(0.0, uL + cL, uR + cR); am = min(0.0, uL - cL, uR - cR)
            denom = ap - am
            
            if denom > 1e-14
                qL = hL * uL; qR = hR * uR
                flux_h = (ap * qL - am * qR) / denom + (ap * am / denom) * (hR - hL)
                pL = 0.5 * g * hL^2; pR = 0.5 * g * hR^2
                FL2 = hL * uL^2 + pL; FR2 = hR * uR^2 + pR
                flux_hu = (ap * FL2 - am * FR2) / denom + (ap * am / denom) * (qR - qL)
                flux_hv = (uL + uR > 0.0) ? qL * vL : qR * vR
                
                F[1,1,j] = flux_h
                F[2,1,j] = flux_hu  
                F[3,1,j] = flux_hv
            else
                F[1,1,j] = 0.0; F[2,1,j] = 0.0; F[3,1,j] = 0.0
            end
            
            # Right boundary (face Nx+1) - same as left boundary for periodic
            F[1,Nx+1,j] = F[1,1,j]
            F[2,Nx+1,j] = F[2,1,j]
            F[3,Nx+1,j] = F[3,1,j]
        end
        
    elseif bc === :outflow
        @inbounds for j in 1:Ny
            # Left boundary - zero gradient
            hL = max(hW[1,j], eps_h); uL = uW[1,j]; vL = vW[1,j]
            hR = hL; uR = uL; vR = vL
            
            cL = sqrt(g * hL); cR = sqrt(g * hR)
            ap = max(0.0, uL + cL, uR + cR); am = min(0.0, uL - cL, uR - cR)
            denom = ap - am
            
            if denom > 1e-14
                qL = hL * uL; qR = hR * uR
                F[1,1,j] = (ap * qL - am * qR) / denom + (ap * am / denom) * (hR - hL)
                pL = 0.5 * g * hL^2; pR = 0.5 * g * hR^2
                FL2 = hL * uL^2 + pL; FR2 = hR * uR^2 + pR
                F[2,1,j] = (ap * FL2 - am * FR2) / denom + (ap * am / denom) * (qR - qL)
                F[3,1,j] = (uL + uR > 0.0) ? qL * vL : qR * vR
            else
                F[1,1,j] = 0.0; F[2,1,j] = 0.0; F[3,1,j] = 0.0
            end
            
            # Right boundary - zero gradient
            hL = max(hE[Nx,j], eps_h); uL = uE[Nx,j]; vL = vE[Nx,j]
            hR = hL; uR = uL; vR = vL
            
            cL = sqrt(g * hL); cR = sqrt(g * hR)
            ap = max(0.0, uL + cL, uR + cR); am = min(0.0, uL - cL, uR - cR)
            denom = ap - am
            
            if denom > 1e-14
                qL = hL * uL; qR = hR * uR
                F[1,Nx+1,j] = (ap * qL - am * qR) / denom + (ap * am / denom) * (hR - hL)
                pL = 0.5 * g * hL^2; pR = 0.5 * g * hR^2
                FL2 = hL * uL^2 + pL; FR2 = hR * uR^2 + pR
                F[2,Nx+1,j] = (ap * FL2 - am * FR2) / denom + (ap * am / denom) * (qR - qL)
                F[3,Nx+1,j] = (uL + uR > 0.0) ? qL * vL : qR * vR
            else
                F[1,Nx+1,j] = 0.0; F[2,Nx+1,j] = 0.0; F[3,Nx+1,j] = 0.0
            end
        end
    end

    return F
end

function build_G(hN, hS, uN, uS, vN, vS, g, bc::Symbol)
    Nx, Ny = size(hN)
    G = zeros(Float64, 3, Nx, Ny+1)
    eps_h = 1e-12

    # Interior faces (j = 1 to Ny-1 for top faces of cells)
    @inbounds for i in 1:Nx, j in 1:Ny-1
        # Face between cell j and j+1 (face j+1)
        hD = max(hN[i,j], eps_h); uD = uN[i,j]; vD = vN[i,j]
        hU = max(hS[i,j+1], eps_h); uU = uS[i,j+1]; vU = vS[i,j+1]
        
        cD = sqrt(g * hD); cU = sqrt(g * hU)
        bp = max(0.0, vD + cD, vU + cU); bm = min(0.0, vD - cD, vU - cU)
        denom = bp - bm
        
        face_idx = j + 1
        if denom <= 1e-14
            G[1,i,face_idx] = 0.0; G[2,i,face_idx] = 0.0; G[3,i,face_idx] = 0.0
            continue
        end
        
        qyD = hD * vD; qyU = hU * vU
        G[1,i,face_idx] = (bp * qyD - bm * qyU) / denom + (bp * bm / denom) * (hU - hD)
        
        pD = 0.5 * g * hD^2; pU = 0.5 * g * hU^2
        GD3 = hD * vD^2 + pD; GU3 = hU * vU^2 + pU
        G[3,i,face_idx] = (bp * GD3 - bm * GU3) / denom + (bp * bm / denom) * (qyU - qyD)
        
        if vD + vU > 0.0
            G[2,i,face_idx] = hD * uD * vD
        else
            G[2,i,face_idx] = hU * uU * vU
        end
    end

    # Boundary faces
    if bc === :reflective
        @inbounds for i in 1:Nx
            # Bottom boundary (face 1)
            hU = max(hS[i,1], eps_h); uU = uS[i,1]; vU = vS[i,1]
            hD = hU; uD = uU; vD = -vU  # Reflective: normal velocity reverses
            
            cD = sqrt(g * hD); cU = sqrt(g * hU)
            bp = max(0.0, vD + cD, vU + cU); bm = min(0.0, vD - cD, vU - cU)
            denom = bp - bm
            
            if denom > 1e-14
                qyD = hD * vD; qyU = hU * vU
                G[1,i,1] = (bp * qyD - bm * qyU) / denom + (bp * bm / denom) * (hU - hD)
                pD = 0.5 * g * hD^2; pU = 0.5 * g * hU^2
                GD3 = hD * vD^2 + pD; GU3 = hU * vU^2 + pU
                G[3,i,1] = (bp * GD3 - bm * GU3) / denom + (bp * bm / denom) * (qyU - qyD)
                G[2,i,1] = (vD + vU > 0.0) ? hD * uD * vD : hU * uU * vU
            else
                G[1,i,1] = 0.0; G[2,i,1] = 0.0; G[3,i,1] = 0.0
            end
            
            # Top boundary (face Ny+1)
            hD = max(hN[i,Ny], eps_h); uD = uN[i,Ny]; vD = vN[i,Ny]
            hU = hD; uU = uD; vU = -vD  # Reflective: normal velocity reverses
            
            cD = sqrt(g * hD); cU = sqrt(g * hU)
            bp = max(0.0, vD + cD, vU + cU); bm = min(0.0, vD - cD, vU - cU)
            denom = bp - bm
            
            if denom > 1e-14
                qyD = hD * vD; qyU = hU * vU
                G[1,i,Ny+1] = (bp * qyD - bm * qyU) / denom + (bp * bm / denom) * (hU - hD)
                pD = 0.5 * g * hD^2; pU = 0.5 * g * hU^2
                GD3 = hD * vD^2 + pD; GU3 = hU * vU^2 + pU
                G[3,i,Ny+1] = (bp * GD3 - bm * GU3) / denom + (bp * bm / denom) * (qyU - qyD)
                G[2,i,Ny+1] = (vD + vU > 0.0) ? hD * uD * vD : hU * uU * vU
            else
                G[1,i,Ny+1] = 0.0; G[2,i,Ny+1] = 0.0; G[3,i,Ny+1] = 0.0
            end
        end
        
    elseif bc === :periodic
        @inbounds for i in 1:Nx
            # Bottom boundary (face 1) - periodic with top boundary
            hD = max(hN[i,Ny], eps_h); uD = uN[i,Ny]; vD = vN[i,Ny]  # Top face of last cell
            hU = max(hS[i,1], eps_h); uU = uS[i,1]; vU = vS[i,1]     # Bottom face of first cell
            
            cD = sqrt(g * hD); cU = sqrt(g * hU)
            bp = max(0.0, vD + cD, vU + cU); bm = min(0.0, vD - cD, vU - cU)
            denom = bp - bm
            
            if denom > 1e-14
                qyD = hD * vD; qyU = hU * vU
                flux_h = (bp * qyD - bm * qyU) / denom + (bp * bm / denom) * (hU - hD)
                pD = 0.5 * g * hD^2; pU = 0.5 * g * hU^2
                GD3 = hD * vD^2 + pD; GU3 = hU * vU^2 + pU
                flux_hv = (bp * GD3 - bm * GU3) / denom + (bp * bm / denom) * (qyU - qyD)
                flux_hu = (vD + vU > 0.0) ? hD * uD * vD : hU * uU * vU
                
                G[1,i,1] = flux_h
                G[2,i,1] = flux_hu
                G[3,i,1] = flux_hv
            else
                G[1,i,1] = 0.0; G[2,i,1] = 0.0; G[3,i,1] = 0.0
            end
            
            # Top boundary (face Ny+1) - same as bottom boundary for periodic
            G[1,i,Ny+1] = G[1,i,1]
            G[2,i,Ny+1] = G[2,i,1]
            G[3,i,Ny+1] = G[3,i,1]
        end
        
    elseif bc === :outflow
        @inbounds for i in 1:Nx
            # Bottom boundary - zero gradient
            hD = max(hS[i,1], eps_h); uD = uS[i,1]; vD = vS[i,1]
            hU = hD; uU = uD; vU = vD
            
            cD = sqrt(g * hD); cU = sqrt(g * hU)
            bp = max(0.0, vD + cD, vU + cU); bm = min(0.0, vD - cD, vU - cU)
            denom = bp - bm
            
            if denom > 1e-14
                qyD = hD * vD; qyU = hU * vU
                G[1,i,1] = (bp * qyD - bm * qyU) / denom + (bp * bm / denom) * (hU - hD)
                pD = 0.5 * g * hD^2; pU = 0.5 * g * hU^2
                GD3 = hD * vD^2 + pD; GU3 = hU * vU^2 + pU
                G[3,i,1] = (bp * GD3 - bm * GU3) / denom + (bp * bm / denom) * (qyU - qyD)
                G[2,i,1] = (vD + vU > 0.0) ? hD * uD * vD : hU * uU * vU
            else
                G[1,i,1] = 0.0; G[2,i,1] = 0.0; G[3,i,1] = 0.0
            end
            
            # Top boundary - zero gradient
            hD = max(hN[i,Ny], eps_h); uD = uN[i,Ny]; vD = vN[i,Ny]
            hU = hD; uU = uD; vU = vD
            
            cD = sqrt(g * hD); cU = sqrt(g * hU)
            bp = max(0.0, vD + cD, vU + cU); bm = min(0.0, vD - cD, vU - cU)
            denom = bp - bm
            
            if denom > 1e-14
                qyD = hD * vD; qyU = hU * vU
                G[1,i,Ny+1] = (bp * qyD - bm * qyU) / denom + (bp * bm / denom) * (hU - hD)
                pD = 0.5 * g * hD^2; pU = 0.5 * g * hU^2
                GD3 = hD * vD^2 + pD; GU3 = hU * vU^2 + pU
                G[3,i,Ny+1] = (bp * GD3 - bm * GU3) / denom + (bp * bm / denom) * (qyU - qyD)
                G[2,i,Ny+1] = (vD + vU > 0.0) ? hD * uD * vD : hU * uU * vU
            else
                G[1,i,Ny+1] = 0.0; G[2,i,Ny+1] = 0.0; G[3,i,Ny+1] = 0.0
            end
        end
    end

    return G
end

function build_S_B(h, Bfx, Bfy, g, dx, dy)
    Nx, Ny = size(h)
    SB = zeros(Float64, 3, Nx, Ny)
    @inbounds for i in 1:Nx, j in 1:Ny
        hij = max(h[i,j], 0.0)
        dBdx = (Bfx[i+1,j] - Bfx[i,j]) / dx
        dBdy = (Bfy[i,j+1] - Bfy[i,j]) / dy
        SB[2,i,j] = -g * hij * dBdx
        SB[3,i,j] = -g * hij * dBdy
    end
    return SB
end

function build_S_C(h, u, v, f)
    Nx, Ny = size(h)
    SC = zeros(Float64, 3, Nx, Ny)
    @inbounds for i in 1:Nx, j in 1:Ny
        fij = f[i,j]
        hu  = max(h[i,j], 0.0) * u[i,j]
        hv  = max(h[i,j], 0.0) * v[i,j]
        SC[2,i,j] =  fij * hv
        SC[3,i,j] = -fij * hu
    end
    return SC
end

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

    # 1) velocities (physical cells only)
    u, v = build_velocities(p.x, p.y, h, hu, hv, p.Hmin)

    # 2) UVKL (physical cells only)
    Uface, Vface, Uc, Vc, K, L = build_UV_KL(h, u, v, st.f, st.Bc, p.dx, p.dy, p.g, p.bc)

    # 3) reconstruct p = (u,v,K,L) (uses ghosts internally but returns physical cell interfaces)
    uE,uW,uN,uS, vE,vW,vN,vS, KE,KW,KN,KS, LE,LW,LN,LS = 
        reconstruct_p(u, v, K, L; limiter=p.limiter, bc=p.bc)

    # 4) reconstruct h (physical cell interfaces)
    hE,hW,hN,hS = reconstruct_h(h, Uface, Vface, KE, KW, LN, LS, st.Bfx, st.Bfy, p.g)

    # 5) Fluxes and sources (fluxes include boundaries, sources are physical only)
    st.F .= build_F(hE, hW, uE, uW, vE, vW, p.g, p.bc)
    st.G .= build_G(hN, hS, uN, uS, vN, vS, p.g, p.bc)
    st.SB .= build_S_B(h, st.Bfx, st.Bfy, p.g, p.dx, p.dy)
    st.SC .= build_S_C(h, u, v, st.f)
    F = st.F; G = st.G; SB = st.SB; SC = st.SC

    # 6) Compute residuals (physical cells only)
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

function step_RK2!(st::State, p::Params)
    q   = st.q
    dq  = st.dq
    q1  = st.q_stage

    # Stage 1: q¹ = qⁿ + dt * R(qⁿ)
    residual!(st, p)
    @. q1 = q + p.dt * dq

    # Temporarily swap q → q1 to compute R(q¹)
    q_orig = st.q
    st.q = q1
    residual!(st, p)
    st.q = q_orig

    # Stage 2: qⁿ⁺¹ = ½ ( qⁿ + q¹ + dt * R(q¹) )
    @. q = 0.5 * (q + q1 + p.dt * dq)

    # Ensure non-negative depth and sync
    @inbounds for i in eachindex(st.h)
        st.h[i] = max(st.q[1,i], 0.0)
    end

    @views begin
        st.h  .= q[1, :, :]
        st.hu .= q[2, :, :]
        st.hv .= q[3, :, :]
    end

    return nothing
end

function init_state(x, y, bfun, f_hat, beta;
                    g, dt, Hmin, limiter=:minmod, bc=:reflective)

    nx, ny = length(x), length(y)
    dx, dy = x[2]-x[1], y[2]-y[1]
    
    # Build bathymetry
    Bc, Bfx, Bfy, Bcorner = build_Btilde(x, y, bfun)
    
    # Build coriolis parameter
    f = build_f(x, y, f_hat, beta)  

    # Conservative vars (physical cells only)
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
    F  = zeros(3, nx+1, ny)   # x-fluxes (includes boundaries)
    G  = zeros(3, nx, ny+1)   # y-fluxes (includes boundaries)
    SB = zeros(3, nx, ny)     # sources (physical only)
    SC = zeros(3, nx, ny)     # sources (physical only)
    dq = zeros(3, nx, ny)     # RHS (physical only)
    q_stage = similar(dq)     # (physical only)

    p  = Params(x, y, nx, ny, dx, dy, g, dt, Hmin, limiter, bc)
    st = State(h, hu, hv, q, Bc, Bfx, Bfy, Bcorner, f, F, G, SB, SC, dq, q_stage)
    return st, p
end

end # module