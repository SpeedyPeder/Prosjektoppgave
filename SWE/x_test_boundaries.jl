module test_CDKLM

export Params, State, init_state, residual!, step_RK2!

# =========================
# Parameters & State
# =========================

struct Params
    x::Vector{Float64}  # x-grid points (cell centers)
    y::Vector{Float64}  # y-grid points (cell centers)
    nx::Int             # number of physical cells in x
    ny::Int             # number of physical cells in y
    dx::Float64         # grid spacing in x
    dy::Float64         # grid spacing in y
    g::Float64          # gravity
    dt::Float64         # time step
    Hmin::Float64       # desingularisation depth
    limiter::Symbol     # :minmod, :vanalbada, ...
    bc::Symbol          # :reflective, :periodic, :outflow
end

mutable struct State
    # --- conservative vars (cell centred, no ghosts) ---
    h::Array{Float64,2}    # depth h(i,j)
    hu::Array{Float64,2}   # x-momentum h*u
    hv::Array{Float64,2}   # y-momentum h*v
    q::Array{Float64,3}    # combined state array (3, nx, ny)

    # --- bathymetry on staggered locations ---
    Bc::Array{Float64,2}       # bottom at cell centres    (nx,   ny)
    Bfx::Array{Float64,2}      # bottom at x-faces         (nx+1, ny)
    Bfy::Array{Float64,2}      # bottom at y-faces         (nx,   ny+1)
    Bcorner::Array{Float64,2}  # bottom at cell corners    (nx+1, ny+1)

    # --- Coriolis parameter at cell centres ---
    f::Array{Float64,2}        # f(i,j) at cell centres    (nx,   ny)

    # --- work arrays (reused every RHS eval) ---
    F::Array{Float64,3}    # x-fluxes  F[c,i,j], size (3, nx+1, ny)
    G::Array{Float64,3}    # y-fluxes  G[c,i,j], size (3, nx,   ny+1)

    SB::Array{Float64,3}   # bathymetry source terms  (3, nx, ny)
    SC::Array{Float64,3}   # Coriolis source terms    (3, nx, ny)

    dq::Array{Float64,3}       # RHS q_t = (h_t, (hu)_t, (hv)_t)
    q_stage::Array{Float64,3}  # temporary stage for RK2 (same layout)
end

# θ in (2.9); can be tuned in [1,2]
const THETA_MINMOD = 1.5

# =========================
# Limiters
# =========================

@inline function _minmod2(a, b)
    (a*b <= 0.0) && return 0.0
    return copysign(min(abs(a), abs(b)), a)
end

@inline function _minmod3(a, b, c)
    if (a > 0.0 && b > 0.0 && c > 0.0)
        return min(a, min(b, c))
    elseif (a < 0.0 && b < 0.0 && c < 0.0)
        return max(a, max(b, c))
    else
        return 0.0
    end
end

@inline function _vanalbada(a, b; eps=1e-12)
    num = (a^2 + eps)*b + (b^2 + eps)*a
    den = a^2 + b^2 + 2eps
    return den == 0.0 ? 0.0 : num/den
end

# Generic 2-argument limiter used for non-minmod choices.
# Here dl, dr are *derivatives* or approximated slopes.
@inline function slope_limited(dl, dr, limiter::Symbol)
    if limiter === :minmod
        return _minmod2(dl, dr)
    elseif limiter === :vanalbada
        return _vanalbada(dl, dr)
    else
        error("Unknown limiter $(limiter). Use :minmod or :vanalbada.")
    end
end

# =========================
# Bathymetry: piecewise bilinear (2.1)–(2.3)
# =========================

function build_Btilde(x, y, bfun)
    nx, ny = length(x), length(y)
    dx = x[2] - x[1]
    dy = y[2] - y[1]

    # Corner grid (x_{j±1/2}, y_{k±1/2})
    xC = range(x[1] - dx/2, x[end] + dx/2, length=nx+1)
    yC = range(y[1] - dy/2, y[end] + dy/2, length=ny+1)

    Bcorner = [bfun(xC[i], yC[j]) for i in 1:nx+1, j in 1:ny+1]

    # Faces: B_{j+1/2,k}, B_{j,k+1/2}
    Bfx = Array{Float64}(undef, nx+1, ny)
    @inbounds for i in 1:nx+1, j in 1:ny
        Bfx[i,j] = 0.5*(Bcorner[i, j+1] + Bcorner[i, j])
    end

    Bfy = Array{Float64}(undef, nx, ny+1)
    @inbounds for i in 1:nx, j in 1:ny+1
        Bfy[i,j] = 0.5*(Bcorner[i+1, j] + Bcorner[i, j])
    end

    # Cell centres: B_{j,k} via (2.1)
    Bc = Array{Float64}(undef, nx, ny)
    @inbounds for i in 1:nx, j in 1:ny
        Bc[i,j] = 0.25*(Bfx[i,j] + Bfx[i+1,j] + Bfy[i,j] + Bfy[i,j+1])
    end

    return Bc, Bfx, Bfy, Bcorner
end

# =========================
# Velocities with desingularisation (2.4 + [27])
# =========================

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
            # Desingularized velocities (one of the strategies from [27])
            h4 = hij^4
            denom = h4 + max(h4, eps)
            s = sqrt(denom)
            uij = (sqrt(2.0)*hij*hu[i,j]) / s
            vij = (sqrt(2.0)*hij*hv[i,j]) / s
            u[i,j] = uij
            v[i,j] = vij
            hu[i,j] = hij*uij
            hv[i,j] = hij*vij
        else
            u[i,j] = hu[i,j] / hij
            v[i,j] = hv[i,j] / hij
        end
    end
    return u, v
end

# =========================
# Coriolis parameter f = f̂ + β y_k
# =========================

function build_f(x, y, f_hat, beta)
    nx, ny = length(x), length(y)
    f = Array{Float64}(undef, nx, ny)
    @inbounds for i in 1:nx, j in 1:ny
        f[i,j] = f_hat + beta*y[j]
    end
    return f
end

# =========================
# Coriolis primitives U, V and potentials K, L (2.5)–(2.7)
# =========================

function build_UV_KL(h, u, v, f, Bc, dx, dy, g)
    nx, ny = size(u)
    @assert size(v) == (nx, ny)
    @assert size(f) == (nx, ny)
    @assert size(Bc) == (nx, ny)

    # Faces: U_{j,k+1/2}, V_{j+1/2,k}
    Uface = zeros(Float64, nx, ny+1)   # (nx,   ny+1)
    Vface = zeros(Float64, nx+1, ny)   # (nx+1, ny)

    @inbounds for i in 1:nx, j in 1:ny
        Uface[i, j+1] = Uface[i, j] + (f[i,j]/g)*u[i,j]*dy
    end

    @inbounds for j in 1:ny, i in 1:nx
        Vface[i+1, j] = Vface[i, j] + (f[i,j]/g)*v[i,j]*dx
    end

    # Cell-centred U, V (2.6)
    Uc = @. 0.5*(Uface[:, 1:end-1] + Uface[:, 2:end])
    Vc = @. 0.5*(Vface[1:end-1, :] + Vface[2:end, :])

    # Potentials K, L (2.7)
    K = similar(h)
    L = similar(h)
    @inbounds for i in 1:nx, j in 1:ny
        K[i,j] = g*(h[i,j] + Bc[i,j] - Vc[i,j])
        L[i,j] = g*(h[i,j] + Bc[i,j] + Uc[i,j])
    end

    return Uface, Vface, Uc, Vc, K, L
end

# =========================
# Ghost cell handling for u,v,K,L
# =========================

function fill_ghosts_uvKL!(ug, vg, Kg, Lg; bc::Symbol)
    nx2, ny2 = size(ug)      # = (nx+2, ny+2)
    nx = nx2 - 2
    ny = ny2 - 2

    if bc === :reflective
        @inbounds begin
            # left/right: u normal, v tangential
            ug[1,  :] .= -ug[2,    :]
            ug[end,:] .= -ug[end-1,:]
            vg[1,  :] .=  vg[2,    :]
            vg[end,:] .=  vg[end-1,:]
            Kg[1,  :] .=  Kg[2,    :]
            Kg[end,:] .=  Kg[end-1,:]
            Lg[1,  :] .=  Lg[2,    :]
            Lg[end,:] .=  Lg[end-1,:]

            # bottom/top: v normal, u tangential
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
            ug[1,    :] .= ug[nx+1, :]
            ug[nx+2, :] .= ug[2,    :]
            vg[1,    :] .= vg[nx+1, :]
            vg[nx+2, :] .= vg[2,    :]
            Kg[1,    :] .= Kg[nx+1, :]
            Kg[nx+2, :] .= Kg[2,    :]
            Lg[1,    :] .= Lg[nx+1, :]
            Lg[nx+2, :] .= Lg[2,    :]

            # bottom/top periodic
            ug[:, 1]    .= ug[:, ny+1]
            ug[:, ny+2] .= ug[:, 2]
            vg[:, 1]    .= vg[:, ny+1]
            vg[:, ny+2] .= vg[:, 2]
            Kg[:, 1]    .= Kg[:, ny+1]
            Kg[:, ny+2] .= Kg[:, 2]
            Lg[:, 1]    .= Lg[:, ny+1]
            Lg[:, ny+2] .= Lg[:, 2]
        end

    elseif bc === :outflow
        # zero-gradient (Neumann) outflow
        @inbounds begin
            ug[1,  :] .= ug[2,    :]
            ug[end,:] .= ug[end-1,:]
            vg[1,  :] .= vg[2,    :]
            vg[end,:] .= vg[end-1,:]
            Kg[1,  :] .= Kg[2,    :]
            Kg[end,:] .= Kg[end-1,:]
            Lg[1,  :] .= Lg[2,    :]
            Lg[end,:] .= Lg[end-1,:]

            ug[:, 1]   .=  ug[:, 2   ]
            ug[:, end] .=  ug[:, end-1]
            vg[:, 1]   .=  vg[:, 2   ]
            vg[:, end] .=  vg[:, end-1]
            Kg[:, 1]   .=  Kg[:, 2   ]
            Kg[:, end] .=  Kg[:, end-1]
            Lg[:, 1]   .=  Lg[:, 2   ]
            Lg[:, end] .=  Lg[:, end-1]
        end
    else
        error("Unknown bc = $bc. Use :reflective, :periodic or :outflow.")
    end
    return nothing
end

# =========================
# Ghost cells for h,hu,hv
# =========================

function fill_ghosts_huv!(hg, hug, hvg; bc::Symbol)
    nx2, ny2 = size(hg)      # (nx+2, ny+2)
    nx = nx2 - 2
    ny = ny2 - 2

    if bc === :periodic
        @inbounds begin
            # left/right
            hg[1,  2:ny+1]     .= hg[nx+1, 2:ny+1]
            hg[nx+2, 2:ny+1]   .= hg[2,    2:ny+1]
            hug[1,  2:ny+1]    .= hug[nx+1, 2:ny+1]
            hug[nx+2, 2:ny+1]  .= hug[2,    2:ny+1]
            hvg[1,  2:ny+1]    .= hvg[nx+1, 2:ny+1]
            hvg[nx+2, 2:ny+1]  .= hvg[2,    2:ny+1]

            # bottom/top
            hg[2:nx+1, 1]      .= hg[2:nx+1, ny+1]
            hg[2:nx+1, ny+2]   .= hg[2:nx+1, 2]
            hug[2:nx+1, 1]     .= hug[2:nx+1, ny+1]
            hug[2:nx+1, ny+2]  .= hug[2:nx+1, 2]
            hvg[2:nx+1, 1]     .= hvg[2:nx+1, ny+1]
            hvg[2:nx+1, ny+2]  .= hvg[2:nx+1, 2]

            # corners
            hg[1, 1]        = hg[nx+1, ny+1]
            hg[1, ny+2]     = hg[nx+1, 2]
            hg[nx+2, 1]     = hg[2, ny+1]
            hg[nx+2, ny+2]  = hg[2, 2]

            hug[1, 1]       = hug[nx+1, ny+1]
            hug[1, ny+2]    = hug[nx+1, 2]
            hug[nx+2, 1]    = hug[2, ny+1]
            hug[nx+2, ny+2] = hug[2, 2]

            hvg[1, 1]       = hvg[nx+1, ny+1]
            hvg[1, ny+2]    = hvg[nx+1, 2]
            hvg[nx+2, 1]    = hvg[2, ny+1]
            hvg[nx+2, ny+2] = hvg[2, 2]
        end

    elseif bc === :reflective
        @inbounds begin
            # left/right: wall normal to x → flip hu
            hg[1,  2:ny+1]     .= hg[2,     2:ny+1]
            hg[nx+2, 2:ny+1]   .= hg[nx+1, 2:ny+1]
            hug[1,  2:ny+1]    .= -hug[2,     2:ny+1]
            hug[nx+2, 2:ny+1]  .= -hug[nx+1, 2:ny+1]
            hvg[1,  2:ny+1]    .=  hvg[2,     2:ny+1]
            hvg[nx+2, 2:ny+1]  .=  hvg[nx+1, 2:ny+1]

            # bottom/top: wall normal to y → flip hv
            hg[2:nx+1, 1]      .= hg[2:nx+1, 2]
            hg[2:nx+1, ny+2]   .= hg[2:nx+1, ny+1]
            hug[2:nx+1, 1]     .= hug[2:nx+1, 2]
            hug[2:nx+1, ny+2]  .= hug[2:nx+1, ny+1]
            hvg[2:nx+1, 1]     .= -hvg[2:nx+1, 2]
            hvg[2:nx+1, ny+2]  .= -hvg[2:nx+1, ny+1]

            # corners (copy nearest interior)
            hg[1, 1]        = hg[2, 2]
            hg[1, ny+2]     = hg[2, ny+1]
            hg[nx+2, 1]     = hg[nx+1, 2]
            hg[nx+2, ny+2]  = hg[nx+1, ny+1]

            hug[1, 1]       = hug[2, 2]
            hug[1, ny+2]    = hug[2, ny+1]
            hug[nx+2, 1]    = hug[nx+1, 2]
            hug[nx+2, ny+2] = hug[nx+1, ny+1]

            hvg[1, 1]       = hvg[2, 2]
            hvg[1, ny+2]    = hvg[2, ny+1]
            hvg[nx+2, 1]    = hvg[nx+1, 2]
            hvg[nx+2, ny+2] = hvg[nx+1, ny+1]
        end

    elseif bc === :outflow
        @inbounds begin
            # zero-gradient everywhere
            hg[1,  2:ny+1]     .= hg[2,     2:ny+1]
            hg[nx+2, 2:ny+1]   .= hg[nx+1, 2:ny+1]
            hug[1,  2:ny+1]    .= hug[2,     2:ny+1]
            hug[nx+2, 2:ny+1]  .= hug[nx+1, 2:ny+1]
            hvg[1,  2:ny+1]    .= hvg[2,     2:ny+1]
            hvg[nx+2, 2:ny+1]  .= hvg[nx+1, 2:ny+1]

            hg[2:nx+1, 1]      .= hg[2:nx+1, 2]
            hg[2:nx+1, ny+2]   .= hg[2:nx+1, ny+1]
            hug[2:nx+1, 1]     .= hug[2:nx+1, 2]
            hug[2:nx+1, ny+2]  .= hug[2:nx+1, ny+1]
            hvg[2:nx+1, 1]     .= hvg[2:nx+1, 2]
            hvg[2:nx+1, ny+2]  .= hvg[2:nx+1, ny+1]

            # corners
            hg[1, 1]        = hg[2, 2]
            hg[1, ny+2]     = hg[2, ny+1]
            hg[nx+2, 1]     = hg[nx+1, 2]
            hg[nx+2, ny+2]  = hg[nx+1, ny+1]

            hug[1, 1]       = hug[2, 2]
            hug[1, ny+2]    = hug[2, ny+1]
            hug[nx+2, 1]    = hug[nx+1, 2]
            hug[nx+2, ny+2] = hug[nx+1, ny+1]

            hvg[1, 1]       = hvg[2, 2]
            hvg[1, ny+2]    = hvg[2, ny+1]
            hvg[nx+2, 1]    = hvg[nx+1, 2]
            hvg[nx+2, ny+2] = hvg[nx+1, ny+1]
        end
    else
        error("Unknown bc = $bc in fill_ghosts_huv!")
    end

    return nothing
end

# =========================
# 2D slopes for u, v, K, L  (generalized minmod, (2.9))
# =========================

function slopes_p2D!(
    σx_u, σx_v, σx_K, σx_L,
    σy_u, σy_v, σy_K, σy_L,
    ug, vg, Kg, Lg;
    limiter::Symbol = :minmod,
    dx::Float64,
    dy::Float64
)
    nx2, ny2 = size(ug)
    @assert size(vg) == (nx2, ny2)
    @assert size(Kg) == (nx2, ny2)
    @assert size(Lg) == (nx2, ny2)

    nx = nx2 - 2
    ny = ny2 - 2

    if limiter === :minmod
        θ = THETA_MINMOD
        @inbounds for I in 2:nx+1, J in 2:ny+1
            # ----- x-direction slopes -----
            duR = (ug[I+1,J] - ug[I,  J]) / dx
            du0 = (ug[I+1,J] - ug[I-1,J]) / (2dx)
            duL = (ug[I,  J] - ug[I-1,J]) / dx
            σx_u[I,J] = _minmod3(θ*duR, du0, θ*duL)

            dvR = (vg[I+1,J] - vg[I,  J]) / dx
            dv0 = (vg[I+1,J] - vg[I-1,J]) / (2dx)
            dvL = (vg[I,  J] - vg[I-1,J]) / dx
            σx_v[I,J] = _minmod3(θ*dvR, dv0, θ*dvL)

            dKR = (Kg[I+1,J] - Kg[I,  J]) / dx
            dK0 = (Kg[I+1,J] - Kg[I-1,J]) / (2dx)
            dKL = (Kg[I,  J] - Kg[I-1,J]) / dx
            σx_K[I,J] = _minmod3(θ*dKR, dK0, θ*dKL)

            dLR = (Lg[I+1,J] - Lg[I,  J]) / dx
            dL0 = (Lg[I+1,J] - Lg[I-1,J]) / (2dx)
            dLL = (Lg[I,  J] - Lg[I-1,J]) / dx
            σx_L[I,J] = _minmod3(θ*dLR, dL0, θ*dLL)

            # ----- y-direction slopes -----
            duR = (ug[I,J+1] - ug[I,J  ]) / dy
            du0 = (ug[I,J+1] - ug[I,J-1]) / (2dy)
            duL = (ug[I,J  ] - ug[I,J-1]) / dy
            σy_u[I,J] = _minmod3(θ*duR, du0, θ*duL)

            dvR = (vg[I,J+1] - vg[I,J  ]) / dy
            dv0 = (vg[I,J+1] - vg[I,J-1]) / (2dy)
            dvL = (vg[I,J  ] - vg[I,J-1]) / dy
            σy_v[I,J] = _minmod3(θ*dvR, dv0, θ*dvL)

            dKR = (Kg[I,J+1] - Kg[I,J  ]) / dy
            dK0 = (Kg[I,J+1] - Kg[I,J-1]) / (2dy)
            dKL = (Kg[I,J  ] - Kg[I,J-1]) / dy
            σy_K[I,J] = _minmod3(θ*dKR, dK0, θ*dKL)

            dLR = (Lg[I,J+1] - Lg[I,J  ]) / dy
            dL0 = (Lg[I,J+1] - Lg[I,J-1]) / (2dy)
            dLL = (Lg[I,J  ] - Lg[I,J-1]) / dy
            σy_L[I,J] = _minmod3(θ*dLR, dL0, θ*dLL)
        end

    else
        # Generic two-argument limiter using left/right derivatives
        @inbounds for I in 2:nx+1, J in 2:ny+1
            # x-direction
            dl = (ug[I,  J] - ug[I-1,J]) / dx
            dr = (ug[I+1,J] - ug[I,  J]) / dx
            σx_u[I,J] = slope_limited(dl, dr, limiter)

            dl = (vg[I,  J] - vg[I-1,J]) / dx
            dr = (vg[I+1,J] - vg[I,  J]) / dx
            σx_v[I,J] = slope_limited(dl, dr, limiter)

            dl = (Kg[I,  J] - Kg[I-1,J]) / dx
            dr = (Kg[I+1,J] - Kg[I,  J]) / dx
            σx_K[I,J] = slope_limited(dl, dr, limiter)

            dl = (Lg[I,  J] - Lg[I-1,J]) / dx
            dr = (Lg[I+1,J] - Lg[I,  J]) / dx
            σx_L[I,J] = slope_limited(dl, dr, limiter)

            # y-direction
            dl = (ug[I,J  ] - ug[I,J-1]) / dy
            dr = (ug[I,J+1] - ug[I,J  ]) / dy
            σy_u[I,J] = slope_limited(dl, dr, limiter)

            dl = (vg[I,J  ] - vg[I,J-1]) / dy
            dr = (vg[I,J+1] - vg[I,J  ]) / dy
            σy_v[I,J] = slope_limited(dl, dr, limiter)

            dl = (Kg[I,J  ] - Kg[I,J-1]) / dy
            dr = (Kg[I,J+1] - Kg[I,J  ]) / dy
            σy_K[I,J] = slope_limited(dl, dr, limiter)

            dl = (Lg[I,J  ] - Lg[I,J-1]) / dy
            dr = (Lg[I,J+1] - Lg[I,J  ]) / dy
            σy_L[I,J] = slope_limited(dl, dr, limiter)
        end
    end

    return nothing
end

# =========================
# Reconstruction of p = (u,v,K,L) (2.8)–(2.10)
# =========================

function reconstruct_p(u, v, K, L, dx, dy;
                       limiter::Symbol = :minmod,
                       bc::Symbol = :reflective)

    nx, ny = size(u)
    @assert size(v) == (nx, ny)
    @assert size(K) == (nx, ny)
    @assert size(L) == (nx, ny)

    # ghosted arrays
    ug = zeros(Float64, nx+2, ny+2)
    vg = zeros(Float64, nx+2, ny+2)
    Kg = zeros(Float64, nx+2, ny+2)
    Lg = zeros(Float64, nx+2, ny+2)

    @inbounds begin
        ug[2:nx+1, 2:ny+1] .= u
        vg[2:nx+1, 2:ny+1] .= v
        Kg[2:nx+1, 2:ny+1] .= K
        Lg[2:nx+1, 2:ny+1] .= L
    end

    fill_ghosts_uvKL!(ug, vg, Kg, Lg; bc=bc)

    # slope arrays
    σx_u = zeros(Float64, nx+2, ny+2)
    σx_v = zeros(Float64, nx+2, ny+2)
    σx_K = zeros(Float64, nx+2, ny+2)
    σx_L = zeros(Float64, nx+2, ny+2)
    σy_u = zeros(Float64, nx+2, ny+2)
    σy_v = zeros(Float64, nx+2, ny+2)
    σy_K = zeros(Float64, nx+2, ny+2)
    σy_L = zeros(Float64, nx+2, ny+2)

    slopes_p2D!(σx_u, σx_v, σx_K, σx_L,
                σy_u, σy_v, σy_K, σy_L,
                ug, vg, Kg, Lg;
                limiter=limiter, dx=dx, dy=dy)

    # interface values: cell-centred i,j → use I=i+1, J=j+1
    uE = similar(u); uW = similar(u); uN = similar(u); uS = similar(u)
    vE = similar(v); vW = similar(v); vN = similar(v); vS = similar(v)
    KE = similar(K); KW = similar(K); KN = similar(K); KS = similar(K)
    LE = similar(L); LW = similar(L); LN = similar(L); LS = similar(L)

    @inbounds for i in 1:nx, j in 1:ny
        I = i + 1
        J = j + 1

        # x-faces: (x_{j±1/2}, y_k)
        uE[i,j] = ug[I,J] + 0.5*dx*σx_u[I,J]
        uW[i,j] = ug[I,J] - 0.5*dx*σx_u[I,J]
        vE[i,j] = vg[I,J] + 0.5*dx*σx_v[I,J]
        vW[i,j] = vg[I,J] - 0.5*dx*σx_v[I,J]
        KE[i,j] = Kg[I,J] + 0.5*dx*σx_K[I,J]
        KW[i,j] = Kg[I,J] - 0.5*dx*σx_K[I,J]
        LE[i,j] = Lg[I,J] + 0.5*dx*σx_L[I,J]
        LW[i,j] = Lg[I,J] - 0.5*dx*σx_L[I,J]

        # y-faces: (x_j, y_{k±1/2})
        uN[i,j] = ug[I,J] + 0.5*dy*σy_u[I,J]
        uS[i,j] = ug[I,J] - 0.5*dy*σy_u[I,J]
        vN[i,j] = vg[I,J] + 0.5*dy*σy_v[I,J]
        vS[i,j] = vg[I,J] - 0.5*dy*σy_v[I,J]
        KN[i,j] = Kg[I,J] + 0.5*dy*σy_K[I,J]
        KS[i,j] = Kg[I,J] - 0.5*dy*σy_K[I,J]
        LN[i,j] = Lg[I,J] + 0.5*dy*σy_L[I,J]
        LS[i,j] = Lg[I,J] - 0.5*dy*σy_L[I,J]
    end

    return uE,uW,uN,uS,
           vE,vW,vN,vS,
           KE,KW,KN,KS,
           LE,LW,LN,LS
end

# =========================
# Reconstruction of h at faces (2.11)–(2.14)
# =========================

function reconstruct_h(h, Uface, Vface, KE, KW, LN, LS, Bfx, Bfy, g)
    nx, ny = size(h)
    @assert size(Uface) == (nx, ny+1)
    @assert size(Vface) == (nx+1, ny)
    @assert size(KE)  == (nx, ny)
    @assert size(KW)  == (nx, ny)
    @assert size(LN)  == (nx, ny)
    @assert size(LS)  == (nx, ny)
    @assert size(Bfx) == (nx+1, ny)
    @assert size(Bfy) == (nx,   ny+1)

    hE = similar(h); hW = similar(h)
    hN = similar(h); hS = similar(h)

    @inbounds for i in 1:nx, j in 1:ny
        # (2.14)
        hE[i,j] = KE[i,j]/g + Vface[i+1,j] - Bfx[i+1,j]
        hW[i,j] = KW[i,j]/g + Vface[i,  j] - Bfx[i,  j]
        hN[i,j] = LN[i,j]/g - Uface[i,  j+1] - Bfy[i,  j+1]
        hS[i,j] = LS[i,j]/g - Uface[i,  j  ] - Bfy[i,  j]
    end

    return hE,hW,hN,hS
end

# =========================
# Fluxes (central-upwind, Section 3)
# =========================

function build_F(hE, hW, uE, uW, vE, vW, g, bc::Symbol)
    nx, ny = size(hE)
    @assert size(hW) == (nx, ny)
    @assert size(uE) == (nx, ny) == size(uW) == size(vE) == size(vW)

    F = zeros(Float64, 3, nx+1, ny)

    @inbounds for i in 1:nx-1, j in 1:ny
        # Left/right states at face i+1/2
        hL = hE[i,  j]; uL = uE[i,  j]; vL = vE[i,  j]
        hR = hW[i+1,j]; uR = uW[i+1,j]; vR = vW[i+1,j]

        cL = sqrt(g*max(hL, 0.0))
        cR = sqrt(g*max(hR, 0.0))
        ap = max(0.0, uL + cL, uR + cR)   # a^+ (3.10)
        am = min(0.0, uL - cL, uR - cR)   # a^- (3.10)
        denom = ap - am
        fi = i + 1

        if denom <= 1e-14
            F[1,fi,j] = 0.0
            F[2,fi,j] = 0.0
            F[3,fi,j] = 0.0
            continue
        end

        qL = hL*uL
        qR = hR*uR

        # F(1) (3.2)
        F[1,fi,j] = (ap*qL - am*qR)/denom + (ap*am/denom)*(hR - hL)

        # F(2) (3.3)
        FL2 = hL*uL*uL + 0.5*g*hL*hL
        FR2 = hR*uR*uR + 0.5*g*hR*hR
        F[2,fi,j] = (ap*FL2 - am*FR2)/denom + (ap*am/denom)*(qR - qL)

        # F(3): upwind in v (3.4)
        if (uL + uR) > 0.0
            F[3,fi,j] = qL*vL
        else
            F[3,fi,j] = qR*vR
        end
    end

    # Boundaries
    if bc === :reflective
        @inbounds for j in 1:ny
            # LEFT boundary (ghost to the left is mirror of cell 1)
            hR = hW[1,j]; uR = uW[1,j]; vR = vW[1,j]
            hL = hR;      uL = -uR;    vL = vR
            cL = sqrt(g*max(hL,0.0))
            cR = sqrt(g*max(hR,0.0))
            ap = max(0.0, uL + cL, uR + cR)
            am = min(0.0, uL - cL, uR - cR)
            denom = ap - am
            if denom <= 1e-14
                F[1,1,j] = 0.0; F[2,1,j] = 0.0; F[3,1,j] = 0.0
            else
                qL = hL*uL; qR = hR*uR
                F[1,1,j] = (ap*qL - am*qR)/denom + (ap*am/denom)*(hR - hL)
                FL2 = hL*uL*uL + 0.5*g*hL*hL
                FR2 = hR*uR*uR + 0.5*g*hR*hR
                F[2,1,j] = (ap*FL2 - am*FR2)/denom + (ap*am/denom)*(qR - qL)
                F[3,1,j] = (uL + uR) > 0 ? qL*vL : qR*vR
            end

            # RIGHT boundary
            hL = hE[nx,j]; uL = uE[nx,j]; vL = vE[nx,j]
            hR = hL;       uR = -uL;     vR = vL
            cL = sqrt(g*max(hL,0.0))
            cR = sqrt(g*max(hR,0.0))
            ap = max(0.0, uL + cL, uR + cR)
            am = min(0.0, uL - cL, uR - cR)
            denom = ap - am
            if denom <= 1e-14
                F[1,nx+1,j] = 0.0; F[2,nx+1,j] = 0.0; F[3,nx+1,j] = 0.0
            else
                qL = hL*uL; qR = hR*uR
                F[1,nx+1,j] = (ap*qL - am*qR)/denom + (ap*am/denom)*(hR - hL)
                FL2 = hL*uL*uL + 0.5*g*hL*hL
                FR2 = hR*uR*uR + 0.5*g*hR*hR
                F[2,nx+1,j] = (ap*FL2 - am*FR2)/denom + (ap*am/denom)*(qR - qL)
                F[3,nx+1,j] = (uL + uR) > 0 ? qL*vL : qR*vR
            end
        end

    elseif bc === :outflow
        @inbounds for j in 1:ny
            F[:,1,j]    .= F[:,2,j]
            F[:,nx+1,j] .= F[:,nx,j]
        end

    elseif bc === :periodic
        @inbounds for j in 1:ny
            # face between cell nx and cell 1
            hL = hE[nx,j]; uL = uE[nx,j]; vL = vE[nx,j]
            hR = hW[1, j]; uR = uW[1, j]; vR = vW[1, j]
            cL = sqrt(g*max(hL,0.0))
            cR = sqrt(g*max(hR,0.0))
            ap = max(0.0, uL + cL, uR + cR)
            am = min(0.0, uL - cL, uR - cR)
            denom = ap - am

            if denom <= 1e-14
                F[1,1,j]    = 0.0; F[2,1,j]    = 0.0; F[3,1,j]    = 0.0
                F[1,nx+1,j] = 0.0; F[2,nx+1,j] = 0.0; F[3,nx+1,j] = 0.0
                continue
            end

            qL = hL*uL; qR = hR*uR

            Fh = (ap*qL - am*qR)/denom + (ap*am/denom)*(hR - hL)
            FL2 = hL*uL*uL + 0.5*g*hL*hL
            FR2 = hR*uR*uR + 0.5*g*hR*hR
            Fhu = (ap*FL2 - am*FR2)/denom + (ap*am/denom)*(qR - qL)
            Fhv = (uL + uR) > 0 ? qL*vL : qR*vR

            F[1,1,j]    = Fh
            F[2,1,j]    = Fhu
            F[3,1,j]    = Fhv
            F[1,nx+1,j] = Fh
            F[2,nx+1,j] = Fhu
            F[3,nx+1,j] = Fhv
        end
    else
        error("Unknown bc = $bc in build_F")
    end

    return F
end

function build_G(hN, hS, uN, uS, vN, vS, g, bc::Symbol)
    nx, ny = size(hN)
    @assert size(hS) == (nx, ny)
    @assert size(uN) == (nx, ny) == size(uS) == size(vN) == size(vS)

    G = zeros(Float64, 3, nx, ny+1)

    # Interior faces
    @inbounds for i in 1:nx, j in 1:ny-1
        hD = hN[i,j];   uD = uN[i,j];   vD = vN[i,j]
        hU = hS[i,j+1]; uU = uS[i,j+1]; vU = vS[i,j+1]

        cD = sqrt(g*max(hD, 0.0))
        cU = sqrt(g*max(hU, 0.0))
        bp = max(0.0, vD + cD, vU + cU)   # b^+ (3.11)
        bm = min(0.0, vD - cD, vU - cU)   # b^- (3.11)
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

        # G(1) (3.5)
        G[1,i,fj] = (bp*qyD - bm*qyU)/denom + (bp*bm/denom)*(hU - hD)

        # G(3) (3.6)
        GL3 = hD*vD*vD + 0.5*g*hD*hD
        GR3 = hU*vU*vU + 0.5*g*hU*hU
        G[3,i,fj] = (bp*GL3 - bm*GR3)/denom + (bp*bm/denom)*(qyU - qyD)

        # G(2): upwind in u (3.7)
        if (vD + vU) > 0.0
            G[2,i,fj] = hD*uD*vD
        else
            G[2,i,fj] = hU*uU*vU
        end
    end

    # Boundaries
    if bc === :reflective
        @inbounds for i in 1:nx
            # BOTTOM
            hU = hS[i,1]; uU = uS[i,1]; vU = vS[i,1]
            hD = hU;      uD = uU;      vD = -vU
            cD = sqrt(g*max(hD,0.0))
            cU = sqrt(g*max(hU,0.0))
            bp = max(0.0, vD + cD, vU + cU)
            bm = min(0.0, vD - cD, vU - cU)
            denom = bp - bm
            if denom <= 1e-14
                G[1,i,1] = 0.0; G[2,i,1] = 0.0; G[3,i,1] = 0.0
            else
                qyD = hD*vD; qyU = hU*vU
                G[1,i,1] = (bp*qyD - bm*qyU)/denom + (bp*bm/denom)*(hU - hD)
                GL3 = hD*vD*vD + 0.5*g*hD*hD
                GR3 = hU*vU*vU + 0.5*g*hU*hU
                G[3,i,1] = (bp*GL3 - bm*GR3)/denom + (bp*bm/denom)*(qyU - qyD)
                G[2,i,1] = (vD + vU) > 0 ? hD*uD*vD : hU*uU*vU
            end

            # TOP
            hD = hN[i,ny]; uD = uN[i,ny]; vD = vN[i,ny]
            hU = hD;       uU = uD;       vU = -vD
            cD = sqrt(g*max(hD,0.0))
            cU = sqrt(g*max(hU,0.0))
            bp = max(0.0, vD + cD, vU + cU)
            bm = min(0.0, vD - cD, vU - cU)
            denom = bp - bm
            if denom <= 1e-14
                G[1,i,ny+1] = 0.0; G[2,i,ny+1] = 0.0; G[3,i,ny+1] = 0.0
            else
                qyD = hD*vD; qyU = hU*vU
                G[1,i,ny+1] = (bp*qyD - bm*qyU)/denom + (bp*bm/denom)*(hU - hD)
                GL3 = hD*vD*vD + 0.5*g*hD*hD
                GR3 = hU*vU*vU + 0.5*g*hU*hU
                G[3,i,ny+1] = (bp*GL3 - bm*GR3)/denom + (bp*bm/denom)*(qyU - qyD)
                G[2,i,ny+1] = (vD + vU) > 0 ? hD*uD*vD : hU*uU*vU
            end
        end

    elseif bc === :outflow
        @inbounds for i in 1:nx
            G[:,i,1]     .= G[:,i,2]
            G[:,i,ny+1]  .= G[:,i,ny]
        end

    elseif bc === :periodic
        @inbounds for i in 1:nx
            # face between cell ny and cell 1
            hD = hN[i,ny];   uD = uN[i,ny];   vD = vN[i,ny]
            hU = hS[i,1];    uU = uS[i,1];    vU = vS[i,1]
            cD = sqrt(g*max(hD,0.0))
            cU = sqrt(g*max(hU,0.0))
            bp = max(0.0, vD + cD, vU + cU)
            bm = min(0.0, vD - cD, vU - cU)
            denom = bp - bm
            if denom <= 1e-14
                G[1,i,1]    = 0.0; G[2,i,1]    = 0.0; G[3,i,1]    = 0.0
                G[1,i,ny+1] = 0.0; G[2,i,ny+1] = 0.0; G[3,i,ny+1] = 0.0
                continue
            end

            qyD = hD*vD; qyU = hU*vU
            Gh  = (bp*qyD - bm*qyU)/denom + (bp*bm/denom)*(hU - hD)
            GL3 = hD*vD*vD + 0.5*g*hD*hD
            GR3 = hU*vU*vU + 0.5*g*hU*hU
            Ghv = (bp*GL3 - bm*GR3)/denom + (bp*bm/denom)*(qyU - qyD)
            Ghu = (vD + vU) > 0 ? hD*uD*vD : hU*uU*vU

            G[1,i,1]    = Gh
            G[2,i,1]    = Ghu
            G[3,i,1]    = Ghv
            G[1,i,ny+1] = Gh
            G[2,i,ny+1] = Ghu
            G[3,i,ny+1] = Ghv
        end
    else
        error("Unknown bc = $bc in build_G")
    end

    return G
end

# =========================
# Source terms S^B and S^C (3.8)–(3.9)
# =========================

function build_S_B(h, Bfx, Bfy, g, dx, dy, bc::Symbol)
    nx, ny = size(h)
    @assert size(Bfx) == (nx+1, ny)
    @assert size(Bfy) == (nx,   ny+1)

    SB = zeros(Float64, 3, nx, ny)

    @inbounds for i in 1:nx, j in 1:ny
        hij = h[i,j]

        # Bj+1/2,k - Bj-1/2,k, etc. (3.8)
        if bc === :periodic
            ip = (i == nx) ? 1 : i+1
            jp = (j == ny) ? 1 : j+1
            dBdx = (Bfx[ip, j] - Bfx[i, j]) / dx
            dBdy = (Bfy[i, jp] - Bfy[i, j]) / dy
        else
            dBdx = (Bfx[i+1, j] - Bfx[i, j]) / dx
            dBdy = (Bfy[i, j+1] - Bfy[i, j]) / dy
        end

        SB[2,i,j] = -g*hij*dBdx
        SB[3,i,j] = -g*hij*dBdy
    end

    return SB
end

function build_S_C(h, u, v, f)
    nx, ny = size(h)
    @assert size(u) == (nx, ny)
    @assert size(v) == (nx, ny)
    @assert size(f) == (nx, ny)

    SC = zeros(Float64, 3, nx, ny)
    @inbounds for i in 1:nx, j in 1:ny
        fij = f[i,j]
        hu  = h[i,j]*u[i,j]
        hv  = h[i,j]*v[i,j]
        SC[2,i,j] =  fij*hv   # (3.9)
        SC[3,i,j] = -fij*hu
    end
    return SC
end

# =========================
# Residual: semi-discrete CU scheme (3.1)
# =========================

function residual!(st::State, p::Params)
    q  = st.q
    dq = st.dq

    _, nx, ny = size(q)
    @assert nx == p.nx && ny == p.ny

    # Views into conservative variables
    @views begin
        h  = q[1, :, :]
        hu = q[2, :, :]
        hv = q[3, :, :]

        dh  = dq[1, :, :]
        dhu = dq[2, :, :]
        dhv = dq[3, :, :]
    end

    # 1) build ghosted conservative vars
    hg  = zeros(Float64, nx+2, ny+2)
    hug = similar(hg)
    hvg = similar(hg)

    @inbounds begin
        hg[2:nx+1, 2:ny+1]  .= h
        hug[2:nx+1, 2:ny+1] .= hu
        hvg[2:nx+1, 2:ny+1] .= hv
    end

    fill_ghosts_huv!(hg, hug, hvg; bc=p.bc)

    @views begin
        hI  = hg[2:nx+1, 2:ny+1]
        huI = hug[2:nx+1, 2:ny+1]
        hvI = hvg[2:nx+1, 2:ny+1]
    end

    # 2) velocities and equilibrium variables (Section 2)
    u, v = build_velocities(p.x, p.y, hI, huI, hvI, p.Hmin)

    Uface, Vface, _, _, K, L =
        build_UV_KL(hI, u, v, st.f, st.Bc, p.dx, p.dy, p.g)

    # 3) reconstruct p = (u,v,K,L)
    uE,uW,uN,uS,
    vE,vW,vN,vS,
    KE,KW,KN,KS,
    LE,LW,LN,LS = reconstruct_p(u, v, K, L, p.dx, p.dy;
                                limiter=p.limiter, bc=p.bc)

    # 4) reconstruct h using well-balanced formulas (2.14)
    hE,hW,hN,hS = reconstruct_h(hI, Uface, Vface, KE, KW, LN, LS,
                                st.Bfx, st.Bfy, p.g)

    # 5) fluxes and sources
    st.F  .= build_F(hE, hW, uE, uW, vE, vW, p.g, p.bc)
    st.G  .= build_G(hN, hS, uN, uS, vN, vS, p.g, p.bc)
    st.SB .= build_S_B(hI, st.Bfx, st.Bfy, p.g, p.dx, p.dy, p.bc)
    st.SC .= build_S_C(hI, u, v, st.f)

    F  = st.F
    G  = st.G
    SB = st.SB
    SC = st.SC

    # 6) residual (3.1)
    @inbounds for i in 1:nx, j in 1:ny
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
# (Paper uses SSP-RK3, but any SSP RK is fine; cf. Remark 3.2.)
# =========================

function step_RK2!(st::State, p::Params)
    q   = st.q
    dq  = st.dq
    q1  = st.q_stage

    # Stage 1: q¹ = qⁿ + dt * R(qⁿ)
    residual!(st, p)
    @. q1 = q + p.dt*dq

    # Stage 2: qⁿ⁺¹ = ½( qⁿ + q¹ + dt * R(q¹) )
    q_orig = st.q
    st.q = q1
    residual!(st, p)     # dq = R(q1)
    st.q = q_orig

    @. q = 0.5*(q + q1 + p.dt*dq)

    # sync scalar fields
    @views begin
        st.h  .= q[1, :, :]
        st.hu .= q[2, :, :]
        st.hv .= q[3, :, :]
    end

    return nothing
end

# =========================
# Initialization helper
# =========================

function init_state(x, y, bfun, f_hat, beta;
                    g::Float64,
                    dt::Float64,
                    Hmin::Float64,
                    limiter::Symbol = :minmod,
                    bc::Symbol = :reflective)

    nx, ny = length(x), length(y)
    dx, dy = x[2]-x[1], y[2]-y[1]

    # Bathymetry (2.1)–(2.3)
    Bc, Bfx, Bfy, Bcorner = build_Btilde(x, y, bfun)

    # Coriolis parameter
    f = build_f(x, y, f_hat, beta)

    # Conservative vars (start from rest)
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
    G  = zeros(3, nx,   ny+1)
    SB = zeros(3, nx,   ny)
    SC = zeros(3, nx,   ny)
    dq = zeros(3, nx,   ny)
    q_stage = similar(dq)

    p  = Params(x, y, nx, ny, dx, dy, g, dt, Hmin, limiter, bc)
    st = State(h, hu, hv, q, Bc, Bfx, Bfy, Bcorner, f, F, G, SB, SC, dq, q_stage)

    return st, p
end

end # module
