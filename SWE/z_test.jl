module WBShallowWater

using LinearAlgebra

export Params, State, TopographyFaces, Equilibrium,
       EquilSlopes, FaceReconstruction,
       FluxX, FluxY, Sources,
       ICValue,
       build_topography_faces!, set_initial_conditions!,
       rk2_step!, rhs!,
       x_center, y_center

# ---------------- Parameters ----------------

struct Params
    Nx::Int           # number of physical cells in x
    Ny::Int           # number of physical cells in y
    ng::Int           # number of ghost cells
    dx::Float64
    dy::Float64
    x0::Float64       # physical domain left boundary
    y0::Float64       # physical domain bottom boundary
    g::Float64
    fhat::Float64     # \hat{f}
    beta::Float64     # beta in f(y) = fhat + beta*y
    theta::Float64    # θ in generalized minmod, [1,2]
end

nx_tot(par::Params) = par.Nx + 2*par.ng
ny_tot(par::Params) = par.Ny + 2*par.ng

# cell center coordinates from global indices (1..Nx+2ng, etc)
@inline function x_center(i::Int, par::Params)
    # physical cell i_phys = i - ng; center at x0 + (i_phys - 0.5)*dx
    return par.x0 + (i - par.ng - 0.5)*par.dx
end

@inline function y_center(j::Int, par::Params)
    return par.y0 + (j - par.ng - 0.5)*par.dy
end

# face coordinates between cells in x and y directions
@inline function x_face(iface::Int, par::Params)
    # face at x = x0 + (i_phys - 1)*dx; i_phys = iface - ng
    return par.x0 + (iface - par.ng - 1)*par.dx
end

@inline function y_face(jface::Int, par::Params)
    return par.y0 + (jface - par.ng - 1)*par.dy
end

# ---------------- State (cell averages) ----------------

mutable struct State
    h::Array{Float64,2}
    hu::Array{Float64,2}
    hv::Array{Float64,2}
    B::Array{Float64,2}      # bottom at cell centers
end

function State(par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    State(zeros(nx,ny), zeros(nx,ny), zeros(nx,ny), zeros(nx,ny))
end

# ---------------- Equilibrium variables ----------------

mutable struct Equilibrium
    u::Array{Float64,2}
    v::Array{Float64,2}
    U::Array{Float64,2}          # cell-centered U
    V::Array{Float64,2}          # cell-centered V
    Uface::Array{Float64,2}      # size (nx, ny+1) ~ U_{j,k+1/2}
    Vface::Array{Float64,2}      # size (nx+1, ny) ~ V_{j+1/2,k}
    K::Array{Float64,2}
    L::Array{Float64,2}
end

function Equilibrium(par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    Equilibrium(
        zeros(nx,ny), zeros(nx,ny),
        zeros(nx,ny), zeros(nx,ny),
        zeros(nx,ny+1), zeros(nx+1,ny),
        zeros(nx,ny), zeros(nx,ny)
    )
end

# Slopes of u,v,K,L
mutable struct EquilSlopes
    ux::Array{Float64,2}; uy::Array{Float64,2}
    vx::Array{Float64,2}; vy::Array{Float64,2}
    Kx::Array{Float64,2}; Ky::Array{Float64,2}
    Lx::Array{Float64,2}; Ly::Array{Float64,2}
end

function EquilSlopes(par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    EquilSlopes(
        zeros(nx,ny), zeros(nx,ny),
        zeros(nx,ny), zeros(nx,ny),
        zeros(nx,ny), zeros(nx,ny),
        zeros(nx,ny), zeros(nx,ny)
    )
end

# ---------------- Topography on faces ----------------

mutable struct TopographyFaces
    Bc::Array{Float64,2}   # cell centers (same as State.B)
    Bx::Array{Float64,2}   # size (nx+1, ny), B at x-faces j+1/2,k
    By::Array{Float64,2}   # size (nx, ny+1), B at y-faces j,k+1/2
end

function TopographyFaces(par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    TopographyFaces(
        zeros(nx,ny),
        zeros(nx+1,ny),
        zeros(nx,ny+1)
    )
end

# ---------------- Face reconstructions ----------------

mutable struct FaceReconstruction
    # u,v,K,L at faces from *each cell* (E,W,N,S)
    uE::Array{Float64,2}; uW::Array{Float64,2}; uN::Array{Float64,2}; uS::Array{Float64,2}
    vE::Array{Float64,2}; vW::Array{Float64,2}; vN::Array{Float64,2}; vS::Array{Float64,2}
    KE::Array{Float64,2}; KW::Array{Float64,2}; KN::Array{Float64,2}; KS::Array{Float64,2}
    LE::Array{Float64,2}; LW::Array{Float64,2}; LN::Array{Float64,2}; LS::Array{Float64,2}
    # reconstructed h at faces
    hE::Array{Float64,2}; hW::Array{Float64,2}; hN::Array{Float64,2}; hS::Array{Float64,2}
end

function FaceReconstruction(par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    FaceReconstruction(
        zeros(nx,ny), zeros(nx,ny), zeros(nx,ny), zeros(nx,ny),
        zeros(nx,ny), zeros(nx,ny), zeros(nx,ny), zeros(nx,ny),
        zeros(nx,ny), zeros(nx,ny), zeros(nx,ny), zeros(nx,ny),
        zeros(nx,ny), zeros(nx,ny), zeros(nx,ny), zeros(nx,ny),
        zeros(nx,ny), zeros(nx,ny), zeros(nx,ny), zeros(nx,ny)
    )
end

function fill_periodic!(A::AbstractArray{T,2}, par::Params) where {T}
    nx, ny, ng = nx_tot(par), ny_tot(par), par.ng

    # x-direction ghosts
    for j in 1:ny
        # left ghosts
        for g in 1:ng
            iL = g
            i_src = ng + par.Nx - (g - 1)   # right interior
            A[iL,j] = A[i_src,j]
        end
        # right ghosts
        for g in 1:ng
            iR = ng + par.Nx + g
            i_src = ng + g                  # left interior
            A[iR,j] = A[i_src,j]
        end
    end

    # y-direction ghosts
    for i in 1:nx
        # bottom ghosts
        for g in 1:ng
            jB = g
            j_src = ng + par.Ny - (g - 1)
            A[i,jB] = A[i,j_src]
        end
        # top ghosts
        for g in 1:ng
            jT = ng + par.Ny + g
            j_src = ng + g
            A[i,jT] = A[i,j_src]
        end
    end

    return A
end

function fill_periodic!(S::State, par::Params)
    fill_periodic!(S.h , par)
    fill_periodic!(S.hu, par)
    fill_periodic!(S.hv, par)
    # B is stationary; if you constructed it from a function on all cells,
    # you may not need to fill, but it's harmless:
    fill_periodic!(S.B , par)
end

struct ICValue
    h::Float64
    u::Float64
    v::Float64
    B::Float64
end

# ic(x,y) should return ICValue
function set_initial_conditions!(S::State, par::Params, ic::Function)
    nx, ny = nx_tot(par), ny_tot(par)
    for j in 1:ny
        y = y_center(j, par)
        for i in 1:nx
            x = x_center(i, par)
            val = ic(x,y)
            S.h[i,j]  = val.h
            S.hu[i,j] = val.h * val.u
            S.hv[i,j] = val.h * val.v
            S.B[i,j]  = val.B
        end
    end
    return S
end

# Bfun(x,y) -> Float64 gives the continuous bottom B(x,y)
function build_topography_faces!(TF::TopographyFaces, par::Params, Bfun::Function)
    nx, ny = nx_tot(par), ny_tot(par)

    # 1) First build face values from the continuous B(x,y)

    # x-faces: i_face = 1..nx+1, center in y
    for j in 1:ny
        y = y_center(j, par)
        for iface in 1:(nx+1)
            x = x_face(iface, par)
            TF.Bx[iface,j] = Bfun(x,y)
        end
    end

    # y-faces: j_face = 1..ny+1, center in x
    for jface in 1:(ny+1)
        y = y_face(jface, par)
        for i in 1:nx
            x = x_center(i, par)
            TF.By[i,jface] = Bfun(x,y)
        end
    end

    # 2) Now reconstruct cell centers as the average of the four surrounding faces
    #    Bj,k = 1/4 ( B_{j+1/2,k} + B_{j-1/2,k} + B_{j,k+1/2} + B_{j,k-1/2} )  (2.1)

    for j in 1:ny
        for i in 1:nx
            # faces around cell (i,j):
            # left  : Bx[i,   j]
            # right : Bx[i+1, j]
            # bottom: By[i,   j]
            # top   : By[i,   j+1]
            TF.Bc[i,j] = 0.25 * (
                TF.Bx[i,  j]   + TF.Bx[i+1,j] +
                TF.By[i,  j]   + TF.By[i,  j+1]
            )
        end
    end

    return TF
end

@inline function coriolis_at_row(j::Int, par::Params)
    y = y_center(j, par)
    return par.fhat + par.beta * y
end

function compute_equilibrium!(E::Equilibrium, S::State, par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    g = par.g
    h_eps = 1e-10

    # 1) velocities u,v (2.4)
    for j in 1:ny, i in 1:nx
        h = S.h[i,j]
        if h < h_eps
            E.u[i,j] = 0.0
            E.v[i,j] = 0.0
        else
            E.u[i,j] = S.hu[i,j] / h
            E.v[i,j] = S.hv[i,j] / h
        end
    end

    # 2) primitives Uface, Vface via recursive formulae (2.5)
    # Uface[i, kface] ~ U_{i, k+1/2}, kface=1..ny+1, k=1..ny
    # Vface[iface, j] ~ V_{i+1/2, j}, iface=1..nx+1, i=1..nx

    # U in y-direction
    for i in 1:nx
        E.Uface[i,1] = 0.0    # "kL-1/2" at bottom
        for j in 1:ny
            fk = coriolis_at_row(j, par)
            E.Uface[i,j+1] = E.Uface[i,j] + fk/g * E.u[i,j] * par.dy
        end
    end

    # V in x-direction
    for j in 1:ny
        E.Vface[1,j] = 0.0    # "jL-1/2" at left
        fk = coriolis_at_row(j, par)
        for i in 1:nx
            E.Vface[i+1,j] = E.Vface[i,j] + fk/g * E.v[i,j] * par.dx
        end
    end

    # 3) cell-centered U,V from faces (2.6)
    for j in 1:ny, i in 1:nx
        E.U[i,j] = 0.5 * (E.Uface[i,j] + E.Uface[i,j+1])
        E.V[i,j] = 0.5 * (E.Vface[i,j] + E.Vface[i+1,j])
    end

    # 4) K,L at centers (2.7)
    for j in 1:ny, i in 1:nx
        h = S.h[i,j]
        B = S.B[i,j]
        E.K[i,j] = g * (h + B - E.V[i,j])
        E.L[i,j] = g * (h + B + E.U[i,j])
    end

    return E
end

@inline function minmod3(a, b, c)
    if a > 0 && b > 0 && c > 0
        return min(a, min(b,c))
    elseif a < 0 && b < 0 && c < 0
        return max(a, max(b,c))
    else
        return 0.0
    end
end

function compute_equilibrium_slopes!(sl::EquilSlopes, E::Equilibrium, par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    θ = par.theta
    dx, dy = par.dx, par.dy

    # initialize to zero (ghost cells will essentially be piecewise constant)
    fill!(sl.ux, 0.0); fill!(sl.vx, 0.0); fill!(sl.Kx, 0.0); fill!(sl.Lx, 0.0)
    fill!(sl.uy, 0.0); fill!(sl.vy, 0.0); fill!(sl.Ky, 0.0); fill!(sl.Ly, 0.0)

    # x-slopes: i = 2..nx-1
    for j in 1:ny
        for i in 2:(nx-1)
            # u
            s1 = θ*(E.u[i+1,j] - E.u[i,j])   / dx
            s2 =    (E.u[i+1,j] - E.u[i-1,j])/(2*dx)
            s3 = θ*(E.u[i,j]   - E.u[i-1,j]) / dx
            sl.ux[i,j] = minmod3(s1,s2,s3)

            # v
            s1 = θ*(E.v[i+1,j] - E.v[i,j])   / dx
            s2 =    (E.v[i+1,j] - E.v[i-1,j])/(2*dx)
            s3 = θ*(E.v[i,j]   - E.v[i-1,j]) / dx
            sl.vx[i,j] = minmod3(s1,s2,s3)

            # K
            s1 = θ*(E.K[i+1,j] - E.K[i,j])   / dx
            s2 =    (E.K[i+1,j] - E.K[i-1,j])/(2*dx)
            s3 = θ*(E.K[i,j]   - E.K[i-1,j]) / dx
            sl.Kx[i,j] = minmod3(s1,s2,s3)

            # L
            s1 = θ*(E.L[i+1,j] - E.L[i,j])   / dx
            s2 =    (E.L[i+1,j] - E.L[i-1,j])/(2*dx)
            s3 = θ*(E.L[i,j]   - E.L[i-1,j]) / dx
            sl.Lx[i,j] = minmod3(s1,s2,s3)
        end
    end

    # y-slopes: j = 2..ny-1
    for j in 2:(ny-1)
        for i in 1:nx
            # u
            s1 = θ*(E.u[i,j+1] - E.u[i,j])   / dy
            s2 =    (E.u[i,j+1] - E.u[i,j-1])/(2*dy)
            s3 = θ*(E.u[i,j]   - E.u[i,j-1]) / dy
            sl.uy[i,j] = minmod3(s1,s2,s3)

            # v
            s1 = θ*(E.v[i,j+1] - E.v[i,j])   / dy
            s2 =    (E.v[i,j+1] - E.v[i,j-1])/(2*dy)
            s3 = θ*(E.v[i,j]   - E.v[i,j-1]) / dy
            sl.vy[i,j] = minmod3(s1,s2,s3)

            # K
            s1 = θ*(E.K[i,j+1] - E.K[i,j])   / dy
            s2 =    (E.K[i,j+1] - E.K[i,j-1])/(2*dy)
            s3 = θ*(E.K[i,j]   - E.K[i,j-1]) / dy
            sl.Ky[i,j] = minmod3(s1,s2,s3)

            # L
            s1 = θ*(E.L[i,j+1] - E.L[i,j])   / dy
            s2 =    (E.L[i,j+1] - E.L[i,j-1])/(2*dy)
            s3 = θ*(E.L[i,j]   - E.L[i,j-1]) / dy
            sl.Ly[i,j] = minmod3(s1,s2,s3)
        end
    end

    return sl
end

function reconstruct_equilibrium_faces!(
    F::FaceReconstruction, E::Equilibrium, sl::EquilSlopes, par::Params
)
    nx, ny = nx_tot(par), ny_tot(par)
    dx, dy = par.dx, par.dy

    for j in 1:ny, i in 1:nx
        # East/West
        F.uE[i,j] = E.u[i,j] + 0.5*dx*sl.ux[i,j]
        F.uW[i,j] = E.u[i,j] - 0.5*dx*sl.ux[i,j]
        F.vE[i,j] = E.v[i,j] + 0.5*dx*sl.vx[i,j]
        F.vW[i,j] = E.v[i,j] - 0.5*dx*sl.vx[i,j]

        F.KE[i,j] = E.K[i,j] + 0.5*dx*sl.Kx[i,j]
        F.KW[i,j] = E.K[i,j] - 0.5*dx*sl.Kx[i,j]
        F.LE[i,j] = E.L[i,j] + 0.5*dx*sl.Lx[i,j]
        F.LW[i,j] = E.L[i,j] - 0.5*dx*sl.Lx[i,j]

        # North/South
        F.uN[i,j] = E.u[i,j] + 0.5*dy*sl.uy[i,j]
        F.uS[i,j] = E.u[i,j] - 0.5*dy*sl.uy[i,j]
        F.vN[i,j] = E.v[i,j] + 0.5*dy*sl.vy[i,j]
        F.vS[i,j] = E.v[i,j] - 0.5*dy*sl.vy[i,j]

        F.KN[i,j] = E.K[i,j] + 0.5*dy*sl.Ky[i,j]
        F.KS[i,j] = E.K[i,j] - 0.5*dy*sl.Ky[i,j]
        F.LN[i,j] = E.L[i,j] + 0.5*dy*sl.Ly[i,j]
        F.LS[i,j] = E.L[i,j] - 0.5*dy*sl.Ly[i,j]
    end

    return F
end

function reconstruct_h_faces!(
    F::FaceReconstruction, E::Equilibrium, TF::TopographyFaces, par::Params
)
    nx, ny = nx_tot(par), ny_tot(par)
    g = par.g

    @assert size(E.Vface,1) == nx+1 && size(E.Vface,2) == ny
    @assert size(E.Uface,1) == nx   && size(E.Uface,2) == ny+1
    @assert size(TF.Bx,1)   == nx+1 && size(TF.Bx,2)   == ny
    @assert size(TF.By,1)   == nx   && size(TF.By,2)   == ny+1

    for j in 1:ny, i in 1:nx
        # x-faces: Vface[iface,j], Bx[iface,j], iface = i or i+1
        F.hE[i,j] = F.KE[i,j]/g + E.Vface[i+1,j] - TF.Bx[i+1,j]
        F.hW[i,j] = F.KW[i,j]/g + E.Vface[i,  j] - TF.Bx[i,  j]

        # y-faces: Uface[i,jface], By[i,jface], jface = j or j+1
        F.hN[i,j] = F.LN[i,j]/g - E.Uface[i,j+1] - TF.By[i,j+1]
        F.hS[i,j] = F.LS[i,j]/g - E.Uface[i,j  ] - TF.By[i,j  ]
    end

    return F
end


"""
    reconstruct_section2!(F, E, sl, S, TF, par)

Given cell averages S and topography faces TF, compute:

1. Equilibrium vars u,v,U,V,K,L
2. Slopes in x,y with TVD limiter
3. Face values of u,v,K,L and h at E/W/N/S

All arrays are modified in-place.
"""
function reconstruct_section2!(
    F::FaceReconstruction,
    E::Equilibrium,
    sl::EquilSlopes,
    S::State,
    TF::TopographyFaces,
    par::Params
)
    # Ensure ghosts of conservative vars are up-to-date
    fill_periodic!(S, par)

    # Copy topography centers if you use TF.Bc as reference
    S.B .= TF.Bc

    compute_equilibrium!(E, S, par)
    compute_equilibrium_slopes!(sl, E, par)
    reconstruct_equilibrium_faces!(F, E, sl, par)
    reconstruct_h_faces!(F, E, TF, par)

    return F
end

###################### Section 3 #################################################
###################### X–fluxes #################################################

mutable struct FluxX
    F1::Array{Float64,2}   # mass flux
    F2::Array{Float64,2}   # x-momentum flux
    F3::Array{Float64,2}   # y-momentum flux
end

function FluxX(par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    FluxX(zeros(nx+1, ny), zeros(nx+1, ny), zeros(nx+1, ny))
end

@inline function x_wavespeeds(hL, uL, hR, uR, g)
    cL = sqrt(max(g*hL, 0.0))
    cR = sqrt(max(g*hR, 0.0))
    a_plus  = max(max(uL + cL, uR + cR), 0.0)
    a_minus = min(min(uL - cL, uR - cR), 0.0)
    return a_plus, a_minus
end

"""
    build_Fx!(Fx, Frec, par)

Compute the x-direction CU fluxes F_{i-1/2,j} for all interfaces
using the reconstructed face values in `Frec`.

Indexing convention:
- For interface between cells `i-1` and `i` (1<i≤nx_tot), the flux is
  stored in `Fx.F*(i,j)`.

We only need fluxes 2:nx in each row j; index 1 is unused.
"""
function build_Fx!(Fx::FluxX, Frec::FaceReconstruction, par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    g = par.g
    eps_den = 1e-13

    fill!(Fx.F1, 0.0)
    fill!(Fx.F2, 0.0)
    fill!(Fx.F3, 0.0)

    # Interfaces: i = 1 .. nx+1
    # Interface i is BETWEEN cells (i-1) and (i)
    for j in 1:ny
        for i in 1:(nx+1)

            iL = i - 1      # left cell index
            iR = i          # right cell index

            # ----------------------------------------
            # Handle ghost boundaries safely
            # ----------------------------------------
            # clamp onto valid cell indices 1..nx
            iL_cl = clamp(iL, 1, nx)
            iR_cl = clamp(iR, 1, nx)

            # States from reconstructed faces:

            # Left state: use east face of cell iL
            hL = Frec.hE[iL_cl, j]
            uL = Frec.uE[iL_cl, j]
            vL = Frec.vE[iL_cl, j]

            # Right state: use west face of cell iR
            hR = Frec.hW[iR_cl, j]
            uR = Frec.uW[iR_cl, j]
            vR = Frec.vW[iR_cl, j]

            # positivity
            hL = max(hL, 0.0)
            hR = max(hR, 0.0)

            # wavespeeds
            a_plus, a_minus = x_wavespeeds(hL, uL, hR, uR, g)
            den = a_plus - a_minus

            F1 = 0.0
            F2 = 0.0
            F3 = 0.0

            if abs(den) < eps_den
                # degeneracy/sonic fix
                F1 = 0.5 * (hL*uL + hR*uR)
                F2 = 0.5 * (hL*uL^2 + 0.5*g*hL^2 +
                            hR*uR^2 + 0.5*g*hR^2)
            else
                # CU flux
                F1 = (a_plus*hL*uL - a_minus*hR*uR)/den +
                     (a_plus*a_minus/den)*(hR - hL)

                FL = hL*uL^2 + 0.5*g*hL^2
                FR = hR*uR^2 + 0.5*g*hR^2

                F2 = (a_plus*FL - a_minus*FR)/den +
                     (a_plus*a_minus/den)*(hR*uR - hL*uL)
            end

            # upwind for F3
            if (uL + uR) > 0
                F3 = hL*uL*vL
            else
                F3 = hR*uR*vR
            end

            # store at interface i
            Fx.F1[i,j] = F1
            Fx.F2[i,j] = F2
            Fx.F3[i,j] = F3
        end
    end

    return Fx
end


###################### Y–fluxes #################################################

mutable struct FluxY
    G1::Array{Float64,2}   # mass flux
    G2::Array{Float64,2}   # x-momentum flux
    G3::Array{Float64,2}   # y-momentum flux
end

function FluxY(par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    FluxY(zeros(nx, ny+1), zeros(nx, ny+1), zeros(nx, ny+1))
end

@inline function y_wavespeeds(hD, vD, hU, vU, g)
    # D = "down" (south), U = "up" (north)
    cD = sqrt(max(g*hD, 0.0))
    cU = sqrt(max(g*hU, 0.0))
    b_plus  = max(max(vD + cD, vU + cU), 0.0)
    b_minus = min(min(vD - cD, vU - cU), 0.0)
    return b_plus, b_minus
end

"""
    build_Gy!(Gy, Frec, par)

Compute the y-direction CU fluxes G_{i,j-1/2} for all interfaces
using the reconstructed face values in `Frec`.

Indexing convention:
- For interface between cells `j-1` and `j` (1<j≤ny_tot), the flux is
  stored in `Gy.G*(i,j)`.

We only need fluxes 2:ny in each column i; index 1 is unused.
"""
function build_Gy!(Gy::FluxY, Frec::FaceReconstruction, par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    g = par.g
    eps_den = 1e-13

    fill!(Gy.G1, 0.0)
    fill!(Gy.G2, 0.0)
    fill!(Gy.G3, 0.0)

    # Interfaces: j = 1 .. ny+1 (between rows j-1 and j)
    for i in 1:nx
        for j in 1:(ny+1)

            jD = j - 1
            jU = j

            # clamp into 1..ny for safe access
            jD_cl = clamp(jD, 1, ny)
            jU_cl = clamp(jU, 1, ny)

            # Down state from north face of cell jD
            hD = Frec.hN[i, jD_cl]
            uD = Frec.uN[i, jD_cl]
            vD = Frec.vN[i, jD_cl]

            # Up state from south face of cell jU
            hU = Frec.hS[i, jU_cl]
            uU = Frec.uS[i, jU_cl]
            vU = Frec.vS[i, jU_cl]

            hD = max(hD, 0.0)
            hU = max(hU, 0.0)

            # wavespeeds
            b_plus, b_minus = y_wavespeeds(hD, vD, hU, vU, g)
            den = b_plus - b_minus

            G1 = 0.0
            G2 = 0.0
            G3 = 0.0

            if abs(den) < eps_den
                G1 = 0.5 * (hD*vD + hU*vU)
                G3 = 0.5 * (hD*vD^2 + 0.5*g*hD^2 +
                            hU*vU^2 + 0.5*g*hU^2)
            else
                G1 = (b_plus*hD*vD - b_minus*hU*vU)/den +
                     (b_plus*b_minus/den)*(hU - hD)

                GD = hD*vD^2 + 0.5*g*hD^2
                GU = hU*vU^2 + 0.5*g*hU^2

                G3 = (b_plus*GD - b_minus*GU)/den +
                     (b_plus*b_minus/den)*(hU*vU - hD*vD)
            end

            # upwind in v
            if (vD + vU) > 0
                G2 = hD*uD*vD
            else
                G2 = hU*uU*vU
            end

            Gy.G1[i,j] = G1
            Gy.G2[i,j] = G2
            Gy.G3[i,j] = G3
        end
    end

    return Gy
end

mutable struct Sources
    SB2::Array{Float64,2}   # bathymetry source in x-momentum
    SB3::Array{Float64,2}   # bathymetry source in y-momentum
    SC2::Array{Float64,2}   # Coriolis source in x-momentum
    SC3::Array{Float64,2}   # Coriolis source in y-momentum
end

function Sources(par::Params)
    nx, ny = nx_tot(par), ny_tot(par)
    Sources(zeros(nx,ny), zeros(nx,ny), zeros(nx,ny), zeros(nx,ny))
end

function build_bathymetry_sources!(
    Src::Sources, S::State, TF::TopographyFaces, par::Params
)
    nx, ny = nx_tot(par), ny_tot(par)
    g = par.g
    dx, dy = par.dx, par.dy

    # Bx size: (nx+1, ny)  -> faces between i and i+1
    # By size: (nx, ny+1)  -> faces between j and j+1

    for j in 1:ny
        for i in 1:nx
            h̄ = S.h[i,j]

            dBdx = (TF.Bx[i+1,j] - TF.Bx[i,j]) / dx
            dBdy = (TF.By[i,j+1] - TF.By[i,j]) / dy

            Src.SB2[i,j] = -g * h̄ * dBdx
            Src.SB3[i,j] = -g * h̄ * dBdy
        end
    end

    return Src
end

function build_coriolis_sources!(Src::Sources, S::State, par::Params)
    nx, ny = nx_tot(par), ny_tot(par)

    for j in 1:ny
        fk = coriolis_at_row(j, par)
        for i in 1:nx
            hv̄ = S.hv[i,j]
            hū = S.hu[i,j]
            Src.SC2[i,j] =  fk * hv̄
            Src.SC3[i,j] = -fk * hū
        end
    end

    return Src
end

function build_sources!(Src::Sources, S::State, TF::TopographyFaces, par::Params)
    build_bathymetry_sources!(Src, S, TF, par)
    build_coriolis_sources!(Src, S, par)
    return Src
end

###################### RHS #################################################

"Set all components of a State to zero."
function zero_state!(S::State)
    fill!(S.h , 0.0)
    fill!(S.hu, 0.0)
    fill!(S.hv, 0.0)
    return S
end


"""
    rhs!(dS, S, TF, par, E, sl, Frec, Fx, Gy, Src)

Compute the semi-discrete RHS dS = dq/dt.

- `S`    : current state (with ghosts)
- `dS`   : output RHS (same shape as S)
- `TF`   : TopographyFaces
- `par`  : Params
- `E`    : Equilibrium workspace
- `sl`   : EquilSlopes workspace
- `Frec` : FaceReconstruction workspace
- `Fx`   : FluxX workspace
- `Gy`   : FluxY workspace
- `Src`  : Sources workspace
"""
function rhs!(dS::State, S::State,
              TF::TopographyFaces,
              par::Params,
              E::Equilibrium,
              sl::EquilSlopes,
              Frec::FaceReconstruction,
              Fx::FluxX,
              Gy::FluxY,
              Src::Sources)

    nx, ny = nx_tot(par), ny_tot(par)
    ng     = par.ng
    dx, dy = par.dx, par.dy

    zero_state!(dS)

    # 1) Reconstruction (updates ghosts and builds face values)
    reconstruct_section2!(Frec, E, sl, S, TF, par)

    # 2) Fluxes and sources
    build_Fx!(Fx, Frec, par)
    build_Gy!(Gy, Frec, par)
    build_sources!(Src, S, TF, par)

    # 3) Flux divergence + sources on PHYSICAL cells only
    for j in (ng+1):(ng+par.Ny)
        for i in (ng+1):(ng+par.Nx)

            # x-direction: interfaces at i (left) and i+1 (right)
            dF1dx = (Fx.F1[i+1, j] - Fx.F1[i, j]) / dx
            dF2dx = (Fx.F2[i+1, j] - Fx.F2[i, j]) / dx
            dF3dx = (Fx.F3[i+1, j] - Fx.F3[i, j]) / dx

            # y-direction: interfaces at j (bottom) and j+1 (top)
            dG1dy = (Gy.G1[i, j+1] - Gy.G1[i, j]) / dy
            dG2dy = (Gy.G2[i, j+1] - Gy.G2[i, j]) / dy
            dG3dy = (Gy.G3[i, j+1] - Gy.G3[i, j]) / dy

            # continuity
            dS.h[i,j]  = -(dF1dx + dG1dy)

            # x-momentum
            dS.hu[i,j] = -(dF2dx + dG2dy) +
                          Src.SB2[i,j] + Src.SC2[i,j]

            # y-momentum
            dS.hv[i,j] = -(dF3dx + dG3dy) +
                          Src.SB3[i,j] + Src.SC3[i,j]
        end
    end

    return dS
end


"dest .= src."
function copy_state!(dest::State, src::State)
    dest.h  .= src.h
    dest.hu .= src.hu
    dest.hv .= src.hv
    return dest
end

"Y .+= α * X."
function axpy_state!(α::Float64, X::State, Y::State)
    @. Y.h  += α * X.h
    @. Y.hu += α * X.hu
    @. Y.hv += α * X.hv
    return Y
end

"""
    rk2_step!(S, dt, TF, par,
              E, sl, Frec, Fx, Gy, Src,
              k1, k2, Sstage)

Advance one SSP-RK2 step: S^{n+1} from S^n.

All workspace arguments are modified in-place.
"""
function rk2_step!(S::State, dt::Float64,
                   TF::TopographyFaces,
                   par::Params,
                   E::Equilibrium,
                   sl::EquilSlopes,
                   Frec::FaceReconstruction,
                   Fx::FluxX,
                   Gy::FluxY,
                   Src::Sources,
                   k1::State,
                   k2::State,
                   Sstage::State)

    # Stage 1: k1 = f(S^n)
    rhs!(k1, S, TF, par, E, sl, Frec, Fx, Gy, Src)

    # Sstage = S + dt * k1
    copy_state!(Sstage, S)
    axpy_state!(dt, k1, Sstage)

    # Stage 2: k2 = f(Sstage)
    rhs!(k2, Sstage, TF, par, E, sl, Frec, Fx, Gy, Src)

    # Final update: S^{n+1} = S^n + 0.5*dt*(k1 + k2)
    @. S.h  = S.h  + 0.5*dt*(k1.h  + k2.h)
    @. S.hu = S.hu + 0.5*dt*(k1.hu + k2.hu)
    @. S.hv = S.hv + 0.5*dt*(k1.hv + k2.hv)

    return S
end




end #module