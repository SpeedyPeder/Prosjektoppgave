cd(@__DIR__)
include("sweSim2D-Rotational.jl")
import .RotSW_CDKLM
using Plots; theme(:default)

limiter = :minmod
steps   = 10
nx, ny  = 100, 100
dx, dy  = 500.0, 500.0
g       = 9.81
dt      = 0.5
bc      = :reflective
Hmin    = 1e-3
f0      = 1e-4
beta    = 0

x = collect(range(0, step=dx, length=nx))
y = collect(range(0, step=dy, length=ny))

bfun(x,y) = 0.0
st, p = RotSW_CDKLM.init_state(x, y, bfun, f0, beta;
                               g=g, dt=dt, Hmin=Hmin,
                               limiter=limiter, bc=bc)

# FLAT water surface
h0 = 10.0
st.h .= h0
st.hu .= 0.0
st.hv .= 0.0

# sync q
@views begin
    st.q[1,:,:] .= st.h
    st.q[2,:,:] .= st.hu
    st.q[3,:,:] .= st.hv
end

println("Testing flat initial height = $h0")

for n in 1:steps
    RotSW_CDKLM.step_RK2!(st, p)
    println("step $n: hmin = $(minimum(st.h)), hmax = $(maximum(st.h))")
end