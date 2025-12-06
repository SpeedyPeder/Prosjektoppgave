include("advectionSim.jl")
using .AdvectionSim
using Plots; 
default(ms = 5, lw = 2, size = (1200, 900))


# Parameters
N  = 200
L  = 1.0
T  = 1.0
a  = 1
CFL = 0.6

# Mixed IC
ic = AdvectionSim.classic_advection_IC

# Make directory if needed
mkpath("plots_advection")

# Run comparison and save figures inside the folder:
res = AdvectionSim.advection_compare_limiters_separate(
    N, L, T;
    a = a,
    CFL = CFL,
    ic = ic,
    limiters = [:minmod, :mc, :superbee, :vanleer],
    saveprefix = "plots_advection/advection"
)
ref     = res[:mc]
x_ref   = ref.x
u_exact = ref.u_exact
u_num   = ref.u

# Collect numeric solutions from all limiters:
num = Dict(lim => res[lim].u for lim in keys(res))

nothing      


