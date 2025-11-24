include("advectionSim.jl")
using .AdvectionSim
using Plots; default();

# Parameters
N  = 150
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
    saveprefix = "plots_advection/advection"   # <-- save here
)

# Access results
x_ref   = res.x
u_exact = res.u_exact
num     = res.num   # Dict: num[:mc], num[:minmod], ...

