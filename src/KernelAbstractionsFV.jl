module KernelAbstractionsFV

using Trixi
using OffsetArrays
using StaticArrays
using Printf

include("FV.jl")

include("equations.jl")
include("EqEuler1D.jl")

export make_grid, Euler1D, prim2cons, SemiDiscretizationHyperbolic, solve, ODE, Parameters,
       flux_rusanov

end # module KernelAbstractionsFV
