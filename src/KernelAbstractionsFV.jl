module KernelAbstractionsFV

using Trixi
using OffsetArrays
using StaticArrays
using Printf

include("FV.jl")

include("EqEuler1D.jl")

export make_grid, Euler1D, prim2cons, SemiDiscretizationHyperbolic

end # module KernelAbstractionsFV
