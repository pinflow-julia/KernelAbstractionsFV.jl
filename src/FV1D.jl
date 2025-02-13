"""
    BoundaryConditions

Struct containing the left and right boundary conditions.
"""

struct BoundaryConditions{LeftBC, RightBC}
    left::LeftBC
    right::RightBC
    function BoundaryConditions(left, right)
       if left isa PeriodicBC || right isa PeriodicBC
          @assert left isa PeriodicBC && right isa PeriodicBC
       end
       new{typeof(left), typeof(right)}(left, right)
    end
end

"""
    CartesianGrid1D

Struct containing the 1-D Cartesian grid information.
"""
struct CartesianGrid1D{RealT <: Real}
    domain::Tuple{RealT,RealT}  # xmin, xmax
    nx::Int                  # nx - number of points
    xc::Array{RealT, 1}      # cell centers
    xf::Array{RealT, 1}      # cell faces
    dx::Array{RealT, 1}      # cell sizes
end

"""
    make_grid(domain::Tuple{<:Real, <:Real}, nx)

Constructor for the CartesianGrid1D struct. It creates a uniform grid with nx points in the domain.
"""
function make_grid(domain::Tuple{<:Real, <:Real}, nx)
    xmin, xmax = domain
    @assert xmin < xmax
    println("Making uniform grid of interval [", xmin, ", ", xmax,"]")
    dx1 = (xmax - xmin)/nx
    xc = LinRange(xmin+0.5*dx1, xmax-0.5*dx1, nx)
    @printf("   Grid of with number of points = %d \n", nx)
    @printf("   xmin,xmax                     = %e, %e\n", xmin, xmax)
    @printf("   dx                            = %e\n", dx1)
    dx = dx1 .* ones(nx)
    xf = LinRange(xmin, xmax, nx+1)
    return CartesianGrid1D(domain, nx, collect(xc), collect(xf), dx)
end

"""
    create_cache(problem, grid)

Struct containing everything about the spatial discretization.
"""
function create_cache(equations, grid::CartesianGrid1D)
    nvar = nvariables(equations)
    nx = grid.nx
    RealT = eltype(grid.xc)
    # Allocating variables

    u = OffsetArray(zeros(RealT, nvar, nx+2), OffsetArrays.Origin(1, 0))
    res = copy(u) # dU/dt + res(U) = 0

    # TODO - dt is a vector to allow mutability. Is that necessary?
    dt = Vector{RealT}(undef, 1)

    cache = (; u, res, dt)

    return cache
end
