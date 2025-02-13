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

struct CartesianGrid1D{RealT <: Real}
    domain::Tuple{RealT,RealT}  # xmin, xmax
    nx::Int                  # nx - number of points
    xc::Array{RealT, 1}      # cell centers
    xf::Array{RealT, 1}      # cell faces
    dx::Array{RealT, 1}      # cell sizes
end

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
