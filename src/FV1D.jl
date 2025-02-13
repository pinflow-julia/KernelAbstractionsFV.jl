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
    dx::OffsetVector{RealT}      # cell sizes
end

"""
    make_grid(domain::Tuple{<:Real, <:Real}, nx)

Constructor for the CartesianGrid1D struct. It creates a uniform grid with nx points in the domain.
"""
function make_grid(domain::Tuple{<:Real, <:Real}, nx)
    xmin, xmax = domain
    RealT = eltype(domain)
    @assert xmin < xmax
    println("Making uniform grid of interval [", xmin, ", ", xmax,"]")
    dx1 = (xmax - xmin)/nx
    xc = LinRange(xmin+0.5*dx1, xmax-0.5*dx1, nx)
    @printf("   Grid of with number of points = %d \n", nx)
    @printf("   xmin,xmax                     = %e, %e\n", xmin, xmax)
    @printf("   dx                            = %e\n", dx1)
    dx_ = dx1 .* ones(nx+2)
    dx = OffsetArray(dx_, OffsetArrays.Origin(0))
    xf = LinRange(xmin, xmax, nx+1)
    return CartesianGrid1D(domain, nx, collect(xc), collect(xf), dx)
end

"""
    create_cache(problem, grid)

Struct containing everything about the spatial discretization.
"""
function create_cache(equations, grid::CartesianGrid1D, backend_kernel)
    nvar = nvariables(equations)
    nx = grid.nx
    RealT = eltype(grid.xc)
    # Allocating variables

    u_ = allocate(backend_kernel, RealT, nvar, nx+2)
    u = OffsetArray(u_, OffsetArrays.Origin(1, 0))
    res = copy(u) # dU/dt + res(U) = 0
    Fn = copy(u) # numerical flux

    # TODO - dt is a vector to allow mutability. Is that necessary?
    dt = allocate(backend_kernel, RealT, 1)

    cache = (; u, res, Fn, dt, backend_kernel)

    return cache
end

"""
    compute_dt!(semi, param)

Compute the time step based on the CFL condition.
"""
function compute_dt!(semi::SemiDiscretizationHyperbolic{<:CartesianGrid1D}, param)
    (; grid, equations, solver, cache) = semi
    (; u, dt) = cache
    (; dx) = grid
    (; Ccfl) = param

    # Compute the maximum wave speed
    max_speed = zero(eltype(u))
    for i in 1:grid.nx
        u_node = get_node_vars(u, equations, solver, i)
        max_speed = max(max_abs_speeds(u_node, equations)[1] / dx[i], max_speed)
    end

    # Compute the time step
    dt[1] = Ccfl * 1.0 / max_speed
end

"""
    set_initial_value!(grid, equations, u, initial_value)

Set the initial value of the solution.
"""
function set_initial_value!(cache, grid::CartesianGrid1D, equations::AbstractEquations{1},
                            initial_value)
    nx = grid.nx
    xc = grid.xc
    (; u) = cache
    for i=1:nx
        u[:,i] .= initial_value(xc[i], 0.0, equations)
    end
end

"""
    apply_left_bc!(grid, left, cache)

Apply the left boundary condition.
"""
function apply_left_bc!(cache, left::PeriodicBC, grid::CartesianGrid1D)
    (; u) = cache
    u[:, 0] .= @views u[:, grid.nx]
end

"""
    apply_right_bc!(grid, right, cache)

Apply the right boundary condition.
"""
function apply_right_bc!(cache, right::PeriodicBC, grid::CartesianGrid1D)
    (; u) = cache
    u[:, grid.nx+1] .= @views u[:, 1]
end

"""
    update_ghost_values!(grid, boundary_conditions, cache)

Update the ghost values of the solution.
"""
function update_ghost_values!(cache, grid::CartesianGrid1D, boundary_conditions::BoundaryConditions)
    apply_left_bc!(cache, boundary_conditions.left, grid)
    apply_right_bc!(cache, boundary_conditions.right, grid)
end

"""
    compute_error(semi, t)

Compute the error of the solution.
"""
function compute_error(semi, t)
    (; grid, equations, initial_condition, cache) = semi
    (; u) = cache
    error_l2, error_l1, error_linf = (zero(eltype(u)) for _=1:3)
    (; nx, dx, xc) = grid
    for i=1:nx
       u_   = u[1, i]
       u_exact = initial_condition(xc[i], t, equations)
       error = abs(u_ - u_exact[1])
       error_l1 += error   * dx[i]
       error_l2 += error^2 * dx[i]
       error_linf = max(error_linf, error)
    end
    error_l2 = sqrt(error_l2)
    return error_l1, error_l2, error_linf
end

"""
    compute_surface_flux!(semi)

Compute the numerical flux at all faces.
"""

@kernel function compute_surface_flux_kernel!(semi)
    i, = @index(Global, NTuple)
    (; equations, surface_flux, solver, cache) = semi
    (; u, Fn) = cache
    # loop over faces
    ul, ur = get_node_vars(u, equations, solver, i-1), get_node_vars(u, equations, solver, i)
    Fn[:, i] .= surface_flux(ul, ur, 1, equations)
end

function compute_surface_flux!(
    semi::SemiDiscretizationHyperbolic{<:CartesianGrid1D}
    )
    (; grid, equations, surface_flux, solver, cache) = semi
    (; nx) = grid
    (; u, Fn) = cache
    # loop over faces
    for i=1:nx+1
        ul, ur = get_node_vars(u, equations, solver, i-1), get_node_vars(u, equations, solver, i)
        Fn[:, i] .= surface_flux(ul, ur, 1, equations)
    end
end

"""
    compute_residual!(semi)

Compute the residual of the solution.
""" # TODO - Dispatch for 1D. The fact that it doesn't work indicates a bug in julia.
function compute_residual!(
    # semi::SemidiscretizationHyperbolic{<:CartesianGrid1D}
    semi
    )
   (; grid, equations, surface_flux, solver, cache) = semi
   (; nx, dx, xf) = grid
   (; Fn, res) = cache
   # loop over faces
    (; backend_kernel) = cache
    kernel! = compute_surface_flux_kernel!(backend_kernel)
    kernel!(semi, ndrange = (nx+1,))

   # loop over elements
   for i=1:nx
      fn_rr = get_node_vars(Fn, equations, solver, i+1)
      fn_ll = get_node_vars(Fn, equations, solver, i)
      res[:, i] .+= (fn_rr - fn_ll)/ dx[i]
   end
end
