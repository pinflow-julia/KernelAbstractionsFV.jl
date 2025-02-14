using KernelAbstractions
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
    dx::AbstractArray      # cell sizes
end

"""
    make_grid(domain::Tuple{<:Real, <:Real}, nx)

Constructor for the CartesianGrid1D struct. It creates a uniform grid with nx points in the domain.
"""
function make_grid(domain::Tuple{<:Real, <:Real}, nx; backend_kernel = KernelAbstractions.CPU())
    xmin, xmax = domain
    RealT = eltype(domain)
    @assert xmin < xmax
    println("Making uniform grid of interval [", xmin, ", ", xmax,"]")
    dx1 = (xmax - xmin)/nx
    xc = LinRange(xmin+0.5*dx1, xmax-0.5*dx1, nx)
    @printf("   Grid of with number of points = %d \n", nx)
    @printf("   xmin,xmax                     = %e, %e\n", xmin, xmax)
    @printf("   dx                            = %e\n", dx1)
    RealT = eltype(xc)
    dx = allocate(backend_kernel, RealT, nx+2)
    dx .= dx1 * 1.0
    @show typeof(dx)
    #dx = OffsetArray(dx_, OffsetArrays.Origin(0))
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
    u = allocate(backend_kernel, RealT, nvar, nx+2)
    #u = OffsetArray(u_, OffsetArrays.Origin(1, 0))
    res = copy(u) # dU/dt + res(U) = 0
    Fn = copy(u) # numerical flux

    # TODO - dt is a vector to allow mutability. Is that necessary?
    #dt = allocate(backend_kernel, RealT, 1)
    dt = Vector{RealT}(undef, 1)
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
    max_speed = zero(eltype(dx))
    backend = get_backend(u)
    
    compute_dt_kernel!(backend)(max_speed, u, equations, solver, dx; ndrange = grid.nx)
    dt[1] = Ccfl * 1.0 / max_speed

    @show max_speed
end

@kernel function compute_dt_kernel!(max_speed, u, equations, solver, dx)
    i = @index(Global, Linear)
    u_node = collect(get_node_vars(u, equations, solver, i))
    max_speed = max_abs_speeds(u_node, equations) / dx[i+1]
    #max_speed = max(max_abs_speeds(u_node, equations)[1] / dx[i+1], max_speed)
end

function get_node_vars_new(u, equations, solver, index)
    return u[:,index]
end

"""
    set_initial_value!(grid, equations, u, initial_value)

Set the initial value of the solution.
"""
function set_initial_value!(cache, grid::CartesianGrid1D, equations::AbstractEquations{1},
                            initial_value)
    nx = grid.nx
    xc = grid.xc
    (; u, backend_kernel) = cache
    
    for i = 1:nx
       u[:,i+1] =  collect(initial_value(xc[i], 0.0, equations))
    end
    
end

"""
    apply_left_bc!(grid, left, cache)

Apply the left boundary condition.
"""
function apply_left_bc!(cache, left::PeriodicBC, grid::CartesianGrid1D)
    (; u) = cache
    u[:, 1] .= @views u[:, grid.nx+1]
end

"""
    apply_right_bc!(grid, right, cache)

Apply the right boundary condition.
"""
function apply_right_bc!(cache, right::PeriodicBC, grid::CartesianGrid1D)
    (; u) = cache
    u[:, grid.nx+2] .= @views u[:, 2]
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
    backend = get_backend(u)
    compute_errors_kernel!(get_backend(u))(error_l2, error_l1, error_linf,u, initial_condition,t, xc, equations, dx; ndrange = nx)
    synchronize(backend)

    error_l2 = sqrt(error_l2)
    return error_l1, error_l2, error_linf
end

@kernel function compute_errors_kernel!(error_l2, error_l1, error_linf,u, initial_condition,t, xc, equations, dx)
    i = @index(Global, Linear)
    u_exact = initial_condition(xc[i], t, equations)
    error = abs(u[1, i+1] - u_exact[1])
    error_l1 += error * dx[i+1]
    error_l2 += error^2 * dx[i+1]
    error_linf = max(error_linf,error)
end


"""
    compute_residual!(semi)

Compute the residual of the solution.
""" # TODO - Dispatch for 1D. The fact that it doesn't work indicates a bug in julia.
function compute_residual!(semi)
   
    compute_surface_fluxes!(semi)
    update_rhs!(semi)

end

function update_rhs!(semi)

    (; grid, equations, surface_flux, solver, cache) = semi
    (; nx, dx, xf) = grid
    (; u, Fn, res) = cache

    nx = grid.nx
    backend = get_backend(u)

    update_rhs_kernel!(backend)(res, Fn, dx, equations, solver; ndrange = nx)
    synchronize(backend)

end

@kernel function update_rhs_kernel!(res, Fn, dx, equations, solver)
    i = @index(Global, Linear)
    for k = 1:3
        res[k,i+1] += (Fn[k,i+2] - Fn[k,i+1])/dx[i]
    end
end

function compute_surface_fluxes!(semi)

    (; grid, equations, surface_flux, solver, cache) = semi
    (; nx, dx, xf) = grid
    (; u, Fn, res) = cache
    # TODO: Is 256 an optimal workgroup size?
    backend = get_backend(u)
    compute_surface_fluxes_kernel!(backend)(Fn, u, equations, solver, surface_flux; ndrange = nx+1)
    synchronize(backend)
end

@kernel function compute_surface_fluxes_kernel!(Fn, u, equations, solver, surface_flux)
    i = @index(Global, Linear)
        ul, ur = get_node_vars(u, equations, solver, i), get_node_vars(u, equations, solver, i+1)
        flux = surface_flux(ul, ur, 1, equations)
        for k = 1:3
        Fn[k, i+1] = flux[k] 
        end
end
