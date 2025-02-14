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
struct CartesianGrid1D{RealT <: Real, ArrayType1, ArrayType2}
    domain::Tuple{RealT,RealT}  # xmin, xmax
    nx::Int                  # nx - number of points
    xc::ArrayType1      # cell centers
    xf::ArrayType1      # cell faces
    dx::ArrayType2      # cell sizes (TODO - This was for offset array)
end

@kernel function linrange_kernel(start, stop, arr)
    i = @index(Global)
    N = length(arr)
    if i ≤ N
        arr[i] = start + (stop - start) * (i - 1) / (N - 1)
    end
end

function gpu_linrange(start, stop, N, RealT, backend)
    arr = KernelAbstractions.zeros(backend, RealT, N)  # GPU array
    kernel = linrange_kernel(backend, N)               # Define kernel
    kernel(start, stop, arr, ndrange=N)                # Launch kernel
    KernelAbstractions.synchronize(backend)            # Sync to ensure execution is complete
    return arr
end

"""
    make_grid(domain::Tuple{<:Real, <:Real}, nx)

Constructor for the CartesianGrid1D struct. It creates a uniform grid with nx points in the domain.
"""
function make_grid(domain::Tuple{<:Real, <:Real}, nx, backend_kernel)
    xmin, xmax = domain
    RealT = eltype(domain)
    @assert xmin < xmax
    println("Making uniform grid of interval [", xmin, ", ", xmax,"]")
    dx1 = (xmax - xmin)/nx
    xc = gpu_linrange(xmin-0.5f0*dx1, xmax+0.5f0*dx1, nx+2, RealT, backend_kernel) # 2 dummy elements
    @printf("   Grid of with number of points = %d \n", nx)
    @printf("   xmin,xmax                     = %e, %e\n", xmin, xmax)
    @printf("   dx                            = %e\n", dx1)
    dx_ = dx1 .* KernelAbstractions.ones(backend_kernel, RealT, nx+2)
    # dx = OffsetArray(dx_, OffsetArrays.Origin(0)) # TODO - This doesn't work with GPU
    dx = dx_
    xf = gpu_linrange(xmin-dx1, xmax+dx1, nx+3, RealT, backend_kernel)
    return CartesianGrid1D(domain, nx, xc, xf, dx)
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
    # u = OffsetArray(u_, OffsetArrays.Origin(1, 0))
    u = u_
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
    for i in 2:grid.nx+1
        u_node = get_node_vars(u, equations, solver, i)
        @allowscalar max_speed = max(max_abs_speeds(u_node, equations)[1] / dx[i], max_speed)
    end

    # Compute the time step
    @allowscalar dt[1] = Ccfl * 1.0 / max_speed
end

"""
    set_initial_value!(grid, equations, u, initial_value)

Set the initial value of the solution.
"""
@kernel function set_initial_value_kernel!(u, xc, equations::AbstractEquations{1},
                                           initial_value)
    i = @index(Global, Linear)
    u[:,i] .= initial_value(xc[i], 0.0f0, equations)
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
    (; grid, equations, initial_condition, cache, solver) = semi
    (; u) = cache
    error_l2, error_l1, error_linf = (zero(eltype(u)) for _=1:3)
    (; nx, dx, xc) = grid
    @allowscalar for i=2:nx+1
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
    # TODO: Is 256 an optimal workgroup size?
    update_rhs_kernel!(get_backend(u),256)(Fn, res, equations, solver, dx; ndrange = nx+1)
end

@kernel function update_rhs_kernel!(Fn, res, equations, solver, dx)
    i = @index(Global, Linear)
    fn_rr = get_node_vars(Fn, equations, solver, i+1)
    fn_ll = get_node_vars(Fn, equations, solver, i)

    rhs = (fn_rr - fn_ll)/ dx[i]
    res[:, i+1] .= rhs
end

@kernel function update_rhs_kernel!(Fn, res, equations::CompressibleEulerEquations1D, solver, dx)
    i = @index(Global, Linear)
    # fn_rr = get_node_vars(Fn, equations, solver, i+1)
    # fn_ll = get_node_vars(Fn, equations, solver, i)

    fn_rr = SVector(Fn[1, i+1], Fn[2, i+1], Fn[3, i+1])
    fn_ll = SVector(Fn[1, i], Fn[2, i], Fn[3, i])
    rhs = (fn_rr - fn_ll)/ dx[i]
    res[:, i+1] .= rhs
end

function compute_surface_fluxes!(semi)

    (; grid, equations, surface_flux, solver, cache) = semi
    (; nx, dx, xf) = grid
    (; u, Fn, res) = cache
    # TODO: Is 256 an optimal workgroup size?
    compute_surface_fluxes_kernel!(get_backend(u), 256)(Fn, u, equations, solver, surface_flux; ndrange = nx+1)
end

@kernel function compute_surface_fluxes_kernel!(Fn, u, equations, solver, surface_flux)
    i = @index(Global, Linear)

    # TODO - Fix the names of ul, ur!!!
    ur = get_node_vars(u, equations, solver, i)
    ul = get_node_vars(u, equations, solver, i+1)
    fn = surface_flux(ul, ur, 1, equations)
    Fn[:, i] .= fn
end

@kernel function compute_surface_fluxes_kernel!(
    Fn, u, equations::CompressibleEulerEquations1D, solver, surface_flux)
    i = @index(Global, Linear)

    # TODO - Fix the names of ul, ur!!!
    ul = SVector(u[1, i+1], u[2, i+1], u[3, i+1])
    ur = SVector(u[1, i], u[2, i], u[3, i])
    fn = surface_flux(ul, ur, 1, equations)
    Fn[:, i] .= fn
end
