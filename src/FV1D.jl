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
    dx0::RealT # constant value of dx0
               # (TODO - This is used for a hacky way to compute errors,
               #         will not support nonuniform grids)
end

@kernel function linrange_kernel(start, stop, arr)
    i = @index(Global, Linear)
    N = length(arr)
    arr[i] = start + (stop - start) * (i - 1) / (N - 1)
end

function gpu_linrange(start, stop, N, RealT, backend)
    arr = KernelAbstractions.zeros(backend, RealT, N)  # GPU array
    KernelAbstractions.synchronize(backend)

    linrange_kernel(backend, 256)(start, stop, arr, ndrange=N)
    KernelAbstractions.synchronize(backend)
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
    dx0 = (xmax - xmin)/nx
    KernelAbstractions.synchronize(backend_kernel)

    xc = gpu_linrange(xmin+0.5f0*dx0, xmax-0.5f0*dx0, nx, RealT, backend_kernel) # 2 dummy elements
    KernelAbstractions.synchronize(backend_kernel)
    @printf("   Grid of with number of points = %d \n", nx)
    @printf("   xmin,xmax                     = %e, %e\n", xmin, xmax)
    @printf("   dx                            = %e\n", dx0)
    dx_ = dx0 .* KernelAbstractions.ones(backend_kernel, RealT, nx+2)
    dx = OffsetArray(dx_, OffsetArrays.Origin(0))
    KernelAbstractions.synchronize(backend_kernel)

    xf = gpu_linrange(xmin, xmax, nx+1, RealT, backend_kernel)
    KernelAbstractions.synchronize(backend_kernel)
    return CartesianGrid1D(domain, nx, xc, xf, dx, dx0)
end

"""
    create_cache(problem, grid)

Struct containing everything about the spatial discretization.
"""
function create_cache(equations, grid::CartesianGrid1D, initial_condition, backend_kernel)
    nvar = nvariables(equations)
    (; xc, nx) = grid
    RealT = eltype(xc)
    # Allocating variables

    u_ = allocate(backend_kernel, RealT, nvar, nx+2)
    u = OffsetArray(u_, OffsetArrays.Origin(1, 0))
    u_physical = @view u[:, 1:nx]
    res = copy(u) # dU/dt + res(U) = 0
    Fn = copy(u) # numerical flux
    Fn .= 0.0f0
    speeds = KernelAbstractions.zeros(backend_kernel, RealT, nx) # Wave speed estimate at each point for
                                                   # taking the maximum
    exact_array = KernelAbstractions.zeros(backend_kernel, RealT, nvar, nx) # Used to store exact solution in
                                                            # error computation
    error_array = copy(exact_array) # Uses to store pointwise in error computation

    # Put in initial value in u

    KernelAbstractions.synchronize(backend_kernel)

    set_initial_value_kernel!(backend_kernel)(
        u, xc, equations, initial_condition, 0.0f0; ndrange = nx)
        KernelAbstractions.synchronize(backend_kernel)

    cache = (; u, u_physical, speeds, res, Fn, exact_array, error_array, backend_kernel)

    return cache
end

"""
    compute_dt!(semi, param)

Compute the time step based on the CFL condition.
"""
@kernel function compute_max_speed_kernel!(speeds, u,
    equations::Union{Euler1D, CompressibleEulerEquations1D}, dx)
    i = @index(Global, Linear)
    nvar = Val(nvariables(equations))
    u_node = get_node_vars_gpu(u, nvar, i)
    local_speed = sum(max_abs_speeds(u_node, equations)) # Since Trixi equations return it
                                                         # as a tuple of one element
    speeds[i] = local_speed / dx[i]
end

function compute_dt!(semi::SemiDiscretizationHyperbolic{<:CartesianGrid1D}, param)
    (; grid, equations, solver, cache) = semi
    (; u, speeds, backend_kernel) = cache
    (; Ccfl) = param
    (; dx) = grid

    KernelAbstractions.synchronize(backend_kernel)

    compute_max_speed_kernel!(backend_kernel, 256)(
        speeds, u, equations, grid.dx; ndrange = grid.nx)
    KernelAbstractions.synchronize(backend_kernel)

    dt = Ccfl * 1.0f0 / maximum(speeds)
    return dt
end


"""
    set_initial_value!(grid, equations, u, initial_value)

Set the initial value of the solution.
"""
function set_initial_value!(cache, grid, equations::AbstractEquations{1}, initial_value)
    (; u) = cache
    (; nx, xc) = grid
    for i=1:nx
        u[:,i] .= initial_value(xc[i], 0.0f0, equations)
    end
end

@kernel function set_initial_value_kernel!(u, xc, equations::AbstractEquations{1},
                                           initial_value, t)
    i = @index(Global, Linear)
    u[:, i] .= initial_value(xc[i], t, equations)
end

"""
    apply_left_bc!(grid, left, cache)

Apply the left boundary condition.
"""
@kernel function apply_left_bc_kernel!(cache, left::PeriodicBC, nx)
    (; u) = cache
    u[:, 0] .= @views u[:, nx]
end

"""
    apply_right_bc!(grid, right, cache)

Apply the right boundary condition.
"""
@kernel function apply_right_bc_kernel!(cache, right::PeriodicBC, nx)
    (; u) = cache
    u[:, nx+1] .= @views u[:, 1]
end

"""
    update_ghost_values!(grid, boundary_conditions, cache)

Update the ghost values of the solution.
"""
function update_ghost_values!(cache, grid::CartesianGrid1D, boundary_conditions::BoundaryConditions)
    (; backend_kernel) = cache
    # TODO - Passing grid instead of grid.nx gives an inbits error, indicating that the `grid`
    # type has something which is not on GPU. This shouldn't happen because the grid has
    # nothing that the cache doesn't have.
    # https://github.com/JuliaGPU/CUDA.jl/issues/372
    apply_left_bc_kernel!(backend_kernel, 256)(
        cache, boundary_conditions.left, grid.nx, ndrange = 1)
    apply_right_bc_kernel!(backend_kernel, 256)(
        cache, boundary_conditions.right, grid.nx, ndrange = 1)
end

"""
    compute_error(semi, t)

Compute the error of the solution.
"""
function compute_error(semi, t)
    (; grid, equations, initial_condition, cache) = semi
    (; exact_array, error_array, backend_kernel, u_physical) = cache
    (; nx, xc, dx0) = grid

    KernelAbstractions.synchronize(backend_kernel)

    set_initial_value_kernel!(backend_kernel, 256)(
    exact_array, xc, equations, initial_condition, t, ndrange = nx)
    KernelAbstractions.synchronize(backend_kernel)
    error_array .= abs.(u_physical .- exact_array) # TODO - Does this have auto-sync?
    error_l1 = sum(error_array * dx0)
    error_l2 = sqrt.(sum(error_array.^2 * dx0))
    error_linf = maximum(error_array)
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

    (; grid, equations, solver, cache) = semi
    (; nx, dx) = grid
    (; Fn, res, backend_kernel) = cache
    # TODO: Is 256 an optimal workgroup size?
    KernelAbstractions.synchronize(backend_kernel)
    update_rhs_kernel!(backend_kernel,256)(Fn, res, equations, solver, dx; ndrange = nx)
    KernelAbstractions.synchronize(backend_kernel)
end

@kernel function update_rhs_kernel!(Fn, res, equations, solver, dx)
    i = @index(Global, Linear)
    nvar = Val(nvariables(equations))
    fn_rr = get_node_vars_gpu(Fn, nvar, i+1)
    fn_ll = get_node_vars_gpu(Fn, nvar, i)

    rhs = (fn_rr - fn_ll)/ dx[i]
    res[:, i] .= rhs
end

function compute_surface_fluxes!(semi)

    (; grid, equations, surface_flux, solver, cache) = semi
    (; nx) = grid
    (; u, Fn, backend_kernel) = cache
    # TODO: Is 256 an optimal workgroup size?
    KernelAbstractions.synchronize(backend_kernel)
    compute_surface_fluxes_kernel!(backend_kernel, 256)(Fn, u, equations, solver, surface_flux; ndrange = nx+1)
    KernelAbstractions.synchronize(backend_kernel)
end

@kernel function compute_surface_fluxes_kernel!(Fn, u, equations, solver, surface_flux)
    i = @index(Global, Linear)

    nvar = Val(nvariables(equations))

    ul = get_node_vars_gpu(u, nvar, i-1)
    ur = get_node_vars_gpu(u, nvar, i)

    fn = surface_flux(ul, ur, 1, equations)
    Fn[:, i] .= fn
end
