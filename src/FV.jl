using Trixi: AbstractEquations
using TimerOutputs

import Trixi: get_node_vars

"""
    AbstractSpatialSolver

Abstract type for solvers that specify the spatial discretization.
"""
abstract type AbstractSpatialSolver end

"""
    AbstractFiniteVolumeSolver

Finite volume spatial discretization.
"""
struct FiniteVolumeSolver <: AbstractSpatialSolver end


"""
    AbstractBoundaryCondition

Abstract type for boundary conditions.
"""
abstract type AbstractBoundaryCondition end

# Returns u[:, indices...] as an SVector. size(u, 1) should thus be
# known at compile time in the caller and passed via Val()
# (Taken from Benedict's fork of Trixi.jl)
@inline function get_node_vars_gpu(u, ::Val{N}, indices...) where {N}
    # There is a cut-off at `n == 10` inside of the method
    # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
    # in Julia `v1.5`, leading to type instabilities if
    # more than ten variables are used. That's why we use
    # `Val(...)` below.
    # We use `@inline` to make sure that the `getindex` calls are
    # really inlined, which might be the default choice of the Julia
    # compiler for standard `Array`s but not necessarily for more
    # advanced array types such as `PtrArray`s, cf.
    # https://github.com/JuliaSIMD/VectorizationBase.jl/issues/55
    SVector(ntuple(@inline(v->u[v, indices...]), N))
end

"""
    InflowBC

Specifies inflow boundary condition.
"""
struct InflowBC <: AbstractBoundaryCondition end

"""
    OutflowBC

Specifies outflow boundary condition.
"""
struct OutflowBC <: AbstractBoundaryCondition end

"""
    PeriodicBC

Specifies periodic boundary condition.
"""
struct PeriodicBC <: AbstractBoundaryCondition end

"""
    SemiDiscretizationHyperbolic

Struct containing everything about the spatial discretization, and the cache
used throughout the simulation.
"""
struct SemiDiscretizationHyperbolic{Grid, Equations <: AbstractEquations, SurfaceFlux, IC, BC,
                                    Solver, Cache, CacheCPUOnly}
    grid::Grid
    equations::Equations
    surface_flux::SurfaceFlux
    initial_condition::IC
    boundary_conditions::BC
    solver::Solver
    cache::Cache
    cache_cpu_only::CacheCPUOnly
end

"""
    SemiDiscretizationHyperbolic(grid, equations, initial_condition, boundary_condition)

Constructor for the SemiDiscretizationHyperbolic struct to ensure periodic boundary conditions
are used by default.
"""
function SemiDiscretizationHyperbolic(grid, equations, surface_flux, initial_condition;
    solver = FiniteVolumeSolver(),
    boundary_conditions = BoundaryConditions(PeriodicBC(), PeriodicBC()),
    backend_kernel = KernelAbstractions.CPU(),
    cache = (;))
    timer = TimerOutput()

    cache = (;cache..., create_cache(equations, grid, initial_condition, backend_kernel)...)
    cache_cpu_only = (; timer)

    SemiDiscretizationHyperbolic(grid, equations, surface_flux, initial_condition,
                                 boundary_conditions, solver, cache, cache_cpu_only)
end

"""
    ODE

Struct containing semidiscretization plus the time interval.
"""
struct ODE{Semi <: SemiDiscretizationHyperbolic, RealT <: Real}
    semi::Semi
    tspan::Tuple{RealT, RealT}
end

"""
    Parameters

Struct containing the CFL number and the time interval for saving the solution.
"""
struct Parameters{RealT <: Real}
    Ccfl::RealT
    save_time_interval::RealT
end

"""
    adjust_time_step(problem, param, dt, t)

Adjusts the time step to reach the final time exactly and to reach the next solution saving time.
"""
function adjust_time_step(ode, param, dt, t)
   # Adjust to reach final time exactly
   final_time = ode.tspan[2]
   (; save_time_interval) = param
   if t + dt > final_time
      dt = final_time - t
      return dt
   end

   # Adjust to reach next solution saving time
   if save_time_interval > 0.0f0
      next_save_time = ceil(t/save_time_interval) * save_time_interval
      # If t is not a plotting time, we check if the next time
      # would step over the plotting time to adjust dt
      if abs(t-next_save_time) > 1e-10 && t + dt - next_save_time > 1e-10
         dt = next_save_time - t
         return dt
      end
   end
   return dt
end

"""
    update_solution!(semi, dt)

Update the solution using the explicit method.
"""
function update_solution!(semi, dt)
    (; cache, cache_cpu_only) = semi
    @timeit cache_cpu_only.timer "update_solution!" begin
    #! format: noindent
    (; u, res) = cache
    res .= 0.0f0
    compute_residual!(semi)
    u.parent .-= dt*res.parent # OffsetArrays work with broadcasting on GPU only with parent
    end # timer
end

"""
    solve(ode, param)

Solve the conservation law.
"""
function solve(ode::ODE, param::Parameters; maxiters = nothing)
    (; semi, tspan) = ode
    (; grid, cache, cache_cpu_only, boundary_conditions) = semi
    @timeit cache_cpu_only.timer "solve" begin
    #! format: noindent
    Tf = tspan[2]

    it, t = 0, 0.0f0
    while t < Tf
       l1, l2, linf = compute_error(semi, t)
       dt = compute_dt!(semi, param)
       dt = adjust_time_step(ode, param, dt, t)
       update_ghost_values!(cache, cache_cpu_only, grid, boundary_conditions)
       update_solution!(semi, dt)

       @show l1, l2, linf
       t += dt; it += 1
       @show t, dt, it
    end
    l1, l2, linf = compute_error(semi, t)

    sol = (; cache.u, semi, l1, l2, linf)
    end # timer
    print_timer(cache_cpu_only.timer)
    return sol
end

include("FV1D.jl")
