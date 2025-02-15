using Trixi: AbstractEquations

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

"""
    get_node_vars(u, equations, solver, indices...)

Get the conservative variables specified indices as an SVector.
"""
@inline function get_node_vars(u, equations, solver::AbstractSpatialSolver, indices...)
    # Copied from Trixi.jl
    SVector(ntuple(@inline(v->u[v, indices...]), Val(nvariables(equations))))
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
                                    Solver, Cache}
    grid::Grid
    equations::Equations
    surface_flux::SurfaceFlux
    initial_condition::IC
    boundary_conditions::BC
    solver::Solver
    cache::Cache
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

    cache = (;cache..., create_cache(equations, grid, backend_kernel)...)
    set_initial_value!(cache, grid, equations, initial_condition)
    SemiDiscretizationHyperbolic(grid, equations, surface_flux, initial_condition,
                                 boundary_conditions, solver, cache)
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
    (; cache) = semi
    (; u, res) = cache
    res .= 0.0f0
    compute_residual!(semi)
    @. u -= dt*res
end

"""
    solve(ode, param)

Solve the conservation law.
"""
function solve(ode::ODE, param::Parameters; maxiters = nothing)
    (; semi, tspan) = ode
    (; grid, cache, boundary_conditions) = semi
    Tf = tspan[2]

    it, t = 0, 0.0f0
    while t < Tf
       l1, l2, linf = compute_error(semi, t)
       dt = compute_dt!(semi, param)
       dt = adjust_time_step(ode, param, dt, t)
       update_ghost_values!(cache, grid, boundary_conditions)
       update_solution!(semi, dt)

       @show l1, l2, linf
       t += dt; it += 1
       @show t, dt, it
    end
    l1, l2, linf = compute_error(semi, t)

    sol = (; cache.u, semi, l1, l2, linf)
    return sol
end

include("FV1D.jl")
