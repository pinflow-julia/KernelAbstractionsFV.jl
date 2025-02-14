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

using GPUArraysCore

"""
    get_node_vars(u, equations, solver, indices...)

Get the conservative variables specified indices as an SVector.
"""
@inline function get_node_vars(u, equations, solver::AbstractSpatialSolver, indices...)
    # Copied from Trixi.jl
    @allowscalar SVector(ntuple(@inline(v->u[v, indices...]), Val(nvariables(equations))))
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
    # set_initial_value!(cache, grid, equations, initial_condition)
    # TODO - This works only for 1-D!
    set_initial_value_kernel!(backend_kernel)(
        cache.u, grid.xc, equations, initial_condition; ndrange = grid.nx+2)
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
function adjust_time_step(ode, param, t)
   # Adjust to reach final time exactly
   final_time = ode.tspan[2]
   (; save_time_interval) = param
    (; dt) = ode.semi.cache
   @allowscalar if t + dt[1] > final_time
      dt[1] = final_time - t
      return nothing
   end

   # Adjust to reach next solution saving time
   if save_time_interval > 0.0
      next_save_time = ceil(t/save_time_interval) * save_time_interval
      # If t is not a plotting time, we check if the next time
      # would step over the plotting time to adjust dt
      if abs(t-next_save_time) > 1e-10 && t + dt[1] - next_save_time > 1e-10
         dt[1] = next_save_time - t
         return nothing
      end
   end
   return nothing
end

"""
    update_solution!(semi, dt)

Update the solution using the explicit method.
"""
function update_solution!(semi)
    (; cache) = semi
    (; u, res, dt) = cache
    res .= 0.0
    compute_residual!(semi)
    @. u -= dt[1]*res
end

"""
    solve(ode, param)

Solve the conservation law.
"""
function solve(ode::ODE, param::Parameters)
    (; semi, tspan) = ode
    (; grid, cache, boundary_conditions) = semi
    (; dt) = cache
    Tf = tspan[2]

    it, t = 0, 0.0
    while t < Tf
       l1, l2, linf = compute_error(semi, t)
       compute_dt!(semi, param)
       adjust_time_step(ode, param, t)
       update_ghost_values!(cache, grid, boundary_conditions)
       update_solution!(semi)

       @show l1, l2, linf
       t += dt[1]; it += 1
       @show t, dt, it
    end
    l1, l2, linf = compute_error(semi, t)

    sol = (; cache.u, semi, l1, l2, linf)
    return sol
end

include("FV1D.jl")
