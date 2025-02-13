using Trixi: AbstractEquations

"""
    InflowBC

Specifies inflow boundary condition.
"""
struct InflowBC end

"""
    OutflowBC

Specifies outflow boundary condition.
"""
struct OutflowBC end

"""
    PeriodicBC

Specifies periodic boundary condition.
"""
struct PeriodicBC end

"""
    SemiDiscretizationHyperbolic

Struct containing everything about the spatial discretization, and the cache
used throughout the simulation.
"""
struct SemiDiscretizationHyperbolic{Grid, Equations <: AbstractEquations, IC, BC, Cache}
    grid::Grid
    equations::Equations
    initial_condition::IC
    boundary_conditions::BC
    cache::Cache
end

"""
    SemiDiscretizationHyperbolic(grid, equations, initial_condition, boundary_condition)

Constructor for the SemiDiscretizationHyperbolic struct to ensure periodic boundary conditions
are used by default.
"""
function SemiDiscretizationHyperbolic(grid, equations, initial_condition,
    boundary_conditions = BoundaryConditions(PeriodicBC(), PeriodicBC()), cache = (;))

    cache = (cache..., create_cache(equations, grid)...)
    SemiDiscretizationHyperbolic(grid, equations, initial_condition, boundary_conditions, cache)
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
   if t + dt[1] > final_time
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

include("FV1D.jl")