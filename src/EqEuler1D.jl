import Trixi: varnames, prim2cons, cons2prim, cons2cons
using Trixi: nvariables

"""
    Euler1D

1D Euler equations.
"""
struct Euler1D{RealT <: Real} <: AbstractEquations{1, 3}
    gamma::RealT
end

"""
    varnames(::Euler1D)

Return the variable names for the Euler1D equations.
"""
function varnames(::typeof(cons2cons), ::Euler1D)
    return ("rho", "rho*u", "E")
end

"""
    prim2cons(u, equations::Euler1D)

Convert primitive to conservative variables for the Euler1D equations.
"""
function prim2cons(u, equations::Euler1D)
    (; gamma) = equations
    rho, v1, p = u
    return (rho, rho*v1, p/(gamma-1) + 0.5*rho*v1^2)
end

"""
    cons2prim(u, equations::Euler1D)

Convert conservative to primitive variables for the Euler1D equations.
"""
function cons2prim(u, equations::Euler1D)
    (; gamma) = equations
    rho, rho_v1, E = u
    v1 = rho_v1 / rho
    p = (gamma-1) * (E - 0.5*rho*v1^2)
    return (rho, v1, p)
end

"""
    cons2cons(u, equations::Euler1D)

Convert conservative to conservative variables for the Euler1D equations.
"""
function cons2cons(u, equations::Euler1D)
    return u
end

