import Trixi: varnames, prim2cons, cons2prim, cons2cons, max_abs_speeds, flux
using Trixi: nvariables, AbstractEquations

"""
    Euler1D

1D Euler equations.
"""
struct Euler1D{RealT <: Real} <: AbstractEquations{1, 3}
    gamma::RealT
end

"""
    flux(u, orientation::Integer, equations::Euler1D)

Compute the flux for the Euler1D equations.
"""
function flux(u, orientation::Integer, equations::Euler1D)
    rho, rho_v1, E = u
    v1 = rho_v1 / rho
    p = (equations.gamma - 1) * (E - 0.5f0*rho*v1^2)
    return SVector(rho_v1, rho*v1^2 + p, (E + p)*v1)
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
    return SVector(rho, rho*v1, p/(gamma-1.0f0) + 0.5f0*rho*v1^2)
end

"""
    cons2prim(u, equations::Euler1D)

Convert conservative to primitive variables for the Euler1D equations.
"""
function cons2prim(u, equations::Euler1D)
    (; gamma) = equations
    rho, rho_v1, E = u
    v1 = rho_v1 / rho
    p = (gamma-1.0f0) * (E - 0.5f0*rho*v1^2)
    return (rho, v1, p)
end

"""
    cons2cons(u, equations::Euler1D)

Convert conservative to conservative variables for the Euler1D equations.
"""
function cons2cons(u, equations::Euler1D)
    return u
end

"""
    max_abs_speeds(u, equation::Euler1D)

Compute the maximum absolute wave speeds for the Euler1D equations, which are
the eigen values of the flux.
"""
@inline function max_abs_speeds(u, equations::Euler1D)
    rho, u, p = cons2prim(u, equations)
    (; gamma) = equations

    c = sqrt(gamma*p/rho) # sound speed
    return abs(u) + c # local wave speed
end
