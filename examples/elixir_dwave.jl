using KernelAbstractionsFV
using Trixi
using KernelAbstractions

RealT = Float32
domain = (0.0f0, 1.0f0)
nx = 800
grid = make_grid(domain, nx)
equations = CompressibleEulerEquations1D(1.4f0)

function initial_condition_dwave(x, t, equations::CompressibleEulerEquations1D)
    RealT = eltype(x)
    rho = map(RealT, 2.0 + 0.2*sin(2*pi*(x - 0.1*t)))
    v1 = 0.1f0
    p = 1.0f0
    return prim2cons((rho, v1, p), equations)
end
surface_flux = flux_rusanov
semi = SemiDiscretizationHyperbolic(grid, equations, surface_flux, initial_condition_dwave)
tspan = map(RealT, (0.0, 0.1))
ode = ODE(semi, tspan)
Ccfl = map(RealT, 0.9)
save_time_interval = map(RealT, 0.0)
param = Parameters(Ccfl, save_time_interval)

sol = solve(ode, param)

@show sol.l1, sol.l2