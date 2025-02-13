using KernelAbstractionsFV

domain = (0.0, 1.0)
nx = 800
grid = make_grid(domain, nx)
equations = Euler1D(1.4)

function initial_condition_dwave(x, t, equations::Euler1D)
    rho = 2.0 + 0.2*sin(2*pi*(x - 0.1*t))
    v1 = 0.1
    p = 1.0
    return prim2cons((rho, v1, p), equations)
end

surface_flux = flux_rusanov
semi = SemiDiscretizationHyperbolic(grid, equations, surface_flux, initial_condition_dwave)
tspan = (0.0, 0.1)
ode = ODE(semi, tspan)
Ccfl = 0.9
save_time_interval = 0.0
param = Parameters(Ccfl, save_time_interval)

sol = solve(ode, param)

@show sol.l1, sol.l2