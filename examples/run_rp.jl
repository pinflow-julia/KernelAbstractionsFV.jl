using KernelAbstractionsFV
using Trixi
using KernelAbstractions

RealT = Float32
domain = map(RealT, (0.0, 1.0))
nx = 400
backend_kernel = MyCPU()

grid = make_grid(domain, nx, backend_kernel);
equations = Euler1D(1.4f0)

function initial_condition_rp(x, t, equations::Euler1D)
    RealT = eltype(x)
    if x < 0.5f0
        rho = map(RealT, 1.0f0)
        v1 = map(RealT, 0.0f0)
        p = map(RealT, 1.0f0)
    else
        rho = map(RealT, 0.125f0)
        v1 = map(RealT, 0.0f0)
        p = map(RealT, 0.1f0)
    end
    return prim2cons((rho, v1, p), equations)
end
surface_flux = flux_rusanov
semi = SemiDiscretizationHyperbolic(grid, equations, surface_flux, initial_condition_rp,
                                    backend_kernel = backend_kernel,
                                    boundary_conditions = BoundaryConditions(OutflowBC(),OutflowBC())
                                    );
tspan = map(RealT, (0.0, 0.1))
ode = ODE(semi, tspan)
Ccfl = map(RealT, 0.9)
save_time_interval = map(RealT, 0.0)
time_stepping = FixedTimeStepping(map(RealT, 1e-5))
param = Parameters(Ccfl, save_time_interval)

sol = solve(ode, param, time_stepping = time_stepping);

@show sol.l1, sol.l2

plot(grid.xc, sol.u[1, 1:nx], label = "Density", xlabel = "x", ylabel = "Density", title = "Riemann Problem at t=0.1")
