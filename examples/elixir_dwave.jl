using KernelAbstractionsFV
using Trixi
using Metal
using KernelAbstractions

RealT = Float32
domain = map(RealT, (0.0, 1.0))
nx = 400
# backend_kernel = MetalBackend()
backend_kernel = CPU()

grid = make_grid(domain, nx, backend_kernel);
equations = CompressibleEulerEquations1D(1.4f0)

function initial_condition_dwave(x, t, equations::CompressibleEulerEquations1D)
    RealT = eltype(x)
    rho = map(RealT, 2.0f0 + 0.2f0*sin(2.0f0*pi*(x - 0.1f0*t)))
    v1 = map(RealT, 0.1f0)
    p = map(RealT, 1.0f0)
    return prim2cons((rho, v1, p), equations)
end
surface_flux = flux_rusanov
semi = SemiDiscretizationHyperbolic(grid, equations, surface_flux, initial_condition_dwave,
                                    backend_kernel = backend_kernel
                                    );
tspan = map(RealT, (0.0, 0.1))
ode = ODE(semi, tspan)
Ccfl = map(RealT, 0.9)
save_time_interval = map(RealT, 0.0)
param = Parameters(Ccfl, save_time_interval)

sol = solve(ode, param);

@show sol.l1, sol.l2
# if backend_kernel isa MetalBackend
#     @assert sol.l1 == 0.0004033974f0 sol.l1
# end
# if backend_kernel isa CPU
#     @assert sol.l1 == 0.00040343273f0 sol.l1
# end
