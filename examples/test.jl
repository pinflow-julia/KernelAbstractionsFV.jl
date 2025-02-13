using KernelAbstractionsFV

grid = make_grid((0.0, 1.0), 100)
equations = Euler1D(1.4)

function initial_condition_dwave(x, t, equations::Euler1D)
    rho = 2.0 + 0.2*sin(2*pi*x)
    v1 = 0.1
    p = 1.0
    return prim2cons((rho, v1, p), equations)
end

semi = SemiDiscretizationHyperbolic(grid, equations, initial_condition_dwave)