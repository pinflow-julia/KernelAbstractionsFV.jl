using KernelAbstractionsFV
using KernelAbstractionsFV: examples_dir
using TrixiBase
using Test
using Metal

@testset "Density wave GPU tests" begin
    tspan = (0.0f0, 0.001f0)
    nx = 10
    trixi_include(joinpath(examples_dir, "run_dwave.jl"), tspan = tspan, nx = nx,
                  backend_kernel = MetalBackend())
    @test isapprox(sol.l1, 0.0002595529f0, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.l2, 0.00025771547f0, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.linf, 0.00037491322f0, atol = 1e-9, rtol = 1e-9)

    trixi_include(joinpath(examples_dir, "run_metal_dwave.jl"), tspan = tspan, nx = nx)
    @test isapprox(sol.l1, 0.0002595529f0, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.l2, 0.00025771547f0, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.linf, 0.00037491322f0, atol = 1e-9, rtol = 1e-9)

    trixi_include(joinpath(examples_dir, "elixir_dwave.jl"), tspan = tspan, nx = nx,
                  backend_kernel = MetalBackend())
    @test isapprox(sol.l1, 0.0002595529f0, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.l2, 0.00025771547f0, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.linf, 0.00037491322f0, atol = 1e-9, rtol = 1e-9)

    trixi_include(joinpath(examples_dir, "run_cpu_dwave.jl"), tspan = tspan, nx = nx,
                  backend_kernel = MetalBackend())
    @test isapprox(sol.l1, 0.0002595529f0, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.l2, 0.00025771547f0, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.linf, 0.00037491322f0, atol = 1e-9, rtol = 1e-9)
end
