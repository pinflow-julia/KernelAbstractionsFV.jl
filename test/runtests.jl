using KernelAbstractionsFV
using KernelAbstractionsFV: examples_dir
using TrixiBase
using Test

@testset "Density wave CPU tests" begin
    tspan = (0.0f0, 0.001f0)
    nx = 10
    trixi_include(joinpath(examples_dir, "run_dwave.jl"))
    @test isapprox(sol.l1, 0.0006434416f0, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.l2, 0.0006504935f0, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.linf, 0.000957489f0, atol = 1e-9, rtol = 1e-9)
end
