# 1-D Finite Volume code for conservation laws using KernelAbstractions.jl for GPU support

The code is written in Julia. To obtain Julia, go to https://julialang.org/downloads/ and download the long-term support (LTS) release v1.10. We recommend to use juliaup as described on the website. In order to run the code, clone the repository, and the enter the `julia` REPL in this directory by
```shell
julia --project=.
```
or by starting plain `julia` REPL and then entering `import Pkg; Pkg.activate(".")`. Install all dependencies (only needed the first time) with
```julia
julia> import Pkg; Pkg.instantiate()
```

For the first time, to precompile parts of code to local drive, it is also recommended that you run

```julia
julia> using KernelAbstractionsFV
```

Now, you can run an example on CPU by using
```julia
julia> include("examples/run_dwave.jl")
```
If you have an AMD GPU, assuming you have the AMD drivers installed (TODO - Add instructions for that?), you can do `Pkg.add("AMDGPU")` and then run `examples/amd_dwave.jl` similarly. Similar to `amd_dwave.jl`, you can run other types of GPU. For example, if you have Apple M series GPU, you can do `using Metal` and modify the `SemiDiscretizationHyperbolic` object by passing a `backend_kernel` keyword argument as
```
semi = SemiDiscretizationHyperbolic(grid, equations, surface_flux,
                                    initial_condition_dwave,
                                    backend_kernel = MetalBackend())
```
By default, all examples use `Float32` arithmetic. This is essential for some GPUs as they only support `Float32` arithmetic. This includes the M series GPU.

## Known issues

- OffsetArrays are crucially used in the code, but are only partially supported in GPUs. Even `show(arr)` for `arr` being an OffsetArray gives an error. That is why we use a semicolon in the example files whenever a container having OffsetArrays is returned, like the definition of the `grid`, `semi`.
- The `grid` object isn't allowed in any GPU kernels because it is said to have something which can't live on the GPUs. This is likely because `grid` is a `struct`, and requires `Adapt.jl` to be converted to a GPU supported format; see https://discourse.julialang.org/t/how-to-properly-pass-structs-into-gpu-mwe-included/93727.
