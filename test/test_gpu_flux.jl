using SmoothedDifferentiation
using Test

using Flux, NNlib
using Metal, JLArrays

if Metal.functional()
    @info "Using Metal as GPU device"
    device = mtl # use Apple Metal locally
else
    @info "Using JLArrays as GPU device"
    device = jl # use JLArrays to fake GPU array
end

model = Flux.Chain(Flux.Dense(10 => 32, relu), Flux.Dense(32 => 5))
input = rand(Float32, 10, 8)
@test_nowarn model(input)
analyzer = SmoothDiff(model, input, 5)
softplus_analyzer = SoftPlusTrick(model, 1.0f0)

model_gpu = device(model)
input_gpu = device(input)
@test_nowarn model_gpu(input_gpu)
analyzer_gpu = SmoothDiff(model_gpu, input_gpu, 5)
softplus_analyzer_gpu = SoftPlusTrick(model_gpu, 1.0f0)

@testset "Run analyzer (CPU)" begin
    attr = analyze(input, analyzer)
    @test attr isa Attribution
end

@testset "Run analyzer (GPU)" begin
    attr = analyze(input_gpu, analyzer_gpu)
    @test attr isa Attribution
end

@testset "Run softplus analyzer (CPU)" begin
    attr = analyze(input, softplus_analyzer)
    @test attr isa Attribution
end

@testset "Run softplus analyzer (GPU)" begin
    attr = analyze(input_gpu, softplus_analyzer_gpu)
    @test attr isa Attribution
end
