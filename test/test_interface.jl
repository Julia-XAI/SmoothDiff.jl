using SmoothedDifferentiation
using Test

using Flux
using Distributions: Normal
using Random: MersenneTwister

# Small conv model so the prepared model exercises both the `ReluAccumulator` and the
# `MaxPoolAccumulator` paths.
model = Chain(
    Conv((3, 3), 3 => 4, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(36 => 5, relu),
    Dense(5 => 3),
)
input = rand(Float32, 8, 8, 3, 2)

smoothdiff = SmoothDiff(model, input, 5, Normal(0.0f0, 1.0f0), MersenneTwister(1), false)
softplus = SoftPlusTrick(model, 1.0f0)

@testset "SmoothDiff interface" begin
    @test XAIBase.test_interface(smoothdiff, input)
    @test XAIBase.test_interface(smoothdiff, input; output_selection = 2)
    @test analyze(input, smoothdiff).pooling isa NormPooling
end

@testset "SoftPlusTrick interface" begin
    @test XAIBase.test_interface(softplus, input)
    @test XAIBase.test_interface(softplus, input; output_selection = 2)
    @test analyze(input, softplus).pooling isa NormPooling
end
