using SmoothedDifferentiation
using Test

using Flux: Flux
using NNlib: relu
using Distributions: Normal, Uniform, Laplace
using Random: MersenneTwister
using XAIBase: Attribution

model = Flux.Chain(Flux.Dense(10 => 32, relu), Flux.Dense(32 => 5))
input = rand(Float32, 10, 4)

@testset "Optional arguments" begin
    distribution = Normal(0.0f0, 1.0f0)
    rng = MersenneTwister(123)

    @test_nowarn SmoothDiff(model, input)
    @test_nowarn SmoothDiff(model, input, 10)
    @test_nowarn SmoothDiff(model, input, 10, distribution)
    @test_nowarn SmoothDiff(model, input, 10, distribution, rng)
    @test_nowarn SmoothDiff(model, input, 10, distribution, rng, false)

    # Stored fields
    analyzer = SmoothDiff(model, input, 10, distribution, rng, false)
    @test analyzer.n == 10
    @test analyzer.distribution == distribution
    @test analyzer.rng === rng
    @test analyzer.show_progress == false
end

@testset "Default distribution" begin
    analyzer = SmoothDiff(model, input)
    @test analyzer.n == 50
    @test analyzer.distribution == Normal(0.0f0, 1.0f0)
end

@testset "`std` convenience constructor" begin
    # `std` is used as the standard deviation, i.e. `Normal(0, std)` (not `Normal(0, std^2)`).
    analyzer = SmoothDiff(model, input, 5, 0.1f0)
    @test analyzer.distribution == Normal(0.0f0, 0.1f0)
end

@testset "Arbitrary distributions" begin
    for distribution in (Normal(0.0f0, 0.5f0), Uniform(-0.1f0, 0.1f0), Laplace(0.0f0, 0.2f0))
        analyzer = SmoothDiff(model, input, 5, distribution, MersenneTwister(1), false)
        @test analyzer.distribution === distribution
        attr = analyze(input, analyzer)
        @test attr isa Attribution
        @test size(attr.val) == size(input)
    end
end

@testset "Reproducibility with seeded RNG" begin
    distribution = Uniform(-0.1f0, 0.1f0)
    a1 = SmoothDiff(model, input, 5, distribution, MersenneTwister(42), false)
    a2 = SmoothDiff(model, input, 5, distribution, MersenneTwister(42), false)
    @test analyze(input, a1).val == analyze(input, a2).val
end

@testset "Input validation" begin
    @test_throws ArgumentError SmoothDiff(model, input, 0)
    @test_throws ArgumentError SmoothDiff(model, input, -1, Normal(0.0f0, 1.0f0))
end
