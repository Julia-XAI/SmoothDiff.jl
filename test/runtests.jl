using SmoothedDifferentiation
using Test

@testset "SmoothedDifferentiation.jl" begin
    @testset verbose = true "Linting" begin
        @info "Running linting tests..."
        include("linting.jl")
    end

    @testset "VEJP numerical tests" begin
        include("test_vejp.jl")
    end

    @testset "Flux" begin
        @testset "VGG preparation tests" begin
            include("test_preparation_flux.jl")
        end
        @testset "GPU tests" begin
            include("test_gpu_flux.jl")
        end
    end
end
