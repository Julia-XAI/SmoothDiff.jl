module SmoothedDifferentiation

using Reexport
@reexport using XAIBase
import XAIBase: call_analyzer

using Base.Iterators
using Distributions: Sampleable, Normal
using Random: AbstractRNG, GLOBAL_RNG, rand!
using ProgressMeter: Progress, next!

using NNlib: relu, ∇maxpool, maxpool, upsample_nearest, σ
using Zygote: pullback
import ChainRulesCore: rrule, NoTangent, unthunk

using Flux: Flux
using Optimisers: Optimisers

include("prepare_model.jl")
include("vejp/relu.jl")
include("vejp/maxpool.jl")
include("softplus_trick.jl")

export SmoothDiff, SoftPlusTrick

## Helpers for Flux & Lux compatibility
mytestmode!(model) = model
mytestmode!(model::Flux.Chain) = Flux.testmode!(model)


samplingmode!(model, mode::Bool) = foreach(x -> samplingmode!(x, mode), Optimisers.trainable(model))

## Interface
const DEFAULT_SAMPLES = 50
const DEFAULT_DISTR = Normal(0.0f0, 1.0f0)

"""
    SmoothDiff(model, input)
    SmoothDiff(model, input, [n, std, rng, show_progress])
    SmoothDiff(model, input, [n, distribution, rng, show_progress])

Analyze `model` by computing a smoothed sensitivity map via *Smoothed Differentiation*.
Defaults to `n = $DEFAULT_SAMPLES` samples from the normal distribution with zero mean and `std = 1.0f0`.

The `input` is required at construction time to prepare (`ReLU`/`MaxPool`-accumulating) copies of the model's layers.

## Arguments
- `n::Int`: Number of noise samples. Defaults to `$DEFAULT_SAMPLES`.
- `std::Real` / `distribution::Sampleable`: Either the standard deviation of a zero-mean
  normal distribution, or an arbitrary scalar `distribution` to sample additive noise from.
  Defaults to `Normal(0.0f0, 1.0f0)`.
- `rng::AbstractRNG`: Random number generator used to sample noise from the `distribution`.
  Defaults to `GLOBAL_RNG`.
- `show_progress::Bool`: Show a progress meter while sampling. Defaults to `true`.
"""
struct SmoothDiff{M, D <: Sampleable, R <: AbstractRNG} <: AbstractXAIMethod
    model::M
    n::Int
    distribution::D
    rng::R
    show_progress::Bool

    function SmoothDiff(
            model,
            input,
            n::Int = DEFAULT_SAMPLES,
            distribution::D = DEFAULT_DISTR,
            rng::R = GLOBAL_RNG,
            show_progress = true,
        ) where {D <: Sampleable, R <: AbstractRNG}
        n < 1 && throw(ArgumentError("Number of samples `n` needs to be larger than zero."))
        prepared_model = prepare(model, input)
        mytestmode!(prepared_model)
        return new{typeof(prepared_model), D, R}(
            prepared_model, n, distribution, rng, show_progress
        )
    end
end

# Convenience constructor: sample from a zero-mean normal distribution with standard deviation `std`
function SmoothDiff(model, input, n::Int, std::Real, rng = GLOBAL_RNG, show_progress = true)
    T = eltype(input)
    distribution = Normal(zero(T), convert(T, std))
    return SmoothDiff(model, input, n, distribution, rng, show_progress)
end

function call_analyzer(
        input, method::SmoothDiff, output_selector::AbstractOutputSelector; kwargs...
    )
    output, vejp_fn = prepare_vejp(input, method::SmoothDiff)
    output_selection = output_selector(output)

    # Evaluate VeJP
    v = zero(output)
    v[output_selection] .= 1
    val = only(vejp_fn(v))

    return Attribution(val, input, output, output_selection, NormPooling())
end

function prepare_vejp(input, method::SmoothDiff)
    model = method.model
    noisy_input = similar(input)
    reset_counts!(model)
    samplingmode!(model, true)

    p = Progress(method.n; desc = "Sampling SmoothDiff...", showspeed = method.show_progress)
    for _ in 1:(method.n - 1)
        noisy_input = rand!(method.rng, method.distribution, noisy_input)
        noisy_input .+= input
        model(noisy_input) # update counts by running inference
        next!(p)
    end

    # On last step, create VeJP function
    noisy_input = rand!(method.rng, method.distribution, noisy_input)
    noisy_input .+= input
    output, vejp_fn = pullback(model, noisy_input) # create expected Jacobian operator (aka "VeJP function")
    samplingmode!(model, false)
    next!(p)
    return output, vejp_fn
end

end # module
