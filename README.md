# SmoothDiff.jl

[![Code Style: Runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

Julia reference implementation of SmoothDiff for the NeurIPS 2025 paper *"Smoothed Differentiation Efficiently Mitigates Shattered Gradients in Explanations"*.

The full experiments can be found here: https://github.com/adrhill/smoothdiff-experiments/

## Installation
This package supports Julia ≥1.10.
It is not yet registered, so install it directly from GitHub by running the following in the Julia REPL:
```julia-repl
julia> ]add https://github.com/Julia-XAI/SmoothDiff.jl
```

## Example
SmoothDiff.jl exports the `SmoothDiff` analyzer,
which expects a differentiable Flux model whose non-linearities are `relu`.
Let's explain why a vision model classifies an image of a castle as such:

```julia
using SmoothedDifferentiation
using VisionHeatmaps         # visualization of attributions as heatmaps
using Flux, Metalhead        # pre-trained vision models in Flux
using DataAugmentation       # input preprocessing
using HTTP, FileIO, ImageIO  # load image from URL

# Load & prepare model
model = VGG(16, pretrain=true).layers

# Load input
url = HTTP.URI("https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg")
img = load(url)

# Preprocess input
mean = (0.485f0, 0.456f0, 0.406f0)
std  = (0.229f0, 0.224f0, 0.225f0)
tfm = CenterResizeCrop((224, 224)) |> ImageToTensor() |> Normalize(mean, std)
input = apply(tfm, Image(img))               # apply DataAugmentation transform
input = reshape(input.data, 224, 224, 3, :)  # unpack data and add batch dimension

# Run XAI method
analyzer = SmoothDiff(model, input)
attr = analyze(input, analyzer)      # or: attr = analyzer(input)
heatmap(attr)                        # show heatmap using VisionHeatmaps.jl
```

Note that `SmoothDiff` requires the `input` already at construction time,
as it prepares copies of the model's layers.

By default, attributions are computed for the class with the highest activation.
We can also compute attributions for a specific class, e.g. the one at output index 5:

```julia
analyze(input, analyzer, 5)  # for attribution
heatmap(input, analyzer, 5)  # for heatmap
```

## Acknowledgements

Adrian Hill gratefully acknowledges funding from the German Federal Ministry of Education and Research under the grant BIFOLD26B.
