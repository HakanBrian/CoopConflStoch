module Exponentials

export Exponential, normalize_exponentials

struct Exponential
    base::Float64  # The exponent value (e.g., 342 for e^342)
end

Base.:-(a::Exponential, b::Exponential) = Exponential(
    clamp(a.base - b.base, -floatmax(Float64), floatmax(Float64)),  # Prevent overflow (Subtraction)
)

function normalize_exponentials(values::Vector{Exponentials.Exponential})
    max_base = maximum(v -> v.base, values)
    sum_probs = 0.0
    probs = similar(values, Float64)  # Pre-allocate for probabilities

    # Compute normalized probabilities
    for (i, v) in pairs(values)
        prob = exp(v.base - max_base)
        probs[i] = prob
        sum_probs += prob
    end

    return probs ./ sum_probs
end

end # module Exponential
