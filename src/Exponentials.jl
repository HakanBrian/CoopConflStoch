module Exponentials

export Exponential

struct Exponential
    base::Float64  # The exponent value (e.g., 342 for e^342)
end

Base.:-(a::Exponential, b::Exponential) = Exponential(
    clamp(a.base - b.base, -floatmax(Float64), floatmax(Float64)),  # Prevent overflow (Subtraction)
)

end # module Exponential