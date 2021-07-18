module Jirachi

const VecI = AbstractVector; # Input  Vector
const VecO = AbstractVector; # Output Vector
const VecB = AbstractVector; # Buffer Vector
const MatI = AbstractMatrix; # Input  Matrix
const MatO = AbstractMatrix; # Output Matrix
const MatB = AbstractMatrix; # Buffer Matrix

const 𝚷 = 2.0 * π

fcall(fn::Function, x::VecI) = fn(x)

include("./BLAS/BLAS_Lv1.jl")
include("./BLAS/BLAS_Lv2.jl")
include("./DiffEq/RungeKutta.jl")
include("./DiffEq/BulirschStoer.jl")
include("./interpolation.jl")
include("./fft.jl")
include("./stats.jl")
include("./sorting.jl")
include("./QLearning.jl")
include("./optimizer/minimizer.jl")

end # module
