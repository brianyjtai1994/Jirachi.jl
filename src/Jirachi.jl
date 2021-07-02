module Jirachi

const VecI = AbstractVector; # Input  Vector
const VecO = AbstractVector; # Output Vector
const VecB = AbstractVector; # Buffer Vector
const MatI = AbstractMatrix; # Input  Matrix
const MatO = AbstractMatrix; # Output Matrix
const MatB = AbstractMatrix; # Buffer Matrix

include("./BLAS_Lv1.jl")
include("./BLAS_Lv2.jl")
include("./RungeKutta.jl")
include("./stats.jl")
include("./sorting.jl")
include("./QLearning.jl")
include("./optimizer.jl")

end # module
