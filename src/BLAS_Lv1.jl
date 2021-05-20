# @code_warntype ✓
function swap!(v::AbstractVector, i::Int, j::Int)
    v[i], v[j] = v[j], v[i]
    return nothing # to avoid a tuple-allocation
end
