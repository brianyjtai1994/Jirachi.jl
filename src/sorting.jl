bilast_insert(arr::AbstractVector, val::T) where T = bilast_insert(arr, val, 1, length(arr)) # @code_warntype ✓
# @code_warntype ✓
function bilast_insert(arr::AbstractVector, val::T, ldx::Int, rdx::Int) where T
    ldx ≥ rdx && return ldx
    ub = rdx # upper bound
    @inbounds begin
        while ldx < rdx
            mdx = (ldx + rdx) >> 1 # midpoint (binary search)
            val < arr[mdx] ? rdx = mdx : ldx = mdx + 1 # arr[mdx].f == val in this case
        end
        if ldx == ub && arr[ldx] ≤ val
            ldx += 1
        end
    end
    return ldx
end

binary_insertsort!(arr::AbstractVector) = binary_insertsort!(arr, 1, length(arr)) # @code_warntype ✓
# @code_warntype ✓
function binary_insertsort!(arr::AbstractVector, ldx::Int, rdx::Int)
    @inbounds for idx in ldx+1:rdx
        val = arr[idx]
        jdx = idx
        loc = bilast_insert(arr, val, ldx, idx)
        while jdx > loc
            swap!(arr, jdx, jdx - 1)
            jdx -= 1
        end
    end
    return nothing
end
