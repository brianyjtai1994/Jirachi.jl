export TXN, TXT, gemv!

# Transpose Operators
const TXN = Val('N')
const TXT = Val('T')

# a * A * x + by → y, @code_warntype ✓
function gemv!(::Val{'N'}, a::Real, A::MatI{TA}, x::VecI{Tx}, b::Real, y::VecO{Ty}) where {TA<:Real,Tx<:Real,Ty<:Real}
    scal!(b, y)                 # y := b * y
    iszero(a) && return nothing # quick return if possible
    #### y := a * A * x + y
    if isone(a)
        for j in eachindex(x)
            @inbounds xj = x[j]
            if !iszero(xj)
                @simd for i in eachindex(y)
                    @inbounds y[i] += A[i, j] * xj
                end
            end
        end
        return nothing
    end
    for j in eachindex(x)
        @inbounds axj = a * x[j]
        if !iszero(axj)
            @simd for i in eachindex(y)
                @inbounds y[i] += A[i, j] * axj
            end
        end
    end
end
# @code_warntype ✓
function gemv!(::Val{'N'}, a::Real, A::MatI{TA}, x::NTuple{N,Tx}, b::Real, y::VecO{Ty}) where {N,TA<:Real,Tx<:Real,Ty<:Real}
    scal!(b, y)                 # y := b * y
    iszero(a) && return nothing # quick return if possible
    #### y := a * A * x + y
    if isone(a)
        for j in eachindex(1:N)
            @inbounds xj = x[j]
            if !iszero(xj)
                @simd for i in eachindex(y)
                    @inbounds y[i] += A[i, j] * xj
                end
            end
        end
        return nothing
    end
    for j in eachindex(1:N)
        @inbounds axj = a * x[j]
        if !iszero(axj)
            @simd for i in eachindex(y)
                @inbounds y[i] += A[i, j] * axj
            end
        end
    end
end
