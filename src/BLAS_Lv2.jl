export TXN, TXT, gemv!

# Transpose Operators
const TXN = Val('N')
const TXT = Val('T')

# a * A * x + by → y, @code_warntype ✓
function gemv!(::Val{'N'}, a::Real, A::AbstractMatrix{TA}, x::AbstractVector{Tx}, b::Real, y::AbstractVector{Ty}) where {TA<:Real,Tx<:Real,Ty<:Real}
    scal!(b, y)                                 # y := b * y
    iszero(a) && return nothing                 # quick return if possible
    @inbounds begin                             # y := a * A * x + y
        if isone(a)
            for j in eachindex(x)
                xj = x[j]
                if !iszero(xj)
                    @simd for i in eachindex(y)
                        y[i] += A[i, j] * xj
                    end
                end
            end
            return nothing
        end
        for j in eachindex(x)
            axj = a * x[j]
            if !iszero(axj)
                @simd for i in eachindex(y)
                    y[i] += A[i, j] * axj
                end
            end
        end
    end
end
# @code_warntype ✓
function gemv!(::Val{'N'}, a::Real, A::AbstractMatrix{TA}, x::NTuple{N,Tx}, b::Real, y::AbstractVector{Ty}) where {N,TA<:Real,Tx<:Real,Ty<:Real}
    scal!(b, y)                                 # y := b * y
    iszero(a) && return nothing                 # quick return if possible
    @inbounds begin                             # y := a * A * x + y
        if isone(a)
            for j in 1:N
                xj = x[j]
                if !iszero(xj)
                    @simd for i in eachindex(y)
                        y[i] += A[i, j] * xj
                    end
                end
            end
            return nothing
        end
        for j in 1:N
            axj = a * x[j]
            if !iszero(axj)
                @simd for i in eachindex(y)
                    y[i] += A[i, j] * axj
                end
            end
        end
    end
end
