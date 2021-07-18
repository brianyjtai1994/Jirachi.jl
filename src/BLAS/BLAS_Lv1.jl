export @nview, apy2, nrm2, swap!, scal!, axpy!, xpby!, axpby!

function mach_acc() # @code_warntype ✓
    sfmin = nextfloat(0.0)
    small = inv(prevfloat(Inf))
    if small ≥ sfmin
        # Use "small" plus 1 bit, to avoid the possibility of rounding
        # causing overflow when computing 1/sfmin.
        sfmin = small * (1.0 + eps())
    end
    return sfmin
end

const SAFMIN = mach_acc() / eps()

macro nview(arr::Symbol, N::Int)
    a = Vector{Expr}(undef, N)
    @inbounds for i in 1:N
        a[i] = Expr(:(=), Symbol("k", i), :(view($arr, :, $i)))
    end
    return Expr(:escape, Expr(:block, a...))
end

macro nview(arr::Symbol, vars::Symbol...)
    N = length(vars)
    a = Vector{Expr}(undef, N)
    @inbounds for i in 1:N
        a[i] = Expr(:(=), vars[i], :(view($arr, :, $i)))
    end
    return Expr(:escape, Expr(:block, a...))
end

function apy2(x::Real, y::Real) # @code_warntype ✓
    isnan(x) && return x
    isnan(y) && return y
    # general case
    xabs = abs(x)
    yabs = abs(y)
    w = max(xabs, yabs)
    z = min(xabs, yabs)
    iszero(z) && return w
    return w * sqrt(1.0 + abs2(z / w))
end

function swap!(v::VecI, i::Int, j::Int) # @code_warntype ✓
    @inbounds temp = v[i]
    @inbounds v[i] = v[j]
    @inbounds v[j] = temp
    return nothing
end

function swap!(v::MatI, i1::Int, j1::Int, i2::Int, j2::Int) # @code_warntype ✓
    @inbounds temp     = v[i1,j1]
    @inbounds v[i1,j1] = v[i2,j2]
    @inbounds v[i2,j2] = temp
    return nothing
end

function nrm2(x::VecI{Tx}, y::VecI{Ty}, b::VecB{Tb}) where {Tx<:Real,Ty<:Real,Tb<:Real} # @code_warntype ✓
    @simd for i in eachindex(b)
        @inbounds b[i] = abs2(x[i] - y[i])
    end
    return sqrt(sum(b))
end

function scal!(a::Real, x::VecO{Tx}) where Tx<:Real # @code_warntype ✓
    isone(a) && return nothing
    if iszero(a)
        zeroTx = zero(Tx)
        @simd for i in eachindex(x)
            @inbounds x[i] = zeroTx
        end
        return nothing
    end
    @simd for i in eachindex(x)
        @inbounds x[i] *= a
    end
end

# a * x + y → y, @code_warntype ✓
function axpy!(a::Real, x::VecI{Tx}, y::VecO{Ty}) where {Tx<:Real,Ty<:Real}
    iszero(a) && return nothing
    if isone(a)
        @simd for i in eachindex(y)
            @inbounds y[i] += x[i]
        end
        return nothing
    end
    @simd for i in eachindex(y)
        @inbounds y[i] += a * x[i]
    end
end

# x + b * y → y, @code_warntype ✓
function xpby!(x::VecI{Tx}, b::Real, y::VecO{Ty}) where {Tx<:Real,Ty<:Real}
    if iszero(b)
        @simd for i in eachindex(y)
            @inbounds y[i] = x[i]
        end
        return nothing
    end
    if isone(b)
        @simd for i in eachindex(y)
            @inbounds y[i] += x[i]
        end
        return nothing
    end
    @simd for i in eachindex(y)
        @inbounds y[i] = x[i] + b * y[i]
    end
end

# a * x + b * y → y, @code_warntype ✓
function axpby!(a::Real, x::VecI{Tx}, b::Real, y::VecO{Ty}) where {Tx<:Real,Ty<:Real}
    isone(b) && return axpy!(a, x, y)
    if iszero(b)
        iszero(a) && return scal!(0, y)
        if isone(a)
            @simd for i in eachindex(y)
                @inbounds y[i] = x[i]
            end
        else
            @simd for i in eachindex(y)
                @inbounds y[i] = a * x[i]
            end
        end
        return nothing
    end
    iszero(a) && return scal!(b, y)
    if isone(a)
        @simd for i in eachindex(y)
            @inbounds y[i] = x[i] + b * y[i]
        end
    else
        @simd for i in eachindex(y)
            @inbounds y[i] = a * x[i] + b * y[i]
        end
    end
end
