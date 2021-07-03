export @nview, scal!, axpy!, xpby!, axpby!

# @code_warntype ✓
function mach_acc()
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

function swap!(v::VecI, i::Int, j::Int) # @code_warntype ✓
    @inbounds begin
        temp = v[i]
        v[i] = v[j]
        v[j] = temp
    end
    return nothing
end

# @code_warntype ✓
function nrm2(x::VecI{Tx}, y::VecI{Ty}, b::VecB{Tb}) where {Tx<:Real,Ty<:Real,Tb<:Real}
    @simd for i in eachindex(b)
        @inbounds b[i] = abs2(x[i] - y[i])
    end
    return sqrt(sum(b))
end

# @code_warntype ✓
function scal!(a::Real, x::AbstractArray{Tx}) where Tx<:Real
    isone(a) && return nothing
    @inbounds begin
        if iszero(a)
            zeroTx = zero(Tx)
            @simd for i in eachindex(x)
                x[i] = zeroTx
            end
            return nothing
        end
        @simd for i in eachindex(x)
            x[i] *= a
        end
    end
end

# a * x + y → y, @code_warntype ✓
function axpy!(a::Real, x::AbstractVector{Tx}, y::AbstractVector{Ty}) where {Tx<:Real,Ty<:Real}
    iszero(a) && return nothing
    @inbounds begin
        if isone(a)
            @simd for i in eachindex(y)
                y[i] += x[i]
            end
            return nothing
        end
        @simd for i in eachindex(y)
            y[i] += a * x[i]
        end
    end
end

# x + b * y → y, @code_warntype ✓
function xpby!(x::AbstractVector{Tx}, b::Real, y::AbstractVector{Ty}) where {Tx<:Real,Ty<:Real}
    @inbounds begin
        if iszero(b)
            @simd for i in eachindex(y)
                y[i] = x[i]
            end
            return nothing
        end
        if isone(b)
            @simd for i in eachindex(y)
                y[i] += x[i]
            end
            return nothing
        end
        @simd for i in eachindex(y)
            y[i] = x[i] + b * y[i]
        end
    end
end

# a * x + b * y → y, @code_warntype ✓
function axpby!(a::Real, x::AbstractVector{Tx}, b::Real, y::AbstractVector{Ty}) where {Tx<:Real,Ty<:Real}
    @inbounds begin
        isone(b) && return axpy!(a, x, y)
        if iszero(b)
            iszero(a) && return scal!(0, y)
            if isone(a)
                @simd for i in eachindex(y)
                    y[i] = x[i]
                end
            else
                @simd for i in eachindex(y)
                    y[i] = a * x[i]
                end
            end
            return nothing
        end
        iszero(a) && return scal!(b, y)
        if isone(a)
            @simd for i in eachindex(y)
                y[i] = x[i] + b * y[i]
            end
        else
            @simd for i in eachindex(y)
                y[i] = a * x[i] + b * y[i]
            end
        end
    end
end
