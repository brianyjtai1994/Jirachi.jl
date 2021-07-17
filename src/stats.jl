#### = = = Benchmark Standards = = = = ####
# X = [range(-100., 100., length=30001);] #
# Y = similar(X)                          #
#### - - - - - - - - - - - - - - - - - ####

#### S-shaped functions ####
"""
    Generic logistic function
"""
logistic(x::Real, x0::Real, a::Real, k::Real, c::Real) = a / (1.0 + exp(k * (x0 - x))) + c
"""
Sigmoid S-shaped function.

```julia
>>> @btime sigmoid(\$Y, \$X)
158.625 μs (0 allocations: 0 bytes)
```
"""
sigmoid(x::Real) = inv(1.0 + exp(-x))
function sigmoid!(y::AbstractVector{S}, x::AbstractVector{T}) where {S<:Real,T<:Real}
    @inbounds @simd for i in eachindex(y)
        y[i] = sigmoid(x[i])
    end
    return nothing
end
"""
Algebraic S-shaped function.

```julia
>>> @btime algebraic(\$Y, \$X, 3.0)
30.127 μs (0 allocations: 0 bytes)
```
"""
algebraic(x::Real, a::Real) = x / (1.0 + abs(x / a))
function algebraic!(y::AbstractVector{S}, x::AbstractVector{T}, a::Real) where {S<:Real,T<:Real}
    @inbounds @simd for i in eachindex(y)
        y[i] = algebraic(x[i], a)
    end
    return nothing
end

#### Sampling functions ####
"""
    Duplicate a series of random numbers.

    Example #1:
    ```julia
    >>> @macroexpand @nrand(3)
    quote
        r1 = rand()
        r2 = rand()
        r3 = rand()
    end
    ```

    Example #2:
    ```julia
    >>> @macroexpand @nrand(2, 2)
    quote
        r11 = rand()
        r21 = rand()
        r12 = rand()
        r22 = rand()
    end
    ```
"""
macro nrand(N::Int, scale::Real=1.0)
    c = isone(scale) ? :(rand()) : :($scale * rand())
    a = Vector{Expr}(undef, N)
    @inbounds for i in 1:N
        a[i] = Expr(:(=), Symbol("r", i), c)
    end
    return Expr(:escape, Expr(:block, a...))
end
macro nrand(M::Int, N::Int, scale::Real=1.0)
    c = isone(scale) ? :(rand()) : :($scale * rand())
    a = Vector{Expr}(undef, M * N)
    @inbounds for j in 1:N, i in 1:M
        a[M*(j-1) + i] = Expr(:(=), Symbol("r", i, j), c)
    end
    return Expr(:escape, Expr(:block, a...))
end
"""
    Exclusive sampling from a given collection.

    params:
    -------
    * collection := the collection to sample from
    * except     := the one to be excluded
"""
function sample(collection, except)
    ret = rand(collection)
    while ret === except
        ret = rand(collection)
    end
    return ret
end
"""
    Perform a single step of Wolford algorithm
"""
function welford_step(μ::Real, s::Real, v::Real, c::Real)
    isone(c) && return v, zero(v)
    s = s * (c - 1)
    m = μ + (v - μ) / c
    s = s + (v - μ) * (v - m)
    μ = m
    return μ, s / (c - 1)
end
