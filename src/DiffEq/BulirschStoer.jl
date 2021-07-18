export StepperBS

# @code_warntype ✓
function stepsize_tup(H::Real, ns::NTuple{KMAX,Int}) where KMAX
    if @generated
        a = Vector{Expr}(undef, KMAX)
        @inbounds for k in 1:KMAX
            a[k] = :(H / ns[$k])
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, a...))
        end
    else
        return ntuple(k -> H / ns[k], KMAX)
    end
end

# @code_warntype ✓
function stepinv2_tup(ns::NTuple{KMAX,Int}) where KMAX
    if @generated
        a = Vector{Expr}(undef, KMAX)
        @inbounds for k in 1:KMAX
            a[k] = :(4 / abs2(ns[$k]))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, a...))
        end
    else
        return ntuple(k -> 4 / abs2(ns[k]), KMAX)
    end
end

# @code_warntype ✓
function multistepBS!(ηt::VecI, t0::Real, hk::Real, y0::VecI, diff!::Function, nk::Int, ηm::VecB, ηn::VecB; fargs...)
    h2 = 2.0 * hk
    copy!(ηm, y0)
    diff!(ηn, t0, ηm; fargs...)
    xpby!(ηm, hk, ηn)
    one2n = eachindex(ηt)

    tj = t0 + hk; j = 1
    while j < nk
        diff!(ηt, tj, ηn; fargs...)
        xpby!(ηm, h2, ηt)
        @simd for i in one2n
            @inbounds ηm[i], ηn[i] = ηn[i], ηt[i]
        end
        tj += hk; j += 1
    end

    diff!(ηt, tj, ηn; fargs...)
    @simd for i in one2n
        @inbounds ηt[i] = 0.5 * (ηn[i] + ηm[i] + hk * ηt[i])
    end
end

# @code_warntype ✓
function extrap2zero!(Ts::MatO{T}, xs::NTuple, ηt::VecI) where T
    zeroT = zero(T)
    @inbounds for i in eachindex(ηt)
        Ts[i,1], Ts[i,2] = ηt[i], zeroT
    end
end
# @code_warntype ✓
function extrap2zero!(Ts::MatO{T}, xs::NTuple, ηt::VecI, k::Int) where T
    isone(k) && return extrap2zero!(Ts, xs, ηt)
    zeroT = zero(T)
    one2n = eachindex(ηt)
    for j in 1:k-1
        @simd for i in one2n
            @inbounds Ts[i,j] -= ηt[i]
        end
    end
    @simd for i in one2n
        @inbounds Ts[i,k] = zeroT
    end
    for j in reverse(1:k-1)
        @inbounds xratio = xs[k] / xs[j] - 1.0
        @inbounds for i in one2n
            Ts[i,j] += (Ts[i,j] - Ts[i,j+1]) / xratio
        end
    end
    for j in eachindex(1:k)
        @simd for i in one2n
            @inbounds Ts[i,j] += ηt[i]
        end
    end
end

# @code_warntype ✓
function estimate_err(Tsol::VecI, Tref::VecI, y0::VecI, atol::Real, rtol::Real)
    e = 0.0 # error
    @inbounds for i in eachindex(y0)
        s  = atol + rtol * max(abs(y0[i]), abs(Tref[i])) # scale
        e += abs2((Tsol[i] - Tref[i]) / s)
    end
    return sqrt(e / length(y0))
end

struct StepperBS{KMAX}
    ns::NTuple{KMAX,Int}
    hs::NTuple{KMAX,Float64}
    xs::NTuple{KMAX,Float64}
    ηt::Vector{Float64}
    ηm::Vector{Float64}
    ηn::Vector{Float64}
    Ts::Matrix{Float64}

    StepperBS{13}(H::Real, ydim::Int) = StepperBS(H, ydim, (2,  6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50))
    StepperBS{ 9}(H::Real, ydim::Int) = StepperBS(H, ydim, (2,  8, 14, 20, 26, 32, 38, 44, 50))
    StepperBS{ 7}(H::Real, ydim::Int) = StepperBS(H, ydim, (2, 10, 18, 26, 34, 42, 50))

    function StepperBS(H::Real, ydim::Int, ns::NTuple{KMAX,Int}) where KMAX
        hs = stepsize_tup(H, ns)
        xs = stepinv2_tup(ns)
        ηt = Vector{Float64}(undef, ydim)
        ηm = Vector{Float64}(undef, ydim)
        ηn = Vector{Float64}(undef, ydim)
        Ts = Matrix{Float64}(undef, ydim, KMAX)
        return new{KMAX}(ns, hs, xs, ηt, ηm, ηn, Ts)
    end
end

# @code_warntype ✓
function (stbs::StepperBS{KMAX})(ys::VecO{T}, t0::Real, y0::VecI, f!::Function; disp::Bool=false, fargs...) where {KMAX,T<:Real}
    ns = stbs.ns
    hs = stbs.hs
    xs = stbs.xs
    ηt = stbs.ηt
    ηm = stbs.ηm
    ηn = stbs.ηn
    Ts = stbs.Ts

    atol = 1.0e-12
    rtol = 1.0e-12
    preE = Inf # previous error

    @nview Ts Tsol Tref # Ts[solution], Ts[reference]
    @inbounds for k in 1:KMAX
        multistepBS!(ηt, t0, hs[k], y0, f!, ns[k], ηm, ηn; fargs...)
        extrap2zero!(Ts, xs, ηt, k)

        nowE = estimate_err(Tsol, Tref, y0, atol, rtol)
        if nowE > preE || iszero(nowE)
            disp && println("converged error @ k = $k: $nowE")
            break
        end
        preE = nowE
    end
    @simd for i in eachindex(ys)
        @inbounds ys[i] = Tsol[i]
    end
end
