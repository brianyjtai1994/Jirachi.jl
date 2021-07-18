"""
    Grey-Wolf Optimizer

    params:
    -------
    * Xb := buffer
    * Xn := n-th solution
    * Xα := α-wolf solution
    * Xβ := β-wolf solution
    * Xδ := δ-wolf solution
"""
function gwo_move!(Xb::VecB, Xn::VecI, Xα::VecI, Xβ::VecI, Xδ::VecI, a::Real) # @code_warntype ✓
    @nrand 3 2 2.0
    @inbounds for i in eachindex(Xb)
        Xni = Xn[i]
        Xαi = Xα[i]
        Xβi = Xβ[i]
        Xδi = Xδ[i]
        tmp = Xαi + Xβi + Xδi
        tmp = tmp - a * (r11 - 1.0) * abs(r12 * Xαi - Xni)
        tmp = tmp - a * (r21 - 1.0) * abs(r22 * Xβi - Xni)
        tmp = tmp - a * (r31 - 1.0) * abs(r32 * Xδi - Xni)
        Xb[i] = tmp / 3.0
    end
end
"""
    Sine-Cosine Optimizer

    params:
    -------
    * Xb := buffer
    * Xn := n-th solution
    * Xr := referred solution
"""
function sco_move!(Xb::VecB, Xn::VecI, Xr::VecI, a::Real) # @code_warntype ✓
    r = 2.0 * rand()
    @inbounds for i in eachindex(Xb)
        Xb[i] = Xn[i] + a * abs(Xn[i] - Xr[i]) * ifelse(rand() < 0.5, sinpi(r), cospi(r))
    end
end
"""
    Arithmetic Search Algorithm Optimizer

    params:
    -------
    * Xb := buffer
    * Xn := n-th solution
    * lb := lower bounds
    * ub := upper bounds
    * p1 := tuning probability, p1 = 0.2 + 0.7 * (it / max. it)
    * p2 := tuning probability, p2 = 1 - exp(log(it / max. it) / 5)
"""
function asa_move!(Xb::VecB, Xn::VecI, lb::NTuple, ub::NTuple, p1::Real, p2::Real) # @code_warntype ✓
    @nrand 3
    if r1 < p1
        # exploitation:  addition / subtractioin
        r3 *= r2 < 0.5 ? 0.5 * p2 : -0.5 * p2
        @simd for i in eachindex(Xb)
            @inbounds Xb[i] = Xn[i] + r3 * (ub[i] - lb[i])
        end
    else
        # exploration:       division     / multiplication
        r3 *= r2 < 0.5 ? 0.5 / (p2 + 1.0) : 0.5 * p2 
        @simd for i in eachindex(Xb)
            @inbounds Xb[i] = Xn[i] * r3 * (ub[i] - lb[i])
        end
    end
end
"""
    Water-Cycle Algorithm Optimizer

    params:
    -------
    * Xb    := buffer
    * Xbest := the best solution currently
    * lb    := lower bounds
    * ub    := upper bounds
"""
function wca_move!(Xb::VecB, Xbest::VecI) # @code_warntype ✓
    r = randn()
    @simd for i in eachindex(Xb)
        @inbounds Xb[i] = Xbest[i] + r * 0.31622776601683794 # sqrt(0.1)
    end
end
