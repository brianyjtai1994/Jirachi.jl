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
function gwo_move!(Xb::AbstractVector,
                   Xn::AbstractVector,
                   Xα::AbstractVector,
                   Xβ::AbstractVector,
                   Xδ::AbstractVector, a::Real) # @code_warntype ✓
    @nrand(3, 2)
    @inbounds for i in eachindex(Xb)
        Xni = Xn[i]
        Xαi = Xα[i]
        Xβi = Xβ[i]
        Xδi = Xδ[i]
        tmp = Xαi + Xβi + Xδi
        tmp = tmp - a * (2. * r11 - 1.) * abs(2. * r12 * Xαi - Xni)
        tmp = tmp - a * (2. * r21 - 1.) * abs(2. * r22 * Xβi - Xni)
        tmp = tmp - a * (2. * r31 - 1.) * abs(2. * r32 * Xδi - Xni)
        Xb[i] = tmp / 3.
    end
    return nothing
end
"""
    Sine-Cosine Optimizer

    params:
    -------
    * Xb := buffer
    * Xn := n-th solution
    * Xr := referred solution
"""
function sco_move!(Xb::AbstractVector,
                   Xn::AbstractVector,
                   Xr::AbstractVector, a::Real) # @code_warntype ✓
    r = rand() * 2.0
    @inbounds @simd for i in eachindex(Xb)
        Xb[i] = Xn[i] + a * abs(Xn[i] - Xr[i]) * ifelse(rand() < 0.5, sinpi(r), cospi(r))
    end
    return nothing
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
function asa_move!(Xb::AbstractVector,
                   Xn::AbstractVector, lb::NTuple{ND}, ub::NTuple{ND}, p1::Real, p2::Real) where ND # @code_warntype ✓
    @nrand(3)
    @inbounds begin
        if r1 < p1
            # exploitation:  addition / subtractioin
            r3 *= r2 < 0.5 ? 0.5 * p2 : -0.5 * p2
            @simd for i in 1:ND
                Xb[i] = Xn[i] + r3 * (ub[i] - lb[i])
            end
        else
            # exploration:       division     / multiplication
            r3 *= r2 < 0.5 ? 0.5 / (p2 + 1.0) : 0.5 * p2 
            @simd for i in 1:ND
                Xb[i] = Xn[i] * r3 * (ub[i] - lb[i])
            end
        end
    end
    return nothing
end
