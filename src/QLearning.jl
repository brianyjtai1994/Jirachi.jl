"""
    Q-table is arranged as (Actions × States)

              s1  s2 ⋅ ⋅ ⋅
        a1 ┌ q11 q12 ⋅ ⋅ ⋅ ┐
        a2 │ q21 q22 ⋅ ⋅ ⋅ │
         ⋅ │  ⋅   ⋅  ⋅     │
         ⋅ │  ⋅   ⋅    ⋅   │
         ⋅ └  ·   ⋅      ⋅ ┘

    , which is updated by
        nextA = argmax Q(A, nextS) ∀ A
    for
        Q(prevA, prevS) += α * [prevR + γ * Q(nextA, nextS) - Q(prevA, prevS)]
"""
function update_Qtable!(Q::MatI, prevS::Int, prevA::Int, prevR::Real, nextS::Int, α::Real, γ::Real) # @code_warntype ✓
    nextA = pick_action(Q, nextS)
    @inbounds Q[prevA, prevS] *= 1.0 - α
    @inbounds Q[prevA, prevS] += α * (prevR + γ * Q[nextA, nextS])
    return nothing
end
"""
    The Q-learning agent picks an action of maximum Q-value of current state.

    params:
    -------
    * Q   := Q-table ∈ (Actions × States)
    * Sdx := index of current state
    * Adx := index of picked action (return)
"""
function pick_action(Q::MatI, Sdx::Int) # @code_warntype ✓
    Adx = 0
    tmp = -Inf
    @inbounds for idx in axes(Q, 1)
        _tmp = Q[idx, Sdx]
        if _tmp ≠ _tmp || tmp < _tmp
            tmp = _tmp
            Adx = idx
        end
    end
    return Adx
end
"""
    The Q-learning agent picks an action based on ε-greedy strategy.

    params:
    -------
    * Q   := Q-table ∈ (States × Actions)
    * Sdx := index of current state
    * Adx := index of picked action (return)
    * ε   := greedy factor for exploration, ε ∈ (0, 1], ε(t = 0) = 1
"""
function pick_action(Q::MatI, Sdx::Int, ε::Real) # @code_warntype ✓
    Adx = pick_action(Q, Sdx)
    rand() < ε ? sample(axes(Q, 1), Adx) : Adx
end
