mutable struct Agent
    x::Vector{Float64}; f::Float64; v::Bool; c::Float64
    # x := parameters passed into models
    # f := function-value of fn!(x)
    # v := viability / feasibility
    # c := contravention / violation

    Agent(lb::NTuple, ub::NTuple) = new(born(lb, ub), Inf, false, 0.0) # @code_warntype ✓
end

function Base.isequal(a1::Agent, a2::Agent) # @code_warntype ✓
    # a1, a2 are both feasible
    a1.v && a2.v && return a1.f == a2.f
    # a1, a2 are both infesasible
    !a1.v && !a2.v && return a1.c == a2.c
    return false
end

function Base.isless(a1::Agent, a2::Agent) # @code_warntype ✓
    # a1, a2 are both feasible
    a1.v && a2.v && return a1.f < a2.f
    # a1, a2 are both infesasible
    !a1.v && !a2.v && return a1.c < a2.c
    # if (a1, a2) = (feasible, infeasible), then a1 < a2 is true
    # if (a1, a2) = (infeasible, feasible), then a2 < a1 is false
    return a1.v
end

#### random initialization, @code_warntype ✓
function born(lb::NTuple{ND}, ub::NTuple{ND}) where ND
    if @generated
        a = Vector{Expr}(undef, ND)
        @inbounds for i in eachindex(a)
            a[i] = :(lb[$i] + rand() * (ub[$i] - lb[$i]))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:vect, a...))
        end
    else
        x = Vector{Float64}(undef, ND)
        @simd for i in eachindex(x)
            @inbounds x[i] = lb[i] + rand() * (ub[i] - lb[i])
        end
        return x
    end
end

#### groups, @code_warntype ✓
function return_agents(lb::NTuple, ub::NTuple, NP::Int)
    agents = Vector{Agent}(undef, NP)
    @inbounds for i in eachindex(agents)
        agents[i] = Agent(lb, ub)
    end
    return agents
end

#### subgroups, @code_warntype ✓
return_elites(agents::VecI{Agent}, NR::Int)          = view(agents, 1:NR)
return_throng(agents::VecI{Agent}, NR::Int, NP::Int) = view(agents, NR+1:NP)
