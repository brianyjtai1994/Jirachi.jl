export CERK, cerk!, IERK, ierk!

abstract type AbstractRK{T} end

#### Classic Explicit Runge-Kutta (CERK) ####
struct CERK <: AbstractRK{Float64}
    c2::Float64
    c3::Float64
    c4::Float64
    a2::Float64
    a3::Float64
    a4::Float64
    bv::NTuple{4,Float64}
    
    CERK(Δt::Real) = new(0.5*Δt, 0.5*Δt, Δt, 0.5*Δt, 0.5*Δt, Δt, (Δt/6.0, Δt/3.0, Δt/3.0, Δt/6.0))
end

function cerk!(ks::AbstractMatrix{Tk}, up::AbstractVector{Tu}, un::AbstractVector{Tu}, tn::Real, diff!::Function, cerk::CERK; diff_args...) where {Tk<:Real,Tu<:Real}
    @inbounds begin
        #### initialize ####
        @nview ks 4
        #### generate k1 ####
        copy!(un, up)
        diff!(k1, un, tn; diff_args...)
        #### generate k2 ####
        @simd for i in eachindex(un)
            un[i] = up[i] + cerk.a2 * k1[i]
        end
        diff!(k2, un, tn + cerk.c2; diff_args...)
        #### generate k3 ####
        @simd for i in eachindex(un)
            un[i] = up[i] + cerk.a3 * k2[i]
        end
        diff!(k3, un, tn + cerk.c3; diff_args...)
        #### generate k4 ####
        @simd for i in eachindex(un)
            un[i] = up[i] + cerk.a4 * k3[i]
        end
        diff!(k4, un, tn + cerk.c4; diff_args...)
        #### update un ####
        copy!(un, up)
        gemv!(TXN, 1, ks, cerk.bv, 1, un)
    end
end

function cerk!(ks::AbstractMatrix{Tk}, up::AbstractVector{Tu}, un::AbstractVector{Tu}, diff!::Function, cerk::CERK; diff_args...) where {Tk<:Real,Tu<:Real}
    @inbounds begin
        #### initialize ####
        @nview ks 4
        #### generate k1 ####
        copy!(un, up)
        diff!(k1, un; diff_args...)
        #### generate k2 ####
        @simd for i in eachindex(un)
            un[i] = up[i] + cerk.a2 * k1[i]
        end
        diff!(k2, un; diff_args...)
        #### generate k3 ####
        @simd for i in eachindex(un)
            un[i] = up[i] + cerk.a3 * k2[i]
        end
        diff!(k3, un; diff_args...)
        #### generate k4 ####
        @simd for i in eachindex(un)
            un[i] = up[i] + cerk.a4 * k3[i]
        end
        diff!(k4, un; diff_args...)
        #### update un ####
        copy!(un, up)
        gemv!(TXN, 1, ks, cerk.bv, 1, un)
    end
end

#### Improved Explicit Runge-Kutta (IERK) ####
struct IERK <: AbstractRK{Float64}
    c2::Float64
    c3::Float64
    c4::Float64
    a2::Float64
    a3::NTuple{2,Float64}
    a4::NTuple{3,Float64}
    bv::NTuple{4,Float64}

    IERK(Δt::Real) = new(
        0.4*Δt, 0.45573725*Δt, Δt, 0.4*Δt, (0.29697761*Δt,  0.15875964*Δt),
        (0.21810040*Δt, -3.05096516*Δt, 3.83286476*Δt), (0.17476028*Δt, -0.55148066*Δt, 1.20553560*Δt, 0.17118478*Δt)
    )
end

function ierk!(ks::AbstractMatrix{Tk}, up::AbstractVector{Tu}, un::AbstractVector{Tu}, tn::Real, diff!::Function, ierk::IERK; diff_args...) where {Tk<:Real,Tu<:Real}
    #### initialize ####
    @nview ks 4
    #### generate k1 ####
    copy!(un, up)
    diff!(k1, un, tn; diff_args...)
    #### generate k2 ####
    @inbounds @simd for i in eachindex(un)
        un[i] = up[i] + ierk.a2 * k1[i]
    end
    diff!(k2, un, tn + ierk.c2; diff_args...)
    #### generate k3 ####
    copy!(un, up)
    gemv!(TXN, 1, ks, ierk.a3, 1, un)
    diff!(k3, un, tn + ierk.c3; diff_args...)
    #### generate k4 ####
    copy!(un, up)
    gemv!(TXN, 1, ks, ierk.a4, 1, un)
    diff!(k4, un, tn + ierk.c4; diff_args...)
    #### update un ####
    copy!(un, up)
    gemv!(TXN, 1, ks, ierk.bv, 1, un)
end

function ierk!(ks::AbstractMatrix{Tk}, up::AbstractVector{Tu}, un::AbstractVector{Tu}, diff!::Function, ierk::IERK; diff_args...) where {Tk<:Real,Tu<:Real}
    #### initialize ####
    @nview ks 4
    #### generate k1 ####
    copy!(un, up)
    diff!(k1, un; diff_args...)
    #### generate k2 ####
    @inbounds @simd for i in eachindex(un)
        un[i] = up[i] + ierk.a2 * k1[i]
    end
    diff!(k2, un; diff_args...)
    #### generate k3 ####
    copy!(un, up)
    gemv!(TXN, 1, ks, ierk.a3, 1, un)
    diff!(k3, un; diff_args...)
    #### generate k4 ####
    copy!(un, up)
    gemv!(TXN, 1, ks, ierk.a4, 1, un)
    diff!(k4, un; diff_args...)
    #### update un ####
    copy!(un, up)
    gemv!(TXN, 1, ks, ierk.bv, 1, un)
end
