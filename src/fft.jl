function clp2(x::Int)
    x == 0 && return 1
    x == 1 && return 2
    x = x - 1
    x = x | (x >> 1)
    x = x | (x >> 2)
    x = x | (x >> 4)
    x = x | (x >> 8)
    x = x | (x >> 16)
    return x + 1
end

function twiddle!(ws::MatI, sz::Int)
    hz = sz >> 1
    ix = 1
    rc = 1.0
    rs = 0.0
    cf = cospi(inv(sz))
    sf = sinpi(inv(sz))
    #### initialize: θ = 0
    @inbounds ws[ix,1] = rc
    @inbounds ws[ix,2] = rs
    while ix < hz
        ix += 1
        tc  = cf * rc - sf * rs
        ts  = sf * rc + cf * rs
        rc  = tc
        rs  = ts
        @inbounds ws[ix,1] = rc
        @inbounds ws[ix,2] = rs
    end
    #### specific case: θ = π / 2
    ix += 1
    @inbounds ws[ix,1] = 0.0
    @inbounds ws[ix,2] = 1.0
    #### remaining cases: θ > π / 2
    jx = ix
    while ix < sz
        ix += 1
        jx -= 1
        @inbounds ws[ix,1] = -ws[jx,1]
        @inbounds ws[ix,2] =  ws[jx,2]
    end
    return ws
end

function butterfly!(y::MatO, x::MatI, w::MatI, jx::Int, kx::Int, wx::Int, H::Int, h::Int, S::Int, s::Int)
    @inbounds for _ in 1:h
        jy = jx + s

        wr = w[wx, 1]
        wi = w[wx, 2]
        xr = x[kx, 1]
        xi = x[kx, 2]
        Xr = x[kx + H, 1]
        Xi = x[kx + H, 2]

        y[jx, 1] = xr + Xr
        y[jx, 2] = xi + Xi
        y[jy, 1] = wr * (xr - Xr) - wi * (xi - Xi)
        y[jy, 2] = wr * (xi - Xi) + wi * (xr - Xr)

        jx += S
        kx += s
        wx += s
    end
end

function difnn(x::AbstractVector, sz::Int)
    a = Matrix{Float64}(undef, sz, 2)
    b = Matrix{Float64}(undef, sz, 2)
    w = Matrix{Float64}(undef, sz, 2)
    H = h = sz >> 1
    s = 1
    S = 2
    r = false

    twiddle!(w, H)
    @inbounds for i in eachindex(1:sz)
        a[i, 1] = x[i]
        a[i, 2] = 0.0
    end

    while h > 0
        for ix in 1:s
            r ? butterfly!(a, b, w, ix, ix, 1, H, h, S, s) : butterfly!(b, a, w, ix, ix, 1, H, h, S, s)
        end

        h >>= 1
        s <<= 1
        S <<= 1
        r = !r
    end

    if r
        fftshift!(b)
        return b
    else
        fftshift!(a)
        return a
    end
end

function fftshift!(x::MatI)
    N = size(x, 1) >> 1
    @inbounds for i in eachindex(1:N)
        temp_r   = x[i,1]
        x[i,1]   = x[i+N,1]
        x[i+N,1] = temp_r
        temp_i   = x[i,2]
        x[i,2]   = x[i+N,2]
        x[i+N,2] = temp_i
    end
end 

function fftfreq(t::VecI{T}) where T<:Real
    f  = similar(t)
    Δf = inv(t[end] - t[1])
    nhalfp1 = length(f) >> 1 + 1
    @simd for i in eachindex(f)
        @inbounds f[i] = Δf * (i - nhalfp1)
    end
    return f
end
