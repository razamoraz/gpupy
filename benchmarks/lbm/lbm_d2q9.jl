#!/usr/bin/env julia
# D2Q9 BGK LBM - Gaussian Acoustic Pulse (Julia Benchmark)
# Aligned with Python coordinate standards: Row=Y (i), Col=X (j).

using Printf
using Statistics
using LinearAlgebra
using Base.Threads
using StaticArrays

# -----------------------------------------------------------------------------
# Conditionally load packages
# -----------------------------------------------------------------------------
const HAS_CUDA = try
    using CUDA
    true
catch e
    false
end

const HAS_HDF5 = try
    using HDF5
    true
catch e
    false
end

# -----------------------------------------------------------------------------
# D2Q9 Constants
# -----------------------------------------------------------------------------
const D = 2
const Q = 9

# Lattice velocities (Matched to python order)
# x: column index shift, y: row index shift
# 0: (0,0), 1: (1,0), 2: (0,1), 3: (-1,0), 4: (0,-1), 5: (1,1), 6: (-1,1), 7: (-1,-1), 8: (1,-1)
const cx = @SVector [0, 1, 0, -1,  0, 1, -1, -1,  1]
const cy = @SVector [0, 0, 1,  0, -1, 1,  1, -1, -1]
const w_f64 = @SVector [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]

# -----------------------------------------------------------------------------
# Parameters & CLI Parsing
# -----------------------------------------------------------------------------
function parse_args()
    nx, ny, iters = 512, 512, 100
    precision = Float32
    export_data = false
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--nx" && i+1 <= length(ARGS)
            nx = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--ny" && i+1 <= length(ARGS)
            ny = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--iterations" && i+1 <= length(ARGS)
            iters = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--precision" && i+1 <= length(ARGS)
            precision = ARGS[i+1] == "f64" ? Float64 : Float32; i += 2
        elseif ARGS[i] == "--export"
            export_data = true; i += 1
        else
            i += 1
        end
    end
    return nx, ny, iters, precision, export_data
end

const NX, NY, ITERS, T, EXPORT = parse_args()
const TOTAL_CELLS = NX * NY
const VISCOSITY = T(0.02)
const OMEGA = T(1.0 / (3.0 * Float64(VISCOSITY) + 0.5))
const w = SVector{Q, T}(w_f64)

const ρ0 = T(1.0)
const δρ = T(1e-3)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@inline function feq(ρ::T, ux::T, uy::T, q::Int) where T
    u2 = ux*ux + uy*uy
    cu = T(cx[q])*ux + T(cy[q])*uy
    return w[q] * ρ * (T(1.0) + T(3.0)*cu + T(4.5)*cu*cu - T(1.5)*u2)
end

function init_density(T_type)
    ρ = fill(T_type(ρ0), NX, NY)
    for i in 1:NX, j in 1:NY
        # Row-major matching: i is row (Y), j is column (X)
        # Python: linspace -1 to 1 for NX (X), NY (Y)
        # meshgrid(x, y) results in X varying across cols, Y across rows.
        x = T_type(-1.0 + 2.0 * (j - 1) / (NX - 1))
        y = T_type(-1.0 + 2.0 * (i - 1) / (NY - 1))
        r2 = x*x + y*y
        ρ[i,j] = T_type(ρ0) + T_type(δρ) * exp(T_type(-100.0) * r2)
    end
    return ρ
end

function compute_moments(f::Array{T, 3}) where T
    rho = sum(f, dims=1)[1, :, :]
    ux = zeros(T, NX, NY)
    uy = zeros(T, NX, NY)
    for i in 1:NX, j in 1:NY
        ri = rho[i,j]
        uxi = zero(T); uyi = zero(T)
        for q in 1:Q
            fi = f[q,i,j]
            uxi += T(cx[q]) * fi
            uyi += T(cy[q]) * fi
        end
        ux[i,j] = uxi / ri
        uy[i,j] = uyi / ri
    end
    return rho, ux, uy
end

function export_results(filename, rho, ux, uy)
    if !HAS_HDF5 return end
    h5open(filename, "w") do file
        # Transpose back for Python-style (Y, X) comparison in standard tools
        # HDF5 in Julia is (Fastest, ..., Slowest). Python is (Slowest, ..., Fastest).
        # We save as is, and Python will see it flipped.
        file["rho"] = rho
        file["ux"] = ux
        file["uy"] = uy
    end
    println("Results exported to $filename")
end

# -----------------------------------------------------------------------------
# 1. CPU Serial
# -----------------------------------------------------------------------------
function run_cpu_serial(iters_count)
    f = zeros(T, Q, NX, NY); fnew = zeros(T, Q, NX, NY)
    ρ = init_density(T)
    for i in 1:NX, j in 1:NY
        ρi = ρ[i,j]; for q in 1:Q f[q,i,j] = feq(ρi, T(0.0), T(0.0), q) end
    end

    t_start = time_ns()
    @inbounds for t in 1:iters_count
        # Collision
        for i in 1:NX, j in 1:NY
            ρi = zero(T); uxi = zero(T); uyi = zero(T)
            @simd for q in 1:Q
                fi = f[q,i,j]; ρi += fi; uxi += T(cx[q]) * fi; uyi += T(cy[q]) * fi
            end
            uxi /= ρi; uyi /= ρi; u2 = uxi*uxi + uyi*uyi
            @simd for q in 1:Q
                cu = T(cx[q])*uxi + T(cy[q])*uyi
                feqi = w[q] * ρi * (T(1.0) + T(3.0)*cu + T(4.5)*cu*cu - T(1.5)*u2)
                f[q,i,j] = f[q,i,j] - OMEGA * (f[q,i,j] - feqi)
            end
        end
        # Streaming (Pull)
        for i in 1:NX
            ni = i == 1 ? NX : i - 1; pi = i == NX ? 1 : i + 1
            for j in 1:NY
                nj = j == 1 ? NY : j - 1; pj = j == NY ? 1 : j + 1
                @inbounds begin
                    # Index shift: cy applies to i (rows), cx applies to j (cols)
                    fnew[1,i,j] = f[1,i,j]
                    fnew[2,i,j] = f[2,i,nj]   # (1,0) pull from -x
                    fnew[3,i,j] = f[3,ni,j]   # (0,1) pull from -y
                    fnew[4,i,j] = f[4,i,pj]   # (-1,0) pull from +x
                    fnew[5,i,j] = f[5,pi,j]   # (0,-1) pull from +y
                    fnew[6,i,j] = f[6,ni,nj]  # (1,1) pull from (-x, -y)
                    fnew[7,i,j] = f[7,ni,pj]  # (-1,1) pull from (+x, -y)
                    fnew[8,i,j] = f[8,pi,pj]  # (-1,-1) pull from (+x, +y)
                    fnew[9,i,j] = f[9,pi,nj]  # (1,-1) pull from (-x, +y)
                end
            end
        end
        f, fnew = fnew, f
    end
    dt = (time_ns() - t_start) * 1e-9
    rho, ux, uy = compute_moments(f)
    return (rho=rho, ux=ux, uy=uy), (TOTAL_CELLS * iters_count) / (dt * 1e6)
end

# -----------------------------------------------------------------------------
# 2. CPU Threaded
# -----------------------------------------------------------------------------
function run_cpu_threaded(iters_count)
    f = zeros(T, Q, NX, NY); fnew = zeros(T, Q, NX, NY)
    ρ = init_density(T)
    @threads for i in 1:NX
        for j in 1:NY
            ρi = ρ[i,j]; for q in 1:Q f[q,i,j] = feq(ρi, T(0.0), T(0.0), q) end
        end
    end

    t_start = time_ns()
    @inbounds for t in 1:iters_count
        let f=f
            @threads for i in 1:NX
                for j in 1:NY
                    ρi = zero(T); uxi = zero(T); uyi = zero(T)
                    @simd for q in 1:Q
                        fi = f[q,i,j]; ρi += fi; uxi += T(cx[q]) * fi; uyi += T(cy[q]) * fi
                    end
                    uxi /= ρi; uyi /= ρi; u2 = uxi*uxi + uyi*uyi
                    @simd for q in 1:Q
                        cu = T(cx[q])*uxi + T(cy[q])*uyi
                        feqi = w[q] * ρi * (T(1.0) + T(3.0)*cu + T(4.5)*cu*cu - T(1.5)*u2)
                        f[q,i,j] = f[q,i,j] - OMEGA * (f[q,i,j] - feqi)
                    end
                end
            end
        end
        let f=f, fnew=fnew
            @threads for i in 1:NX
                ni = i == 1 ? NX : i - 1; pi = i == NX ? 1 : i + 1
                for j in 1:NY
                    nj = j == 1 ? NY : j - 1; pj = j == NY ? 1 : j + 1
                    fnew[1,i,j] = f[1,i,j]; fnew[2,i,j] = f[2,i,nj]; fnew[3,i,j] = f[3,ni,j]
                    fnew[4,i,j] = f[4,i,pj]; fnew[5,i,j] = f[5,pi,j]; fnew[6,i,j] = f[6,ni,nj]
                    fnew[7,i,j] = f[7,ni,pj]; fnew[8,i,j] = f[8,pi,pj]; fnew[9,i,j] = f[9,pi,nj]
                end
            end
        end
        f, fnew = fnew, f
    end
    dt = (time_ns() - t_start) * 1e-9
    rho, ux, uy = compute_moments(f)
    return (rho=rho, ux=ux, uy=uy), (TOTAL_CELLS * iters_count) / (dt * 1e6)
end

# -----------------------------------------------------------------------------
# 3. GPU High-Level
# -----------------------------------------------------------------------------
function run_gpu_highlevel(iters_count)
    if !HAS_CUDA return nothing, 0.0 end
    ρ_init = CuArray(init_density(T))
    f = CuArray(zeros(T, Q, NX, NY))
    fnew = CuArray(zeros(T, Q, NX, NY))
    
    # Initialization
    f .= reshape(ρ_init, 1, NX, NY) .* reshape(T.(w), Q, 1, 1)
    
    # Pre-calculate constants for broadcasting
    cu_cx = CuArray(reshape(T.(cx), Q, 1, 1))
    cu_cy = CuArray(reshape(T.(cy), Q, 1, 1))

    CUDA.synchronize()
    t_start = time_ns()
    @inbounds for t in 1:iters_count
        # Collision (High-Level Broadcasting)
        rho = dropdims(sum(f, dims=1), dims=1)
        ux = dropdims(sum(f .* cu_cx, dims=1), dims=1) ./ rho
        uy = dropdims(sum(f .* cu_cy, dims=1), dims=1) ./ rho
        u2 = ux.^2 .+ uy.^2
        
        for q in 1:Q
            cu = T(cx[q]) .* ux .+ T(cy[q]) .* uy
            feq_q = T(w[q]) .* rho .* (T(1.0) .+ T(3.0).*cu .+ T(4.5).*cu.*cu .- T(1.5).*u2)
            # Update f in place using views
            fq_view = @view f[q,:,:]
            fq_view .-= OMEGA .* (fq_view .- feq_q)
        end

        # Streaming
        for q in 1:Q
            fnew[q,:,:] = circshift(@view(f[q,:,:]), (cy[q], cx[q]))
        end
        f, fnew = fnew, f
    end
    CUDA.synchronize()
    dt = (time_ns() - t_start) * 1e-9
    rho, ux, uy = compute_moments(Array(f))
    return (rho=rho, ux=ux, uy=uy), (TOTAL_CELLS * iters_count) / (dt * 1e6)
end

# -----------------------------------------------------------------------------
# 4. GPU Low-Level
# -----------------------------------------------------------------------------
if HAS_CUDA
    function lbm_kernel_fused!(f, fnew, omega::T) where T
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        if i <= NX && j <= NY
            f_loc = MVector{Q, T}(undef)
            for q in 1:Q; f_loc[q] = f[q, i, j] end
            rho = zero(T); ux = zero(T); uy = zero(T)
            for q in 1:Q
                fi = f_loc[q]; rho += fi; ux += T(cx[q]) * fi; uy += T(cy[q]) * fi
            end
            ux /= rho; uy /= rho; u2 = ux*ux + uy*uy
            for q in 1:Q
                cu = T(cx[q])*ux + T(cy[q])*uy
                feqi = w[q] * rho * (T(1.0) + T(3.0)*cu + T(4.5)*cu*cu - T(1.5)*u2)
                ni = mod1(i + cy[q], NX); nj = mod1(j + cx[q], NY)
                fnew[q, ni, nj] = f_loc[q] - omega * (f_loc[q] - feqi)
            end
        end
        return nothing
    end
end

function run_gpu_lowlevel(iters_count)
    if !HAS_CUDA return nothing, 0.0 end
    f = CuArray(zeros(T, Q, NX, NY)); fnew = CuArray(zeros(T, Q, NX, NY))
    ρ = CuArray(init_density(T))
    f .= reshape(ρ, 1, NX, NY) .* reshape(T.(w), Q, 1, 1)
    threads = (16, 16); blocks = (cld(NX, threads[1]), cld(NY, threads[2]))
    CUDA.synchronize()
    t_start = time_ns()
    for t in 1:iters_count
        @cuda threads=threads blocks=blocks lbm_kernel_fused!(f, fnew, T(OMEGA))
        f, fnew = fnew, f
    end
    CUDA.synchronize()
    dt = (time_ns() - t_start) * 1e-9
    rho, ux, uy = compute_moments(Array(f))
    return (rho=rho, ux=ux, uy=uy), (TOTAL_CELLS * iters_count) / (dt * 1e6)
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
function main()
    println("-"^50)
    println("Julia D2Q9 LBM Benchmark (Refactored/Aligned)")
    println("Precision: $T | Grid: $NX x $NY | Iterations: $ITERS")
    println("-"^50)
    
    println("Initializing & Warmup (JIT)...")
    run_cpu_serial(1); run_cpu_threaded(1)
    if HAS_CUDA; run_gpu_highlevel(1); run_gpu_lowlevel(1) end
    
    print("1. CPU Serial ... ")
    res_ser, ml_ser = run_cpu_serial(ITERS)
    @printf("%.2f MLUPS\n", ml_ser)
    if EXPORT export_results("julia_ser_$(T)_results.h5", res_ser.rho, res_ser.ux, res_ser.uy) end
    
    print("2. CPU Threaded ($(nthreads()) threads) ... ")
    res_thr, ml_thr = run_cpu_threaded(ITERS)
    @printf("%.2f MLUPS (x%.2f speedup)\n", ml_thr, ml_thr/ml_ser)
    if EXPORT export_results("julia_thr_$(T)_results.h5", res_thr.rho, res_thr.ux, res_thr.uy) end
    
    if HAS_CUDA
        print("3. GPU High-Level ... ")
        res_gh, ml_gh = run_gpu_highlevel(ITERS)
        @printf("%.2f MLUPS (x%.2f speedup)\n", ml_gh, ml_gh/ml_ser)
        if EXPORT export_results("julia_gh_$(T)_results.h5", res_gh.rho, res_gh.ux, res_gh.uy) end
        
        print("4. GPU Low-Level (Fused) ... ")
        res_gl, ml_gl = run_gpu_lowlevel(ITERS)
        @printf("%.2f MLUPS (x%.2f speedup)\n", ml_gl, ml_gl/ml_ser)
        if EXPORT export_results("julia_gl_$(T)_results.h5", res_gl.rho, res_gl.ux, res_gl.uy) end
    end
    println("-"^50)
end

main()
