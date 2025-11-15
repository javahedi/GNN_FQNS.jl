############################ src/Optim/SR.jl ###################################
module SR

using Flux
using Zygote
using LinearAlgebra
using Statistics
using ..Wavefunction
using ..Heisenberg

export compute_logderivs, sr_step!

################################################################################
# 1. Compute Log-Derivatives O_k(s)
################################################################################

"""
    compute_logderivs(ψ, σ_batch, J_batch)

Return:
    O : B × P matrix of log-derivatives
    logψ : length-B vector of logψ values
"""
function compute_logderivs(ψ::GNNWavefunction,
                           σ_batch::Array{<:Integer,2},
                           J_batch)

    net = ψ.net

    # Flatten parameters
    flatθ, unflatten = Flux.destructure(net)
    P = length(flatθ)
    B = size(σ_batch, 1)

    # Allocate output
    O = zeros(Float32, B, P)

    # Precompute logψ (not used for gradients, but useful to return)
    logψ = Wavefunction.logpsi_batch(ψ, σ_batch, J_batch)

    # Zygote needs a REAL scalar output; we differentiate Re(logψ)
    for b in 1:B
        f_real = θ_flat -> begin
            net2 = unflatten(θ_flat)
            ψtmp = GNNWavefunction(ψ.graph, net2)
            real(Wavefunction.logpsi(ψtmp, σ_batch[b,:], J_batch[b,:]))
        end

        g_real = Zygote.gradient(f_real, flatθ)[1]

        # SR uses real log-derivatives
        O[b, :] .= g_real
    end

    return O, logψ
end


################################################################################
# 2. SR Update
################################################################################

"""
    sr_step!(ψ, σ_batch, J_batch; η, diag_reg)

Perform in-place SR (natural gradient) update.
"""
function sr_step!(ψ::GNNWavefunction,
                  σ_batch, J_batch;
                  η=0.05,
                  diag_reg=1e-4)

    net = ψ.net

    # 1. Log-derivatives
    O, logψ = compute_logderivs(ψ, σ_batch, J_batch)

    B, P = size(O)

    # 2. Local energies
    E_loc = Heisenberg.local_energy_batch(ψ, σ_batch, J_batch)
    meanE = mean(E_loc)

    # 3. Mean of O_k
    meanO = vec(mean(O, dims=1))

    # 4. Covariant gradient g_k
    g = zeros(Float32, P)
    for b in 1:B
        Ob = O[b, :] .- meanO
        g .+= (E_loc[b] - meanE) .* Ob
    end
    g ./= B

    # 5. Covariance matrix S
    S = zeros(Float32, P, P)
    for b in 1:B
        Ob = O[b, :] .- meanO
        S .+= Ob * Ob'
    end
    S ./= B

    # 6. Regularize
    for k in 1:P
        S[k,k] += diag_reg
    end

    # 7. Solve natural gradient
    Δθ = -η * (S \ g)

    # 8. Zygote-safe parameter update (NO mutation of flat)
    flat, unflatten = Flux.destructure(net)
    newflat = flat .+ Δθ        # non-mutating
    ψ.net = unflatten(newflat)

    return nothing
end

end # module SR
