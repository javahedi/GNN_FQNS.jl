########################## src/Physics/Heisenberg.jl ##############################
# Spin-1/2 Heisenberg Local Energy (Diagonal + Off-Diagonal)
#
# Hamiltonian:
#   H = Σ_{⟨ij⟩} J_ij [ S_i ⋅ S_j ]
#
# With σ ∈ {−1,+1}, S^z = σ/2  ⇒  S_i^z S_j^z = (σ_i σ_j)/4
#
# Off-diagonal:
#   S_i^+ S_j^- + S_i^- S_j^+
# acts only when σ_i ≠ σ_j (↑↓ or ↓↑):
#
#   |↑↓⟩ → |↓↑⟩
#   |↓↑⟩ → |↑↓⟩
#
###################################################################################

module Heisenberg

using ..Wavefunction
using ..Graph
using LinearAlgebra

export local_energy_batch

################################################################################
# Batched Local Energy
################################################################################

"""
    local_energy_batch(ψ, σ_batch, J_batch)

Compute local energies for B walkers for the spin-1/2 Heisenberg model.

Inputs:
    ψ         : GNNWavefunction
    σ_batch   : B × N integer matrix, σ ∈ {−1,+1}
    J_batch   : B × E matrix, bond strengths

Returns:
    energies  : Vector length B
"""
function local_energy_batch(ψ::GNNWavefunction,
                            σ_batch::Array{<:Integer,2},
                            J_batch)

    g = ψ.graph
    B, N = size(σ_batch)
    E = length(g.edges)

    energies = zeros(Float32, B)

    # Base wavefunction log ψ(σ)
    logψ0 = Wavefunction.logpsi_batch(ψ, σ_batch, J_batch)

    # A buffer for flipped configurations
    σ_flip = similar(σ_batch)

    for e in 1:E
        i, j = g.edges[e]

        ########################################################################
        # 1. Diagonal term: (σ_i σ_j) / 4
        ########################################################################
        @inbounds for b in 1:B
            energies[b] += 0.25f0 * J_batch[b,e] * (σ_batch[b,i] * σ_batch[b,j])
        end

        ########################################################################
        # 2. Off-diagonal term (only if spins are opposite)
        ########################################################################

        # Compute flips only where needed
        @inbounds for b in 1:B
            if σ_batch[b,i] != σ_batch[b,j]          # only for ↑↓ or ↓↑
                # copy base config
                @inbounds for k in 1:N
                    σ_flip[b,k] = σ_batch[b,k]
                end
                # perform swap (↑↓ ↔ ↓↑)
                σ_flip[b,i] = -σ_batch[b,i]
                σ_flip[b,j] = -σ_batch[b,j]
            else
                # mark as non-flippable by storing impossible σ = 2
                σ_flip[b,1] = 2
            end
        end

        # evaluate wavefunction ratios only for valid flips
        logψ_flip = Wavefunction.logpsi_batch(ψ, σ_flip, J_batch)

        @inbounds for b in 1:B
            # skip ↑↑ or ↓↓
            if σ_flip[b,1] == 2
                continue
            end
            ratio = exp(logψ_flip[b] - logψ0[b])
            energies[b] += 0.5f0 * J_batch[b,e] * real(ratio)
        end
    end

    return energies
end

end # module Heisenberg
