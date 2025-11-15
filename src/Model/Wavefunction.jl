############################ src/Model/Wavefunction.jl ########################
# Wavefunction wrapper for GNN-FQNS
###############################################################################

module Wavefunction

using ..Graph
using ..GNN

export GNNWavefunction, logpsi, logpsi_batch

###############################################################################
# 1. Wavefunction container (MUTABLE for SR updates!)
###############################################################################

mutable struct GNNWavefunction
    graph::LatticeGraph
    net
end

###############################################################################
# 2. Single-sample logpsi
###############################################################################

function logpsi(ψ::GNNWavefunction,
                σ::Vector{Int},
                J::Vector)

    return GNN.logpsi_gnn_single(ψ.net, ψ.graph, σ, J)
end

###############################################################################
# 3. Batched logpsi
###############################################################################

function logpsi_batch(ψ::GNNWavefunction,
                      σ_batch::Array{<:Integer,2},
                      J_batch)

    return GNN.logpsi_gnn(ψ.net, σ_batch, J_batch)
end

end # module Wavefunction
