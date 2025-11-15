############################ src/Sampler/MCMC.jl ############################
# Batched Metropolis-Hastings sampler for FQNS
#
# Supports:
#   • batched sampling (B walkers)
#   • GNNFQNS backend
#   • edge-vector J disorder
#   • GPU or CPU execution
#   • easy interface for SR and training
##############################################################################

module MCMC

using ..Wavefunction
using ..Graph
using Random

export MCMCSampler,
       init_states,
       step!,
       sample_batch

################################################################################
# 1. Sampler container
################################################################################

"""
    MCMCSampler(graph, B)

Container holding:
    • graph     : lattice structure
    • B         : number of walkers
    • σ         : B×N spin configurations
    • logψ      : B complex values
"""
struct MCMCSampler
    graph::LatticeGraph
    B::Int
    σ::Array{Int,2}         # B × N
    logψ::Vector{ComplexF64}
end

################################################################################
# 2. Initialize states
################################################################################

"""
    init_states(graph, B; init="random")

Return σ_batch (B×N) initial spin states.
"""
function init_states(graph::LatticeGraph, B::Int; init="random")
    N = graph.N
    σ = zeros(Int, B, N)

    if init == "random"
        for b in 1:B, i in 1:N
            σ[b,i] = rand() < 0.5 ? 1 : -1
        end
    elseif init == "all_up"
        σ .= 1
    elseif init == "all_down"
        σ .= -1
    else
        error("Unknown init=$init")
    end

    return σ
end


################################################################################
# 3. Perform one MCMC transition (Metropolis-Hastings)
################################################################################

"""
    step!(sampler, ψ, J_batch)

Perform one MH step for all walkers.

Inputs:
    sampler   : MCMCSampler
    ψ         : GNNWavefunction
    J_batch   : B × E couplings
"""
function step!(sampler::MCMCSampler,
               ψ::GNNWavefunction,
               J_batch)

    g = sampler.graph
    B = sampler.B
    N = g.N

    σ = sampler.σ
    logψ_old = sampler.logψ

    # 1. Pick which site to flip for each walker
    flip_sites = rand(1:N, B)

    # 2. Create proposal batch
    σ_new = copy(σ)
    for b in 1:B
        i = flip_sites[b]
        σ_new[b,i] = -σ[b,i]   # flip
    end

    # 3. Evaluate wavefunction for proposals
    logψ_new = Wavefunction.logpsi_batch(ψ, σ_new, J_batch)

    # 4. Metropolis acceptance
    for b in 1:B
        Δ = 2 * real(logψ_new[b] - logψ_old[b])

        # accept?
        if Δ ≥ 0 || rand() < exp(Δ)
            σ[b,:] .= σ_new[b,:]
            logψ_old[b] = logψ_new[b]
        end
    end
end


################################################################################
# 4. Collect samples after N steps
################################################################################

"""
    sample_batch(ψ, graph, B, nsteps, J_batch)

Return:
    σ_batch    : B × N spins
    logψ_batch : B complex log amplitudes
"""
function sample_batch(ψ::GNNWavefunction,
                      graph::LatticeGraph,
                      B::Int,
                      nsteps::Int,
                      J_batch)

    # initialize sampler
    sampler = MCMCSampler(graph, B,
                           init_states(graph, B),
                           zeros(ComplexF64, B))

    # compute initial logψ
    sampler.logψ .= Wavefunction.logpsi_batch(ψ, sampler.σ, J_batch)

    # do MCMC steps
    for _ in 1:nsteps
        step!(sampler, ψ, J_batch)
    end

    return sampler.σ, sampler.logψ
end

end # module MCMC
