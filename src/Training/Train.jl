############################ src/Training/Train.jl ###############################
module Train

using Random
using Statistics
using Printf
using ..Graph
using ..Wavefunction
using ..MCMC
using ..Heisenberg
using ..SR

export train_epoch!, train_disorder!


################################################################################
# 1. train_epoch!
################################################################################

"""
    train_epoch!(ψ, graph, R, B, nsteps, disorder_fn; η, diag_reg)

Perform ONE epoch of SR training over R disorder realizations.

Returns a NamedTuple:
    (
        E_avg,        # average energy across disorders
        E_all,        # vector of energies per disorder
        logpsi_avg,   # mean log|ψ|
        E_var         # variance of energy estimator
    )
"""
function train_epoch!(ψ::GNNWavefunction,
                      graph::LatticeGraph,
                      R::Int,
                      B::Int,
                      nsteps::Int,
                      disorder_fn;
                      η=0.05,
                      diag_reg=1e-4)

    E_all      = Float32[]         # energies per disorder
    logpsi_all = Float32[]         # log|ψ| stats per disorder

    for r in 1:R
        # 1. Disorder realization: J[e]
        J = disorder_fn()
        J_batch = repeat(J', B, 1)        # B × E

        # 2. MCMC sampling
        σ_batch, logψ_batch = MCMC.sample_batch(
            ψ, graph,
            B, nsteps,
            J_batch
        )

        # 3. Local energies
        E_loc = Heisenberg.local_energy_batch(ψ, σ_batch, J_batch)

        push!(E_all, mean(E_loc))
        push!(logpsi_all, mean(abs.(logψ_batch)))

        # 4. Natural gradient step
        SR.sr_step!(ψ, σ_batch, J_batch; η=η, diag_reg=diag_reg)
    end

    return (
        E_avg      = mean(E_all),
        E_all      = E_all,
        logpsi_avg = mean(logpsi_all),
        E_var      = var(E_all),
    )
end


################################################################################
# 2. train_disorder!
################################################################################

"""
    train_disorder!(ψ, graph;
                    epochs, R, B, nsteps, disorder_fn,
                    η, diag_reg, verbose=true)

Runs the full disorder-averaged VMC training.

Returns:
    history :: Dict with:
        :losses       → [E_avg(epoch)]
        :E_per_epoch  → [[E_all(epoch_1)], [E_all(epoch_2)], ...]
        :logpsi_avg   → per-epoch mean |logψ|
        :E_var        → per-epoch variance
"""
function train_disorder!(ψ::GNNWavefunction,
                         graph::LatticeGraph;
                         epochs::Int,
                         R::Int,
                         B::Int,
                         nsteps::Int,
                         disorder_fn,
                         η=0.05,
                         diag_reg=1e-4,
                         verbose=true)

    # Storage for full training history
    losses        = Float32[]
    E_per_epoch   = Vector{Vector{Float32}}()
    logpsi_avgs   = Float32[]
    E_vars        = Float32[]

    prev_loss = NaN

    for ep in 1:epochs
        result = train_epoch!(ψ, graph, R, B, nsteps, disorder_fn;
                              η=η, diag_reg=diag_reg)

        # unpack results
        E_avg      = result.E_avg
        E_all      = result.E_all
        logpsi_avg = result.logpsi_avg
        E_var      = result.E_var

        # Store
        push!(losses, E_avg)
        push!(E_per_epoch, E_all)
        push!(logpsi_avgs, logpsi_avg)
        push!(E_vars, E_var)

        # Pretty progress print
        if verbose
            dE = isnan(prev_loss) ? 0 : E_avg - prev_loss
            println(@sprintf(
                "Epoch %3d/%d | E_avg = %8.5f | ΔE = %+8.5f | Var = %.4e | log|ψ| = %.4f",
                ep, epochs, E_avg, dE, E_var, logpsi_avg
            ))
        end

        prev_loss = E_avg
    end

    return Dict(
        :losses       => losses,
        :E_per_epoch  => E_per_epoch,
        :logpsi_avg   => logpsi_avgs,
        :E_var        => E_vars,
    )
end

end # module Train
