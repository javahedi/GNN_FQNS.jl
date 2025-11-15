#!/usr/bin/env julia
using Distributed

# -------------------------------------------------------------
# Launch worker processes
# -------------------------------------------------------------
addprocs(8)

@everywhere using GNN_FQNS
@everywhere using Random, Statistics


# -------------------------------------------------------------
# ±J disorder distribution
# -------------------------------------------------------------
@everywhere function generate_pmJ(graph; p)
    E = length(graph.edges)
    return Float32[ rand() < p ? 1f0 : -1f0 for _ in 1:E ]
end


# -------------------------------------------------------------
# Build a NEW model ψ for each p (no cross-contamination)
# -------------------------------------------------------------
@everywhere function make_model(graph)
    net = GNNFQNS(graph)
    return GNNWavefunction(graph, net)
end


# -------------------------------------------------------------
# Core function: train + sample + return full training history
# -------------------------------------------------------------
@everywhere function run_single_p(graph, p;
        epochs = 30,
        R      = 6,
        B      = 32,
        nsteps = 40,
        η      = 0.01,
        diag_reg = 1e-4
    )

    Random.seed!()   # independent RNG per worker

    ψ = make_model(graph)
    disorder_fn = () -> generate_pmJ(graph; p = p)

    # ---------------- TRAINING (capture full history) ----------------
    history = train_disorder!(ψ, graph;
        epochs      = epochs,
        R           = R,
        B           = B,
        nsteps      = nsteps,
        disorder_fn = disorder_fn,
        η           = η,
        diag_reg    = diag_reg,
        verbose     = false
    )
    # history is a Dict:
    #   :losses       → energy per epoch
    #   :E_var        → variance per epoch
    #   :logpsi_avg   → log|ψ| per epoch
    #   :E_per_epoch  → all disorder energies


    # ---------------- SAMPLING AFTER TRAINING ----------------
    Bsample = 128
    nsteps_sample = 80

    J = disorder_fn()
    J_batch = repeat(J', Bsample, 1)

    σ_batch, _ = sample_batch(
        ψ,
        graph,
        Bsample,
        nsteps_sample,
        J_batch
    )

    # Observables
    mFM = mean(magnetization(σ_batch))
    mAF = mean(neel_order(σ_batch, graph))

    # Return everything
    return (
        p     = p,
        mFM   = mFM,
        mAF   = mAF,
        hist  = history   # full training curves!
    )
end


# -------------------------------------------------------------
# Main script
# -------------------------------------------------------------
println("Running parallel 4×4 lattice sweep...")

L = 4
g = build_square_lattice(L)

ps = collect(range(0.0, 1.0; length=11))

# Parallel p-sweep using pmap
results = pmap(p -> run_single_p(g, p), ps)

# Save results
using Serialization
serialize("results_parallel_L4x4.jld2", results)

println("\nSaved results to results_parallel_L4x4.jld2")
println("Each entry is a NamedTuple: (p, mFM, mAF, hist)")
println("Where hist contains: :losses, :E_var, :logpsi_avg, :E_per_epoch")
