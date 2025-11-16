#!/usr/bin/env julia
using Distributed, Serialization

###############################################################################
# 1. Add workers safely
###############################################################################

addprocs(8)

# Workers should ONLY activate, never instantiate
@everywhere begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))   # Activates environment
end

###############################################################################
# 2. Load project modules on workers
###############################################################################

@everywhere using GNN_FQNS
@everywhere using Random, Statistics

###############################################################################
# 3. Disorder distribution
###############################################################################

@everywhere function generate_pmJ(graph; p)
    E = length(graph.edges)
    return Float32[rand() < p ? 1f0 : -1f0 for _ in 1:E]
end

@everywhere function make_model(graph)
    net = GNNFQNS(graph)
    return GNNWavefunction(graph, net)
end

###############################################################################
# 4. Single-p training + sampling
###############################################################################

@everywhere function run_single_p(graph, p;
        epochs    = 30,
        R         = 6,
        B         = 32,
        nsteps    = 40,
        η         = 0.01,
        diag_reg  = 1e-4
    )

    Random.seed!()

    ψ = make_model(graph)
    disorder_fn = () -> generate_pmJ(graph; p=p)

    # ---- TRAINING ----
    hist = train_disorder!(
        ψ, graph;
        epochs    = epochs,
        R         = R,
        B         = B,
        nsteps    = nsteps,
        disorder_fn = disorder_fn,
        η         = η,
        diag_reg  = diag_reg,
        verbose   = false
    )

    # ---- SAMPLING ----
    Bsample = 128
    nsteps_sample = 80

    J = disorder_fn()
    J_batch = repeat(J', Bsample, 1)

    σ_batch, _ = sample_batch(ψ, graph, Bsample, nsteps_sample, J_batch)

    mFM = mean(magnetization(σ_batch))
    mAF = mean(neel_order(σ_batch, graph))

    return (p=p, mFM=mFM, mAF=mAF, hist=hist)
end

###############################################################################
# 5. Main loop with checkpoints
###############################################################################

println("Running parallel 4×4 lattice sweep with checkpoints...")

L = 8
g = build_square_lattice(L)

ps = collect(range(0.0, 1.0; length=11))

checkpoint_file = "checkpoint_L4x4.jld2"

# Load checkpoint
results = Dict{Float64,Any}()
if isfile(checkpoint_file)
    println("Loading checkpoint...")
    results = deserialize(checkpoint_file)
end

# Remaining points
todo = [p for p in ps if !haskey(results, p)]
println("Remaining p-values: ", todo)

# Sequential dispatch (safe; can parallelize later with pmap)
for p in todo
    println("\n=== Running p = $p ===")
    res = run_single_p(g, p)
    results[p] = res

    serialize(checkpoint_file, results)
    println("Checkpoint saved.")
end

# Save final result
final_file = "results_parallel_L4x4_full.jld2"
serialize(final_file, results)

println("\nFINISHED. Saved to $final_file")
