#!/usr/bin/env julia
using Distributed

addprocs(2)

@everywhere using GNN_FQNS
@everywhere using Random, Statistics

@everywhere begin
    function generate_pmJ(graph; p)
        E = length(graph.edges)
        return Float32[rand() < p ? 1f0 : -1f0 for _ in 1:E]
    end

    function make_model(graph)
        net = GNNFQNS(graph)
        return GNNWavefunction(graph, net)
    end

    # Tiny 1-epoch run (quick sanity check)
    function quick_test(graph, p)
        ψ = make_model(graph)
        disorder_fn = () -> generate_pmJ(graph; p=p)

        hist = train_disorder!(ψ, graph;
            epochs=1,
            R=1,
            B=2,
            nsteps=3,
            disorder_fn=disorder_fn,
            η=0.01,
            diag_reg=1e-4,
            verbose=false
        )

        @assert :losses ∈ keys(hist)
        @assert isfinite(hist[:losses][1])

        # short sampling
        J = disorder_fn()
        J_batch = repeat(J', 4, 1)

        σ_batch, _ = sample_batch(ψ, graph, 4, 5, J_batch)

        mFM = mean(magnetization(σ_batch))
        mAF = mean(neel_order(σ_batch, graph))

        @assert isfinite(mFM)
        @assert isfinite(mAF)

        return true
    end
end

println("\nRunning fast parallel self-test…")

L = 4
g = build_square_lattice(L)

ps = [0.2, 0.8]

results = pmap(p -> quick_test(g, p), ps)

println("Self-test passed for p = 0.2 and p = 0.8 ✔")
println("Safe to launch full cluster job.")
