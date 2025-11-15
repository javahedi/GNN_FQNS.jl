#!/usr/bin/env julia
using GNN_FQNS
using Random
using Statistics

Random.seed!(1234)

# 4×4 lattice
L = 4
println("Running 4×4 lattice sweep...")
g = build_square_lattice(L)

# Model
net = GNNFQNS(g)                 # your GNN constructor
ψ = GNNWavefunction(g, net)

# Sweep p
ps = range(0.0, 1.0; length=11)

# Results
mFM_list = Float32[]
mAF_list = Float32[]

for p in ps
    println("\n=== p = $p ===")

    # ±J disorder
    disorder_fn = () -> generate_pmJ(g; p=p)

    # Training parameters (small lattice → more disorder averaging)
    epochs = 30
    R = 6         # disorder realizations per epoch
    B = 32        # walkers 
    nsteps = 40   # MCMC , Metropolis–Hastings steps

    train_disorder!(ψ, g;
        epochs=epochs,
        R=R,
        B=B,
        nsteps=nsteps,
        disorder_fn=disorder_fn,
        η=0.01,
        diag_reg=1e-4,
        verbose=true
    )

    # Sample after training
    σ_batch, _ = sample_batch(ψ, g, disorder_fn(); B=128, nsteps=80)

    # Observables
    mFM = mean(magnetization(σ_batch))
    mAF = mean(neel_order(σ_batch, g))

    push!(mFM_list, mFM)
    push!(mAF_list, mAF)

    println("FM magnetization = $mFM")
    println("AFM order        = $mAF")
end

# Save results
using Serialization
serialize("results_L4x4.jld2", (
    ps=ps,
    mFM=mFM_list,
    mAF=mAF_list
))

println("\nSaved results to results_L4x4.jld2")
