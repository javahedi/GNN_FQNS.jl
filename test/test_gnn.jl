using Test
using Random
using GNN_FQNS

# Explicitly import the Wavefunction API
import GNN_FQNS.Wavefunction: logpsi, logpsi_batch

@testset "GNN Tests" begin 
    Random.seed!(123)

    L = 4
    g = build_square_lattice(L)
    net = GNNFQNS(g; d_node=8, d_edge=4, d_hidden=16)
    ψ = GNNWavefunction(g, net)

    N = g.N
    E = length(g.edges)

    σ = rand([-1,1], N)
    J = randn(Float32, E)

    # single value
    logψ_single = logpsi(ψ, σ, J)
    @test logψ_single isa Complex

    # batch of size 1 must match
    σb = reshape(σ, 1, :)
    Jb = reshape(J, 1, :)

    logψ_batch = logpsi_batch(ψ, σb, Jb)[1]
    @test isapprox(logψ_single, logψ_batch; atol=1e-6)
end
