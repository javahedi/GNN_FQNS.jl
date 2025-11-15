using Test
using Random
using GNN_FQNS

# IMPORTANT: Explicit import to avoid ambiguity
import GNN_FQNS.MCMC: sample_batch

@testset "MCMC Tests" begin
    Random.seed!(1)

    g = build_square_lattice(4)
    net = GNNFQNS(g)
    ψ = GNNWavefunction(g, net)

    B = 8
    E = length(g.edges)
    Jb = randn(Float32, B, E)

    σb, logψb = sample_batch(ψ, g, B, 10, Jb)

    @test size(σb) == (B, g.N)
    @test all(abs.(σb) .== 1)
    @test length(logψb) == B
end
