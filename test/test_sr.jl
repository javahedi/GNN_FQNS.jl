using Test
using Random
using GNN_FQNS
using Flux
 

@testset "SR Tests" begin
    Random.seed!(1)

    g = build_square_lattice(4)
    net = GNNFQNS(g; d_node=4, d_edge=2, d_hidden=8)
    ψ = GNNWavefunction(g, net)

    B = 10
    E = length(g.edges)
    σb = rand([-1,1], B, g.N)
    Jb = randn(Float32, B, E)

    flat0, _ = Flux.destructure(ψ.net)

    sr_step!(ψ, σb, Jb; η=0.01)

    flat1, _ = Flux.destructure(ψ.net)
    @test !(flat0 ≈ flat1)
end
