using Test
using GNN_FQNS   # imports all public API, including logpsi

@testset "Wavefunction Wrapper" begin
    g = build_square_lattice(4)
    net = GNNFQNS(g)
    ψ = GNNWavefunction(g, net)

    σ = rand([-1,1], g.N)
    J = randn(Float32, length(g.edges))

    logψ1 = logpsi(ψ, σ, J)
    logψ2 = logpsi(ψ, σ, J)

    @test isapprox(logψ1, logψ2; atol=1e-6)
end
