using Test
using GNN_FQNS

# IMPORTANT: explicitly import to avoid ambiguity
import GNN_FQNS.Heisenberg: local_energy_batch

@testset "Local Energy Tests" begin
    L = 4
    g = build_square_lattice(L)
    net = GNNFQNS(g)
    ψ = GNNWavefunction(g, net)

    B = 4
    σb = ones(Int, B, g.N)          # all spins up
    Jb = ones(Float32, B, length(g.edges))

    E = local_energy_batch(ψ, σb, Jb)

    # For S^z S^z Heisenberg diagonal part:
    # all-up ⇒ each bond gives +1
   
    diag_energy = length(g.edges) * 0.25
    @test all(abs.(E .- diag_energy) .< 1e-3)


end
