using Test
using Statistics       # ← REQUIRED
using GNN_FQNS

@testset "Observables" begin
    g = build_square_lattice(4)

    B = 3
    σb = ones(Int, B, g.N)

    m = magnetization(σb)
    @test all(m .≈ 1)

    mAF = neel_order(σb, g)

    # AFM order for all-up ≈ 0
    @test abs(mean(mAF)) < 1e-6

    S0 = structure_factor_fm(σb, g)

    # Perfect FM: S(0,0) = N
    @test abs(S0 - g.N) < 1e-6
end
