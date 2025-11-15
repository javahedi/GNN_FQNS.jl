using Test
using Random
using GNN_FQNS

@testset "Disorder Phase Test (p=0.2 vs p=0.8)" begin
    Random.seed!(123)

    # Build small system
    L = 4
    g = build_square_lattice(L)

    # ±J disorder generator: P(+1)=p, P(-1)=1-p
    generate_pmJ(graph; p) = Float32[
        rand() < p ? 1f0 : -1f0
        for _ in 1:length(graph.edges)
    ]

    """
        run_phase_measure(g; p)

    For a given p:
      * build a fresh GNN-FQNS ψ
      * do a tiny bit of SR training with ±J disorder
      * run short MCMC
      * return (m_FM, m_AF)
    """
    function run_phase_measure(g; p)
        # New model per p
        net = GNNFQNS(g)
        ψ   = GNNWavefunction(g, net)

        disorder_fn = () -> generate_pmJ(g; p=p)

        # ----- VERY LIGHT TRAINING (just exercise pipeline) -----
        history = train_disorder!(ψ, g;
            epochs      = 1,
            R           = 1,
            B           = 4,
            nsteps      = 5,
            disorder_fn = disorder_fn,
            η           = 0.02,
            diag_reg    = 1e-4,
            verbose     = false
        )

        @test :losses ∈ keys(history)
        @test length(history[:losses]) == 1
        @test isfinite(history[:losses][1])

        # ----- SHORT MCMC SAMPLING -----
        Bsample       = 8
        nsteps_sample = 10

        J       = disorder_fn()
        J_batch = repeat(J', Bsample, 1)

        σ_batch, _ = sample_batch(
            ψ,
            g,
            Bsample,
            nsteps_sample,
            J_batch
        )

        mFM = mean(magnetization(σ_batch))
        mAF = mean(neel_order(σ_batch, g))

        return mFM, mAF
    end

    # Case 1: p = 0.2
    mFM_02, mAF_02 = run_phase_measure(g; p=0.2)

    # Case 2: p = 0.8
    mFM_08, mAF_08 = run_phase_measure(g; p=0.8)

    # -------- Sanity checks only (no physics inequality!) --------

    # Finite values
    @test isfinite(mFM_02)
    @test isfinite(mAF_02)
    @test isfinite(mFM_08)
    @test isfinite(mAF_08)

    # Magnetizations must be in [-1, 1]
    @test abs(mFM_02) ≤ 1f0
    @test abs(mFM_08) ≤ 1f0
    @test abs(mAF_02) ≤ 1f0
    @test abs(mAF_08) ≤ 1f0

    # the two parameter points should give *different* observables
    @test abs(mFM_02 - mFM_08) > 1f-6
    @test abs(mAF_02 - mAF_08) > 1f-6
end
