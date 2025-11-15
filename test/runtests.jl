using Test

@testset "GNN_FQNS Test Suite" begin
    include("test_graph.jl")
    include("test_gnn.jl")
    include("test_wavefunction.jl")
    include("test_mcmc.jl")
    include("test_local_energy.jl")
    include("test_observables.jl")
    include("test_sr.jl")
    include("test_training.jl")
end
