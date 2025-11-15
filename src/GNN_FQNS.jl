module GNN_FQNS

# Model
include("Model/Graph.jl")
include("Model/GNN.jl")
include("Model/Wavefunction.jl")

# Physics
include("Physics/Heisenberg.jl")
include("Physics/Observables.jl")
include("Physics/Disorder.jl")

# Sampler
include("Sampler/MCMC.jl")

# Optimizer
include("Optim/SR.jl")

# Training
include("Training/Train.jl")

# Utils
include("Utils/Plotting.jl")
#include("Utils/Device.jl")



using .Graph
using .GNN
using .Wavefunction
using .Heisenberg
using .Observables
using .Disorder
using .MCMC
using .SR
using .Train
using .Plotting
#using .Device



export
        Train,

        # Graph
        LatticeGraph,
        build_square_lattice,
        coord_to_index,
        index_to_coord,
        num_sites,
        num_edges,
        degree,
        edge_sites,

        # GNN Model (public)
        GNNFQNS,
        gnn_forward,
        logpsi_gnn,

        # Wavefunction (public API)
        GNNWavefunction,
        logpsi,
        logpsi_batch,

        # Sampler
        MCMCSampler,
        init_states,
        step!,
        sample_batch,

        # Physics
        local_energy_batch,

        # Observables
        magnetization,
        neel_order,
        correlation_function,
        structure_factor,
        structure_factor_afm,
        structure_factor_fm,

        # SR
        compute_logderivs,
        sr_step!,

        # Training
        train_epoch!,
        train_disorder!,

        # Plotting
        plot_energy_curve,
        plot_magnetization,
        plot_correlation,
        plot_structure_factor,
        plot_structure_factor2D,
        plot_disorder_stats,

        # Device
        #to_device, isgpu

        # Disorder
        generate_pmJ


end # module
