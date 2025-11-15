############################ src/Utils/Plotting.jl ###############################
# Visualization & Plotting Tools for FQNS
#
# Provides:
#   • plot_energy_curve(losses)
#   • plot_magnetization(m, mAF)
#   • plot_structure_factor(Sq_dict)
#   • plot_correlation(C)
#   • plot_disorder_stats(energies)
#
# Default backend: CairoMakie (best-quality static plots)
##################################################################################

module Plotting

using CairoMakie
using Statistics

export plot_energy_curve,
       plot_magnetization,
       plot_correlation,
       plot_structure_factor,
       plot_structure_factor2D,
       plot_disorder_stats

###############################################################################
# 1. Energy vs Epoch
###############################################################################

"""
    plot_energy_curve(losses)

Plot disorder-averaged energy over training epochs.

Input:
    losses : Vector of Float32 (from train_disorder!)
"""
function plot_energy_curve(losses)
    fig = Figure()
    ax = Axis(fig[1,1],
              xlabel="Epoch",
              ylabel="⟨E⟩ disorder-avg",
              title="Energy vs Epoch")

    epochs = 1:length(losses)
    lines!(ax, epochs, losses)

    return fig
end


###############################################################################
# 2. Magnetization (FM + AFM)
###############################################################################

"""
    plot_magnetization(m, mAF)

Plot FM and AFM magnetization distributions for B samples.

Inputs:
    m     : FM magnetization vector (B)
    mAF   : AFM (Neel) order vector (B)
"""
function plot_magnetization(m, mAF)
    fig = Figure(resolution=(800,400))

    # FM histogram
    ax1 = Axis(fig[1,1],
               xlabel="m",
               ylabel="Count",
               title="FM Magnetization")

    hist!(ax1, m, bins=40, color=:dodgerblue)

    # AFM histogram
    ax2 = Axis(fig[1,2],
               xlabel="m_AF",
               ylabel="Count",
               title="AFM Magnetization")

    hist!(ax2, mAF, bins=40, color=:crimson)

    return fig
end


###############################################################################
# 3. Correlation Function C(r)
###############################################################################

"""
    plot_correlation(C)

Plot C(r) vs r.
"""
function plot_correlation(C)
    fig = Figure()
    ax = Axis(fig[1,1],
              xlabel="r",
              ylabel="C(r)",
              title="Spin Correlation Function")

    rs = 1:length(C)
    lines!(ax, rs, C, linewidth=2)

    return fig
end


###############################################################################
# 4. Structure Factor (1D and 2D)
###############################################################################

"""
    plot_structure_factor(Sq_dict)

Plot S(q) for selected momenta.

Sq_dict: Dict where keys are tuples (qx,qy) and values are S(q)
"""
function plot_structure_factor(Sq_dict::Dict)
    fig = Figure()
    ax = Axis(fig[1,1],
              xlabel="Momentum index",
              ylabel="S(q)",
              title="Structure Factor")

    labels = []
    values = Float32[]
    for (q, S) in Sq_dict
        push!(labels, string(q))
        push!(values, S)
    end

    barplot!(ax, values, color=:seagreen)
    ax.xticks = (1:length(labels), labels)

    return fig
end


"""
    plot_structure_factor2D(Sqq, L)

Plot S(qx,qy) as heatmap for qx,qy on a square lattice.
"""
function plot_structure_factor2D(Sqq, L)
    fig = Figure()
    ax = Axis(fig[1,1],
              xlabel="q_x index",
              ylabel="q_y index",
              title="Structure Factor S(q)")

    heatmap!(ax, 1:L, 1:L, Sqq, colormap=:viridis)

    return fig
end


###############################################################################
# 5. Disorder statistics: histogram of energies
###############################################################################

"""
    plot_disorder_stats(energies)

Plot histogram of disorder-averaged energies over R realizations.
"""
function plot_disorder_stats(energies)
    fig = Figure()
    ax = Axis(fig[1,1],
              xlabel="Energy",
              ylabel="Count",
              title="Disorder Energy Distribution")

    hist!(ax, energies, bins=40, color=:purple)

    return fig
end

end # module Plotting
