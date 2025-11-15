module Observables

using ..Graph
using LinearAlgebra
using Statistics

export magnetization,
       neel_order,
       correlation_function,
       structure_factor,
       structure_factor_afm,
       structure_factor_fm

###############################################################################
# 1. Ferromagnetic magnetization
###############################################################################

function magnetization(σ_batch)
    B, N = size(σ_batch)
    return vec(sum(Float32.(σ_batch), dims=2) ./ N)
end

###############################################################################
# 2. Néel staggered order m_AF  (π,π)
###############################################################################

function neel_order(σ_batch, graph::LatticeGraph)
    B, N = size(σ_batch)
    L = graph.L

    stag = zeros(Float32, N)

    # correct alternating pattern (-1)^{x+y} with 0-based coordinates
    for s in 1:N
        i, j = index_to_coord(s, L)
        stag[s] = (-1) ^ ((i - 1) + (j - 1))
    end

    # compute m_AF = (1/N) Σ σ_i * stag_i
    m = (Float32.(σ_batch) * stag) ./ N
    return vec(m)
end

###############################################################################
# 3. Real-space correlation C(r)
###############################################################################

function correlation_function(σ_batch, graph::LatticeGraph)
    B, N = size(σ_batch)
    L = graph.L
    max_r = div(L, 2)

    σ = Float32.(σ_batch)
    C = zeros(Float32, max_r)

    for r in 1:max_r
        vals = Float32[]
        for s in 1:N
            i, j = index_to_coord(s, L)
            j2 = ((j + r - 1) % L) + 1
            s2 = coord_to_index(i, j2, L)

            for b in 1:B
                push!(vals, σ[b,s] * σ[b,s2])
            end
        end
        C[r] = mean(vals)
    end

    return C
end

###############################################################################
# 4. Structure factor S(q)
###############################################################################

function structure_factor(σ_batch,
                          graph::LatticeGraph,
                          qx::Float64,
                          qy::Float64)

    B, N = size(σ_batch)
    L = graph.L

    σ = Float32.(σ_batch)

    # site coordinates (0-based)
    xs = zeros(Float64, N)
    ys = zeros(Float64, N)
    for s in 1:N
        i, j = index_to_coord(s, L)
        xs[s] = i - 1
        ys[s] = j - 1
    end

    # correlation matrix <σ_i σ_j>
    C = (σ' * σ) ./ B   # N×N

    S = 0.0
    for i in 1:N, j in 1:N
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        S += C[i,j] * cos(qx * dx + qy * dy)
    end

    return S / N
end

###############################################################################
# 5. Special q-points
###############################################################################

structure_factor_afm(σ_batch, graph) = structure_factor(σ_batch, graph, π,  π)
structure_factor_fm( σ_batch, graph) = structure_factor(σ_batch, graph, 0.0, 0.0)

end # module
