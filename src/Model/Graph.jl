############################### src/Model/Graph.jl ###############################
# Extended graph utilities for GNN-FQNS
#
# Adds:
#   • neighbors_edges[i] = list of edge indices touching node i
#   • consistent edge ordering
#   • efficient structure for J(edge) vector representation
###################################################################################

module Graph

export LatticeGraph,
       build_square_lattice,
       coord_to_index,
       index_to_coord,
       num_sites,
       num_edges,
       degree,
       edge_sites

################################################################################
# 1. LatticeGraph Structure
################################################################################

"""
    LatticeGraph

Graph representation of a 2D (or generic) spin lattice.

Fields
------
- N::Int                            number of sites
- L::Int                            linear size (for square lattices)
- edges::Vector{Tuple{Int,Int}}     list of edges (i,j) with i<j
- neighbors::Vector{Vector{Int}}    adjacency list
- neighbors_edges::Vector{Vector{Int}}
      neighbors_edges[i] = [e1, e4, ...] edge indices incident on i
- pbc::Bool                         periodic boundaries
"""
struct LatticeGraph
    N::Int
    L::Int
    edges::Vector{Tuple{Int,Int}}
    neighbors::Vector{Vector{Int}}
    neighbors_edges::Vector{Vector{Int}}
    pbc::Bool
end


################################################################################
# 2. Coordinate ↔ Index conversion
################################################################################

@inline function coord_to_index(i::Int, j::Int, L::Int)
    @assert 1 ≤ i ≤ L
    @assert 1 ≤ j ≤ L
    return (i - 1) * L + j
end

@inline function index_to_coord(s::Int, L::Int)
    @assert 1 ≤ s ≤ L*L
    i = div(s - 1, L) + 1
    j = (s - 1) % L + 1
    return i, j
end


################################################################################
# 3. Build 2D square lattice with PBC
################################################################################

"""
    build_square_lattice(L; pbc=true)

Return an `L×L` square-lattice graph with edges stored in canonical
order and with `neighbors_edges` for fast message passing.

J couplings will be stored as a vector of length `nedges` aligned with
the edge ordering of `graph.edges`.
"""
function build_square_lattice(L::Int; pbc::Bool = true)
    @assert L ≥ 2 "L must be ≥ 2"

    N = L * L
    edges = Tuple{Int,Int}[]
    neighbors = [Int[] for _ in 1:N]

    # We'll accumulate neighbors_edges after building edges
    # but we need edge count
    tmp_neighbors_edges = [Int[] for _ in 1:N]

    # Helper to add edges in canonical order
    function add_edge!(i::Int, j::Int)
        if i == j
            return
        end
        a, b = i < j ? (i, j) : (j, i)
        push!(edges, (a, b))
        e = length(edges)

        # adjacency list
        push!(neighbors[a], b)
        push!(neighbors[b], a)

        # edge index list
        push!(tmp_neighbors_edges[a], e)
        push!(tmp_neighbors_edges[b], e)
    end

    # Generate edges
    for i in 1:L, j in 1:L
        s = coord_to_index(i, j, L)

        # right neighbor
        if j < L
            add_edge!(s, coord_to_index(i, j+1, L))
        elseif pbc
            add_edge!(s, coord_to_index(i, 1, L))
        end

        # down neighbor
        if i < L
            add_edge!(s, coord_to_index(i+1, j, L))
        elseif pbc
            add_edge!(s, coord_to_index(1, j, L))
        end
    end

    return LatticeGraph(
        N,
        L,
        edges,
        neighbors,
        tmp_neighbors_edges,
        pbc
    )
end


################################################################################
# 4. Convenience Helpers
################################################################################

num_sites(g::LatticeGraph) = g.N
num_edges(g::LatticeGraph) = length(g.edges)
degree(g::LatticeGraph, i::Int) = length(g.neighbors[i])

"""
    edge_sites(graph, e)

Return sites (i, j) for edge index e.
"""
@inline function edge_sites(g::LatticeGraph, e::Int)
    return g.edges[e]
end

end # module Graph
