using Test
using GNN_FQNS

@testset "Graph Tests" begin
    L = 4
    g = build_square_lattice(L; pbc=true)

    @test g.N == L^2
    @test length(g.edges) == 2 * L^2       # 2 per site in PBC 2D
    @test length(g.neighbors) == g.N
    @test length(g.neighbors_edges) == g.N

    # Check that each edge index is consistent
    for e in 1:length(g.edges)
        i, j = g.edges[e]
        @test i < j
        @test e in g.neighbors_edges[i]
        @test e in g.neighbors_edges[j]
    end

    # Check coordinate mapping round trip
    for s in 1:g.N
        i, j = index_to_coord(s, L)
        @test coord_to_index(i, j, L) == s
    end
end
