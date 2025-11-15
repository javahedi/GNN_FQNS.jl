module Disorder

    using Random
    export generate_pmJ
    """
        generate_pmJ(graph; p)

    Generate ±1 disorder J[e] with:
    P(J_e = +1) = p
    P(J_e = -1) = 1 - p
    """
    function generate_pmJ(graph; p)
        E = length(graph.edges)
        return Float32[ rand() < p ? 1f0 : -1f0 for _ in 1:E ]
    end

    
end