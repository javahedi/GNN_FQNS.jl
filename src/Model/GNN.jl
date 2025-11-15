##############################################
# Zygote-safe, mutation-free GNN for FQNS
##############################################

module GNN

using Flux
using Flux: glorot_uniform, gelu
using ..Graph

export GNNFQNS, gnn_forward, logpsi_gnn, logpsi_gnn_single

# ------------------------------------------
# Safe normalization
# ------------------------------------------
_norm(x) = x ./ (sqrt(sum(abs2, x)) + 1f-6)


# ------------------------------------------
# Model struct
# ------------------------------------------
struct GNNFQNS
    graph::LatticeGraph
    d_node::Int
    d_edge::Int
    d_hidden::Int
    K::Int
    embed_spin
    embed_J
    msg_net
    upd_net
    readout_amp
    readout_phase
    use_phase::Bool
end


# ------------------------------------------
# Constructor
# ------------------------------------------
function GNNFQNS(
        graph::LatticeGraph;
        d_node=32, d_edge=8, d_hidden=64,
        K=3, use_phase=true
    )

    embed_spin = Dense(1, d_node, gelu; init=glorot_uniform)
    embed_J    = Dense(1, d_edge, gelu; init=glorot_uniform)

    msg_net = Chain(
        Dense(d_node*2 + d_edge, d_hidden, gelu; init=glorot_uniform),
        Dense(d_hidden, d_hidden, gelu; init=glorot_uniform)
    )

    upd_net = Chain(
        Dense(d_node + d_hidden, d_hidden, gelu; init=glorot_uniform),
        Dense(d_hidden, d_node; init=glorot_uniform)
    )

    readout_amp = Chain(
        Dense(d_node, d_hidden, gelu; init=glorot_uniform),
        Dense(d_hidden, 1; init=glorot_uniform)
    )

    readout_phase = use_phase ? Chain(
        Dense(d_node, d_hidden, gelu; init=glorot_uniform),
        Dense(d_hidden, 1; init=glorot_uniform)
    ) : nothing

    return GNNFQNS(graph, d_node, d_edge, d_hidden, K,
                   embed_spin, embed_J, msg_net, upd_net,
                   readout_amp, readout_phase, use_phase)
end


# ------------------------------------------------------
# Embed spins: returns B × N × feature vectors
# ------------------------------------------------------
function embed_spins(model::GNNFQNS, σ_batch)
    B, N = size(σ_batch)
    return [
        [
            _norm(model.embed_spin(Float32[σ_batch[b,i]]))
            for i in 1:N
        ]
        for b in 1:B
    ]
end


edge_emb(model, Jbe) = _norm(model.embed_J(Float32[Jbe]))


# ------------------------------------------------------
# One message-passing layer
# ------------------------------------------------------
function message_layer(model::GNNFQNS, h, σ_batch, J_batch)
    g = model.graph
    B = length(h)
    N = length(h[1])

    return [
        [
            begin
                # collect msgs from neighbors
                msgs = map(e -> begin
                        (u,v) = g.edges[e]

                        if u == i
                            ee = edge_emb(model, J_batch[b,e])
                            model.msg_net(vcat(h[b][v], h[b][u], ee))
                        elseif v == i
                            ee = edge_emb(model, J_batch[b,e])
                            model.msg_net(vcat(h[b][u], h[b][v], ee))
                        else
                            zeros(Float32, model.d_hidden)
                        end
                    end, 1:length(g.edges))

                msum = reduce(+, msgs)
                upd  = model.upd_net(vcat(h[b][i], msum))
                _norm(h[b][i] .+ upd)
            end
            for i in 1:N
        ]
        for b in 1:B
    ]
end


# ------------------------------------------------------
# Full forward pass
# ------------------------------------------------------
function gnn_forward(model::GNNFQNS,
                     σ_batch::Array{<:Integer,2},
                     J_batch)

    h = embed_spins(model, σ_batch)

    for layer in 1:model.K
        h = message_layer(model, h, σ_batch, J_batch)
    end

    return h
end


# ------------------------------------------------------
# logpsi (batched)
# ------------------------------------------------------
function logpsi_gnn(model::GNNFQNS,
                    σ_batch::Array{<:Integer,2},
                    J_batch)

    h = gnn_forward(model, σ_batch, J_batch)
    B = length(h)
    N = length(h[1])

    pooled = [
        reduce(+, h[b]) ./ N
        for b in 1:B
    ]

    amps = map(b -> model.readout_amp(pooled[b])[1], 1:B)
    phases = model.use_phase ?
        map(b -> model.readout_phase(pooled[b])[1], 1:B) :
        fill(0f0, B)

    return amps .+ im .* phases
end


# ------------------------------------------------------
# logpsi (single)
# ------------------------------------------------------
function logpsi_gnn_single(model::GNNFQNS,
                           graph::LatticeGraph,
                           σ::Vector{Int},
                           J::Vector)

    σb = reshape(σ,1,:)
    Jb = reshape(J,1,:)
    return logpsi_gnn(model, σb, Jb)[1]
end


end # module
