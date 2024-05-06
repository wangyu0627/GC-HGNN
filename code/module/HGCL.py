import dgl
import numpy as np
import dgl.function as fn
from dgl.nn.pytorch import GATConv, GraphConv, HeteroGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DropEdge
from .autocoder import Autoencoder
from .contrast import Contrast


def InfoNCE(view1, view2, temperature=0.5):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def labor_sample(g, fanout, edge_dir):
    #  {'drug':[0, 1]}
    num_nodes = {}
    for ntype in g.ntypes:
        num_node = g.num_nodes(ntype)
        num_nodes[ntype] = [i for i in range(num_node)]
    sg = dgl.sampling.sample_labors(g, num_nodes, fanout=fanout, edge_dir=edge_dir)
    # sg = dgl.sampling.sample_neighbors(g, num_nodes, 3)
    return sg

# Semantic attention in the metapath-based aggregation (the same as that in the HAN)
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


# Metapath-based aggregation (the same as the HANLayer)
class HANLayer(nn.Module):
    def __init__(
        self, num_meta_paths, in_size, out_size, encoder, decoder, num_heads, feat_drop,
            attn_drop, enc_num_layer, dec_num_layer, mask_rate, remask_rate, num_remasking
    ):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.layers.append(
                # GATConv(64, 64, num_heads, 0.1, 0.1, activation=F.elu)
                # GraphConv(in_size, out_size, weight=False, bias=False, allow_zero_in_degree=True)
                Autoencoder(in_dim=in_size, hidden_dim=out_size, encoder=encoder,decoder=decoder,
                            feat_drop=feat_drop, attn_drop=attn_drop, enc_num_layer=enc_num_layer,
                            dec_num_layer=dec_num_layer, num_heads=num_heads, mask_rate=mask_rate,
                            remask_rate=remask_rate, num_remasking=num_remasking)
            )

    def forward(self, gs, h):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.layers[i](g, h).flatten(1))
        # semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        return semantic_embeddings # (N, D * K)


# Relational neighbor aggregation
class RelationalAGG(nn.Module):
    def __init__(self, g, in_size, out_size, dropout=0.1):
        super(RelationalAGG, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        # Transform weights for different types of edges
        self.W_T = nn.ModuleDict(
            {str(name): nn.Linear(in_size, out_size, bias=False)
            for name in g.etypes})

        # Attention weights for different types of edges
        self.W_A = nn.ModuleDict(
            {str(name): nn.Linear(out_size, 1, bias=False) for name in g.etypes})

        # layernorm
        self.layernorm = nn.LayerNorm(out_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.nodes[dsttype].data["h"] = feat_dict[dsttype]
            g.nodes[srctype].data["h"] = feat_dict[srctype]  # nodes' original feature
            g.nodes[srctype].data["t_h"] = self.W_T[etype](feat_dict[srctype])  # src nodes' transformed feature

            # compute the attention numerator (exp)
            g.apply_edges(fn.u_mul_v("t_h", "h", "x"), etype=etype)
            g.edges[etype].data["x"] = torch.exp(self.W_A[etype](g.edges[etype].data["x"]))

            # first update to compute the attention denominator (\sum exp)
            funcs[etype] = (fn.copy_e("x", "m"), fn.sum("m", "att"))
        g.multi_update_all(funcs, "sum")

        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.apply_edges(
                fn.e_div_v("x", "att", "att"), etype=etype
            )  # compute attention weights (numerator/denominator)
            funcs[etype] = (
                fn.u_mul_e("h", "att", "m"),
                fn.sum("m", "h"),
            )  # \sum(h0*att) -> h1
        # second update to obtain h1
        g.multi_update_all(funcs, "sum")

        # apply activation, layernorm, and dropout
        feat_dict = {}
        for ntype in g.ntypes:
            feat_dict[ntype] = self.dropout(
                self.layernorm(F.elu_(g.nodes[ntype].data["h"]))
            )  # apply activation, layernorm, and dropout

        return feat_dict

# class RelationalAGG(nn.Module):
#     def __init__(self, g, in_size, out_size, dropout=0.1):
#         super(RelationalAGG, self).__init__()
#         layers = {etype: GATConv(in_size, out_size, num_heads=1, feat_drop=dropout,
#                                  attn_drop=dropout, activation=F.elu) for etype in g.etypes}
#         self.hetero_gat_conv = HeteroGraphConv(layers, aggregate='sum')
#
#     def forward(self, g, feat_dict):
#         h = self.hetero_gat_conv(g, feat_dict)
#         h = {ntype: torch.flatten(data, start_dim=1)
#                           for ntype, data in h.items()}
#         return h

class HGCL(nn.Module):
    def __init__(
        self, g, gs, fanout, edge_dir, num_meta_paths, in_size, hidden_size, out_size, 
        encoder, decoder, num_heads, rela_dropout, feat_drop, attn_drop, enc_num_layer, 
        dec_num_layer, mask_rate, remask_rate, num_remasking, alpha, beta, tau, lam):
        super(HGCL, self).__init__()
        # hg and mpg processed
        # g = labor_sample(g, fanout, edge_dir)[0]
        self.g = g
        self.gs = gs

        # dim
        self.adapt_ws = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_size) for ntype, in_dim in in_size.items()})

        # relational neighbor aggregation, this produces h1
        self.RelationalAGG = RelationalAGG(g, hidden_size, hidden_size, rela_dropout)

        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleList()
        self.hans.append(
            HANLayer(num_meta_paths, hidden_size, out_size, encoder, decoder, num_heads,
                     feat_drop, attn_drop, enc_num_layer, dec_num_layer, mask_rate,
                     remask_rate, num_remasking)
        )
        # meta-path fusion
        self.semantic_attention = SemanticAttention(in_size=out_size, hidden_size=out_size)
        # self.attention = Attention(hidden_size, attn_drop)
        # contrast loss
        self.contrast = Contrast(hidden_size, tau, lam)
        # loss weight
        self.alpha = alpha
        self.beta = beta

    def forward(self, feats, pos, key):
        # feats_To_dim64
        h = {ntype: F.gelu(self.adapt_ws[ntype](feats[ntype])) for ntype in feats}

        # relational neighbor aggregation, h1
        h1 = self.RelationalAGG(self.g, h)

        # metapath-based aggregation, h2
        for gnn in self.hans:
            h2 = gnn(self.gs, h[key])

        # intra-contrast
        intra_pos = get_intra_pos(pos[0])
        # 2 meta-path
        if len(self.gs) == 2:
            intra_loss = sce_loss(h2[0][intra_pos], h2[1][intra_pos])
            # inter_loss = InfoNCE(h2[0], h2[1], 1)
        # 3 meta-path
        else:
            loss1 = sce_loss(h2[0][intra_pos], h2[1][intra_pos])
            loss2 = sce_loss(h2[0][intra_pos], h2[2][intra_pos])
            loss3 = sce_loss(h2[1][intra_pos], h2[2][intra_pos])
            intra_loss = loss1 + loss2 + loss3
        semantic_h = torch.stack(h2, dim=1)
        # meta-path fusion
        h2 = self.semantic_attention(semantic_h)
        
        # intra-contrast
        inter_loss = self.contrast(h1[key], h2, pos)
        loss = self.alpha * inter_loss + self.beta * intra_loss
        return loss



    def get_embeds(self, feats, key):
        z_mp = F.elu(self.adapt_ws[key](feats[key]))
        for gnn in self.hans:
            z_mp = gnn(self.gs, z_mp)
        z_mp = torch.stack(z_mp, dim=1)
        z_mp = self.semantic_attention(z_mp)
        return z_mp.detach()

def get_intra_pos(pos, k_percent=0.5):
    pos = pos.to_dense()
    # 计算每个节点的度
    degree = torch.sum(pos, dim=1)

    # 计算要选择的节点数量
    k = int(pos.size(0) * k_percent)

    # 找出度最大的前k个节点的索引
    _, top_k_indices = torch.topk(degree, k)

    return top_k_indices