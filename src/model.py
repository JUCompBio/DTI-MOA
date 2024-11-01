import urllib.request
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_KQV(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm1(q), self.norm2(k), self.norm3(v), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.norm(self.fn(x))


class PostNorm_(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.norm(self.fn(q, k, v, **kwargs)[0])


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MultiLayerTransformer(nn.Module):
    def __init__(self, patch_dim, num_patches, dim, depth, heads, mlp_dim, pool="mean", dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        return x


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
    num_bond_direction = 3

    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault("aggr", aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(self.num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(self.num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations
    """

    num_atom_type = 120  # including the extra mask tokens
    num_chirality_tag = 3

    def __init__(self, num_layer=5, emb_dim=300, JK="last", drop_ratio=0, gnn_type="gin", pretrained=True):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(self.num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(self.num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add"))
            else:
                raise NotImplementedError("Only using GINConv layer.")
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        if pretrained:
            urllib.request.urlretrieve("https://github.com/junxia97/Mole-BERT/raw/main/model_gin/Mole-BERT.pth", "Mole-BERT.pth")
            self.load_state_dict(torch.load("Mole-BERT.pth", map_location="cpu"))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.load_state_dict(torch.load(model_file))


class FFLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, act: str = "ReLU", bn: str = "LayerNorm"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = getattr(nn, act)()
        self.bn = getattr(nn, bn)(out_features)

    def forward(self, x):
        return self.bn(self.act(self.linear(x)))


class DTI_MOA_Model(nn.Module):
    def __init__(self, num_classes, drug_enc_dim, drug_graph_dim, drug_avg_len, target_enc_dim, target_dssp_dim, target_avg_len):
        super().__init__()
        self._dim = 512
        self.linear1_1 = FFLayer(drug_enc_dim, self._dim)
        self.linear1_2 = FFLayer(drug_graph_dim, drug_graph_dim)
        self.linear1_3 = FFLayer(target_enc_dim, self._dim)
        self.linear1_4 = Transformer(target_dssp_dim, 1, 8, 64, target_dssp_dim)
        self.linear2_1 = FFLayer(self._dim + drug_graph_dim, self._dim)
        self.linear2_2 = FFLayer(self._dim + target_dssp_dim, self._dim)
        # WRX-Attn
        self.smiles_attn1 = PreNorm_KQV(self._dim, nn.MultiheadAttention(self._dim, 8, batch_first=True))
        self.smiles_attn1_ln = PreNorm(self._dim, FeedForward(self._dim, self._dim))
        self.smiles_alpha1 = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.prot_attn1 = PreNorm_KQV(self._dim, nn.MultiheadAttention(self._dim, 8, batch_first=True))
        self.prot_attn1_ln = PreNorm(self._dim, FeedForward(self._dim, self._dim))
        self.prot_alpha1 = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.smiles_attn2 = PreNorm_KQV(self._dim, nn.MultiheadAttention(self._dim, 8, batch_first=True))
        self.smiles_attn2_ln = PreNorm(self._dim, FeedForward(self._dim, self._dim))
        self.smiles_alpha2 = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.prot_attn2 = PreNorm_KQV(self._dim, nn.MultiheadAttention(self._dim, 8, batch_first=True))
        self.prot_attn2_ln = PreNorm(self._dim, FeedForward(self._dim, self._dim))
        self.prot_alpha2 = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.prot_pool = nn.AdaptiveAvgPool2d((drug_avg_len, self._dim))
        self.linear3 = FFLayer(self._dim, self._dim)
        self.linear4 = FFLayer(self._dim, self._dim)
        self.mlt = MultiLayerTransformer(self._dim, 2 * drug_avg_len, self._dim, 2, 8, self._dim)
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.linear5 = FFLayer(self._dim, self._dim)
        self.linear6 = nn.Linear(self._dim, num_classes if num_classes > 2 else 1)

    def forward(self, x1, x2, x3, x4):
        """
        Input: Drug Enc, Drug Graph, Target Enc, Target DSSP
        """
        x1 = self.linear1_1(x1)
        x2 = self.linear1_2(x2)
        x3 = self.linear1_3(x3)
        x4 = self.linear1_4(x4)
        x1 = torch.cat([x1, x2], dim=-1)
        x2 = torch.cat([x3, x4], dim=-1)
        x1 = self.linear2_1(x1)
        x2 = self.linear2_2(x2)

        x3 = self.smiles_attn1(x1, x2, x2)[0] + self.smiles_alpha1 * x1
        x3 = self.smiles_attn1_ln(x3) + x3
        x4 = self.prot_attn1(x2, x1, x1)[0] + self.prot_alpha1 * x2
        x4 = self.prot_attn1_ln(x4) + x4

        x1 = self.smiles_attn2(x3, x4, x4)[0] + self.smiles_alpha2 * x3
        x1 = self.smiles_attn2_ln(x1) + x1
        x2 = self.prot_attn2(x4, x3, x3)[0] + self.prot_alpha2 * x4
        x4 = self.prot_attn2_ln(x4) + x4
        x2 = self.prot_pool(x2)
        x2 = self.linear3(x2)
        x = torch.concat([x1, x2], dim=1)

        x = self.linear4(x)
        x = self.mlt(x)
        x = self.linear5(x)
        x = self.linear6(x)
        return x
