import torch
from torch import nn
from torch.nn import functional as F


class GraphModel(torch.nn.Module):
    def __init__(self, gnn_type, num_layers, dim0, h_dim, out_dim, last_layer_fully_adjacent,
                 unroll, layer_norm, use_activation, use_residual):
        super(GraphModel, self).__init__()
        self.gnn_type = gnn_type
        self.unroll = unroll
        self.last_layer_fully_adjacent = last_layer_fully_adjacent
        self.use_layer_norm = layer_norm
        self.use_activation = use_activation
        self.use_residual = use_residual
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_layers = num_layers
        self.layer0_keys = nn.Embedding(num_embeddings=dim0 + 1, embedding_dim=h_dim)
        self.layer0_values = nn.Embedding(num_embeddings=dim0 + 1, embedding_dim=h_dim)
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        if unroll:
            self.layers.append(gnn_type.get_layer(
                in_dim=h_dim,
                out_dim=h_dim))
        else:
            for i in range(num_layers):
                self.layers.append(gnn_type.get_layer(
                    in_dim=h_dim,
                    out_dim=h_dim))
        if self.use_layer_norm:
            for i in range(num_layers):
                self.layer_norms.append(nn.LayerNorm(h_dim))

        self.out_dim = out_dim
        # self.out_layer = nn.Linear(in_features=h_dim, out_features=out_dim, bias=False)
        self.out_layer = nn.Linear(in_features=h_dim, out_features=out_dim + 1, bias=False)

    def forward(self, data):
        x, edge_index, batch, roots = data.x, data.edge_index, data.batch, data.root_mask

        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed

        for i in range(self.num_layers):
            if self.unroll:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            new_x = x
            if self.last_layer_fully_adjacent and i == self.num_layers - 1:
                root_indices = torch.nonzero(roots, as_tuple=False).squeeze(-1)
                target_roots = root_indices.index_select(dim=0, index=batch)
                source_nodes = torch.arange(0, data.num_nodes).to(self.device)
                edges = torch.stack([source_nodes, target_roots], dim=0)

            else:
                edges = edge_index
            new_x = layer(new_x, edges)
            if self.use_activation:
                new_x = F.relu(new_x)
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        root_nodes = x[roots]
        logits = self.out_layer(root_nodes)
        # logits = F.linear(root_nodes, self.layer0_values.weight)
        return logits
