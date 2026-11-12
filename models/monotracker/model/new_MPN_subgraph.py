from copy import deepcopy
import math

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max, scatter_add


class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        super(MLP, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), \
            f'fc_dims must be either a list or a tuple, but got {type(fc_dims)}'

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(nn.ReLU(inplace=True))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)


class EdgeModel(nn.Module):
    """
    Edge update model.
    """
    def __init__(self, edge_mlp):
        super(EdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1)
        return self.edge_mlp(out)


class MLPGraphIndependent(nn.Module):
    """
    Independent node / edge encoders or classifiers.
    """
    def __init__(self, edge_in_dim=None, node_in_dim=None, edge_out_dim=None, node_out_dim=None,
                 node_fc_dims=None, edge_fc_dims=None, dropout_p=None, use_batchnorm=None):
        super(MLPGraphIndependent, self).__init__()

        if node_in_dim is not None:
            self.node_mlp = MLP(
                input_dim=node_in_dim,
                fc_dims=list(node_fc_dims) + [node_out_dim],
                dropout_p=dropout_p,
                use_batchnorm=use_batchnorm,
            )
        else:
            self.node_mlp = None

        if edge_in_dim is not None:
            self.edge_mlp = MLP(
                input_dim=edge_in_dim,
                fc_dims=list(edge_fc_dims) + [edge_out_dim],
                dropout_p=dropout_p,
                use_batchnorm=use_batchnorm,
            )
        else:
            self.edge_mlp = None

    def forward(self, edge_feats=None, nodes_feats=None):
        out_node_feats = self.node_mlp(nodes_feats) if self.node_mlp is not None else nodes_feats
        out_edge_feats = self.edge_mlp(edge_feats) if self.edge_mlp is not None else edge_feats
        return out_edge_feats, out_node_feats


class DepthTemporalAttention(nn.Module):
    """
    Reused only to build the INITIAL edge feature of the depth subgraph.
    It returns a scalar edge cue from depth vectors + optional frame encoding.
    """
    def __init__(self, depth_dim, hidden_dim=128):
        super().__init__()
        self.depth_proj = nn.Linear(depth_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def positional_encoding(self, t, dim):
        device = t.device
        pe = torch.zeros(t.shape[0], dim, device=device)

        position = t.unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device).float() *
            (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x_depth, frame_id, edge_index):
        h = self.depth_proj(x_depth)

        if frame_id is not None:
            pe = self.positional_encoding(frame_id, self.hidden_dim)
            q = h + pe
            k = h + pe
        else:
            q = h
            k = h

        row, col = edge_index
        qi = q[row]
        kj = k[col]

        attn = (qi * kj).sum(dim=1) / math.sqrt(self.hidden_dim)
        return torch.sigmoid(attn)


class MetaLayer(torch.nn.Module):
    """
    Core message passing block with batch-aware node update.
    """
    def __init__(self, edge_model=None, node_model=None):
        super(MetaLayer, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr,
                num_prev_nodes=None, batch=None, ptr=None):
        row, col = edge_index

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        if self.node_model is not None:
            x = self.node_model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_prev_nodes=num_prev_nodes,
                batch=batch,
                ptr=ptr,
            )

        return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(
            self.__class__.__name__, self.edge_model, self.node_model
        )


class BipartiteNodeModel(nn.Module):
    def __init__(self, flow_in_mlp, flow_out_mlp, node_mlp, node_agg_fn):
        super(BipartiteNodeModel, self).__init__()

        self.flow_in_mlp = flow_in_mlp     # prev -> curr messages
        self.flow_out_mlp = flow_out_mlp   # curr -> prev messages
        self.node_mlp = node_mlp
        self.node_agg_fn = node_agg_fn

    @staticmethod
    def _to_prev_tensor(num_prev_nodes, device):
        if num_prev_nodes is None:
            raise AssertionError("num_prev_nodes must be provided for online bipartite update")

        if isinstance(num_prev_nodes, int):
            return torch.tensor([num_prev_nodes], dtype=torch.long, device=device)

        if isinstance(num_prev_nodes, list):
            return torch.tensor(num_prev_nodes, dtype=torch.long, device=device)

        if isinstance(num_prev_nodes, torch.Tensor):
            return num_prev_nodes.view(-1).to(device=device, dtype=torch.long)

        raise TypeError(f"Unsupported type for num_prev_nodes: {type(num_prev_nodes)}")

    @staticmethod
    def _build_prev_curr_masks(num_nodes, num_prev_nodes, ptr=None, device=None):
        device = device if device is not None else num_prev_nodes.device
        prev_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        if ptr is None:
            n_prev = int(num_prev_nodes[0].item())
            prev_mask[:n_prev] = True
            curr_mask = ~prev_mask
            return prev_mask, curr_mask

        ptr = ptr.to(device=device, dtype=torch.long).view(-1)
        assert ptr.numel() == num_prev_nodes.numel() + 1, \
            "For batched graphs, ptr should have length batch_size + 1."

        for g in range(num_prev_nodes.numel()):
            start = int(ptr[g].item())
            end = int(ptr[g + 1].item())
            n_prev = int(num_prev_nodes[g].item())
            assert 0 <= n_prev <= (end - start), \
                f"Invalid num_prev_nodes[{g}]={n_prev} for graph size {end-start}"
            prev_mask[start:start + n_prev] = True

        curr_mask = ~prev_mask
        return prev_mask, curr_mask

    def forward(self, x, edge_index, edge_attr, num_prev_nodes=None, batch=None, ptr=None):
        num_prev_nodes = self._to_prev_tensor(num_prev_nodes, x.device)
        prev_mask, curr_mask = self._build_prev_curr_masks(
            num_nodes=x.size(0),
            num_prev_nodes=num_prev_nodes,
            ptr=ptr,
            device=x.device,
        )

        row, col = edge_index  # row: prev, col: curr

        # curr -> prev
        msg_to_prev = torch.cat([x[col], edge_attr], dim=1)
        msg_to_prev = self.flow_out_mlp(msg_to_prev)
        agg_prev = self.node_agg_fn(msg_to_prev, row, x.size(0))

        # prev -> curr
        msg_to_curr = torch.cat([x[row], edge_attr], dim=1)
        msg_to_curr = self.flow_in_mlp(msg_to_curr)
        agg_curr = self.node_agg_fn(msg_to_curr, col, x.size(0))

        x_new = torch.zeros_like(x)

        if prev_mask.any():
            prev_input = torch.cat([x[prev_mask], agg_prev[prev_mask]], dim=1)
            x_new[prev_mask] = self.node_mlp(prev_input)

        if curr_mask.any():
            curr_input = torch.cat([x[curr_mask], agg_curr[curr_mask]], dim=1)
            x_new[curr_mask] = self.node_mlp(curr_input)

        return x_new


class MOTMPNet(nn.Module):
    def __init__(self, model_params):
        super(MOTMPNet, self).__init__()

        encoder_feats_dict = deepcopy(model_params['encoder_feats_dict'])
        classifier_feats_dict = model_params['classifier_feats_dict']

        self.encoder_feats_dict = encoder_feats_dict
        self.model_params = model_params

        # ----- Base graph -----
        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        self.classifier = MLPGraphIndependent(**classifier_feats_dict)
        self.MPNet = self._build_core_MPNet(
            model_params=model_params,
            encoder_feats_dict=encoder_feats_dict
        )

        # ----- Depth subgraph -----
        self.use_depth_subgraph = model_params.get('depth_dim', None) is not None
        # self.use_depth_subgraph = False
        if self.use_depth_subgraph:
            depth_dim = model_params['depth_dim']
            depth_attn_hidden_dim = model_params.get('depth_attn_hidden_dim', 128)

            # Old attention formula reused as initial depth-edge cue.
            self.depth_edge_builder = DepthTemporalAttention(
                depth_dim=depth_dim,
                hidden_dim=depth_attn_hidden_dim
            )

            # Depth subgraph encoder: node=x_depth, edge=initial depth scalar
            depth_encoder_feats_dict = {
                'edge_in_dim': 1,
                'node_in_dim': depth_dim,
                'edge_out_dim': encoder_feats_dict['edge_out_dim'],
                'node_out_dim': encoder_feats_dict['node_out_dim'],
                'node_fc_dims': encoder_feats_dict['node_fc_dims'],
                'edge_fc_dims': encoder_feats_dict['edge_fc_dims'],
                'dropout_p': encoder_feats_dict['dropout_p'],
                'use_batchnorm': encoder_feats_dict['use_batchnorm'],
            }
            self.depth_encoder = MLPGraphIndependent(**depth_encoder_feats_dict)
            self.depth_MPNet = self._build_core_MPNet(
                model_params=model_params,
                encoder_feats_dict=depth_encoder_feats_dict
            )

            edge_dim = encoder_feats_dict['edge_out_dim']

            # Depth subgraph -> scalar gate alpha_ij
            self.depth_gate = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(inplace=True),
                nn.Linear(edge_dim, 1),
            )

        self.num_enc_steps = model_params['num_enc_steps']
        self.num_class_steps = model_params['num_class_steps']

    def _build_core_MPNet(self, model_params, encoder_feats_dict):
        node_agg_fn = model_params['node_agg_fn']
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), \
            "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)
        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]
        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        self.reattach_initial_nodes = model_params['reattach_initial_nodes']
        self.reattach_initial_edges = model_params['reattach_initial_edges']

        edge_factor = 2 if self.reattach_initial_edges else 1
        node_factor = 2 if self.reattach_initial_nodes else 1

        edge_model_in_dim = (
            node_factor * 2 * encoder_feats_dict['node_out_dim']
            + edge_factor * encoder_feats_dict['edge_out_dim']
        )
        node_model_in_dim = (
            node_factor * encoder_feats_dict['node_out_dim']
            + encoder_feats_dict['edge_out_dim']
        )

        edge_model_feats_dict = model_params['edge_model_feats_dict']
        node_model_feats_dict = model_params['node_model_feats_dict']

        edge_mlp = MLP(
            input_dim=edge_model_in_dim,
            fc_dims=edge_model_feats_dict['fc_dims'],
            dropout_p=edge_model_feats_dict['dropout_p'],
            use_batchnorm=edge_model_feats_dict['use_batchnorm'],
        )

        flow_in_mlp = MLP(
            input_dim=node_model_in_dim,
            fc_dims=node_model_feats_dict['fc_dims'],
            dropout_p=node_model_feats_dict['dropout_p'],
            use_batchnorm=node_model_feats_dict['use_batchnorm'],
        )

        flow_out_mlp = MLP(
            input_dim=node_model_in_dim,
            fc_dims=node_model_feats_dict['fc_dims'],
            dropout_p=node_model_feats_dict['dropout_p'],
            use_batchnorm=node_model_feats_dict['use_batchnorm'],
        )

        node_mlp_in_dim = encoder_feats_dict['node_out_dim'] + node_model_feats_dict['fc_dims'][-1]
        node_mlp = nn.Sequential(
            nn.Linear(node_mlp_in_dim, encoder_feats_dict['node_out_dim']),
            nn.ReLU(inplace=True),
        )

        return MetaLayer(
            edge_model=EdgeModel(edge_mlp=edge_mlp),
            node_model=BipartiteNodeModel(
                flow_in_mlp=flow_in_mlp,
                flow_out_mlp=flow_out_mlp,
                node_mlp=node_mlp,
                node_agg_fn=node_agg_fn,
            )
        )

    def _reattach_if_needed(self, latent_node_feats, latent_edge_feats,
                            initial_node_feats, initial_edge_feats):
        if self.reattach_initial_edges:
            latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)
        if self.reattach_initial_nodes:
            latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)
        return latent_node_feats, latent_edge_feats

    def forward(self, x, edge_index, edge_attr, x_depth=None, frame_id=None,
                num_prev_nodes=None, batch=None, ptr=None):
        # ----- Base initial encodings -----
        latent_edge_feats, latent_node_feats = self.encoder(edge_attr, x)
        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats

        # ----- Depth subgraph initial encodings -----
        if self.use_depth_subgraph:
            if x_depth is None:
                raise ValueError("x_depth must be provided when depth subgraph is enabled.")

            depth_edge_scalar = self.depth_edge_builder(
                x_depth=x_depth,
                frame_id=frame_id,
                edge_index=edge_index
            ).unsqueeze(1)  # (E, 1)

            latent_depth_edge_feats, latent_depth_node_feats = self.depth_encoder(
                depth_edge_scalar, x_depth
            )
            initial_depth_edge_feats = latent_depth_edge_feats
            initial_depth_node_feats = latent_depth_node_feats
        else:
            latent_depth_edge_feats = None
            latent_depth_node_feats = None
            initial_depth_edge_feats = None
            initial_depth_node_feats = None

        first_class_step = self.num_enc_steps - self.num_class_steps + 1
        outputs_dict = {'classified_edges': []}

        for step in range(1, self.num_enc_steps + 1):
            # -------- depth subgraph first --------
            if self.use_depth_subgraph:
                depth_node_in, depth_edge_in = self._reattach_if_needed(
                    latent_depth_node_feats,
                    latent_depth_edge_feats,
                    initial_depth_node_feats,
                    initial_depth_edge_feats,
                )

                latent_depth_node_feats, latent_depth_edge_feats = self.depth_MPNet(
                    depth_node_in,
                    edge_index,
                    depth_edge_in,
                    num_prev_nodes=num_prev_nodes,
                    batch=batch,
                    ptr=ptr,
                )

                # alpha_ij from depth subgraph
                depth_alpha = torch.sigmoid(self.depth_gate(latent_depth_edge_feats))  # (E, 1)

            # -------- fuse depth prior into base graph --------
            if self.use_depth_subgraph:
                # Use depth prior to modulate BASE edges
                fused_latent_edge_feats = (1.0 + depth_alpha) * latent_edge_feats
            else:
                fused_latent_edge_feats = latent_edge_feats

            # -------- reattach initial embeddings if needed --------
            base_node_in, base_edge_in = self._reattach_if_needed(
                latent_node_feats,
                fused_latent_edge_feats,
                initial_node_feats,
                initial_edge_feats,
            )

            # -------- update base graph --------
            latent_node_feats, latent_edge_feats = self.MPNet(
                base_node_in,
                edge_index,
                base_edge_in,
                num_prev_nodes=num_prev_nodes,
                batch=batch,
                ptr=ptr,
            )

            # classify only from updated base graph
            if step >= first_class_step:
                dec_edge_feats, _ = self.classifier(latent_edge_feats)
                outputs_dict['classified_edges'].append(dec_edge_feats)

        if self.num_enc_steps == 0:
            dec_edge_feats, _ = self.classifier(latent_edge_feats)
            outputs_dict['classified_edges'].append(dec_edge_feats)

        return outputs_dict
