from typing import Dict, Any, List

import tensorflow as tf

from utils import MLP
from .sparse_graph_model import Sparse_Graph_Model
from tasks import Sparse_Graph_Task
from gnns import sparse_gnn_edge_mlp_layer


class No_Struct_MLP_Model(Sparse_Graph_Model):
    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'max_nodes_in_batch': 25000,
            'hidden_size': 128,
            "graph_activation_function": "gelu",
            "message_aggregation_function": "sum",
            'graph_inter_layer_norm': True,
            'use_target_state_as_input': True,
            'num_edge_hidden_layers': 0,
        })
        return params

    @staticmethod
    def name(params: Dict[str, Any]) -> str:
        return "NoStruct-MLP%i" % (params['num_edge_hidden_layers'])

    def __init__(self, params: Dict[str, Any], task: Sparse_Graph_Task, run_id: str, result_dir: str) -> None:
        super().__init__(params, task, run_id, result_dir)

    def _apply_gnn_layer(self,
                         node_representations: tf.Tensor,
                         adjacency_lists: List[tf.Tensor],
                         type_to_num_incoming_edges: tf.Tensor,
                         num_timesteps: int
                         ) -> tf.Tensor:
        graph_to_nodes = self._Sparse_Graph_Model__placeholders['graph_to_nodes']
        graph_nodes_list = self._Sparse_Graph_Model__placeholders['graph_nodes_list'] # (None, )
        max_nodes = tf.shape(graph_to_nodes)[1]
        tiled_nodes = tf.tile(tf.expand_dims(graph_to_nodes, axis=-1), (1, 1, max_nodes))
        pairs = tf.concat(
            [tf.expand_dims(tiled_nodes, axis=-1), tf.expand_dims(tf.transpose(tiled_nodes, [0, 2, 1]), axis=-1)],
            axis=-1)
        flat_pairs = tf.reshape(pairs, [-1, 2])
        relevant_edges = tf.reshape(tf.gather(flat_pairs, tf.where(tf.reduce_min(flat_pairs, axis=-1) >= 0)), [-1, 2])
        
        num_types = tf.shape(type_to_num_incoming_edges)[0]
        num_nodes_in_graph = tf.reduce_sum(tf.cast(tf.greater(graph_to_nodes, -1), dtype=tf.float32), axis=-1)
        num_incoming_nodes_per_node = tf.gather(params=num_nodes_in_graph, indices=graph_nodes_list)
        type_to_num_incoming_edges = tf.tile(tf.expand_dims(num_incoming_nodes_per_node, axis=0), [num_types, 1])
        
        return sparse_gnn_edge_mlp_layer(
            node_embeddings=node_representations,
            adjacency_lists=[relevant_edges for _ in adjacency_lists],
            type_to_num_incoming_edges=type_to_num_incoming_edges,
            state_dim=self.params['hidden_size'],
            num_timesteps=num_timesteps,
            activation_function=self.params['graph_activation_function'],
            message_aggregation_function=self.params['message_aggregation_function'],
            use_target_state_as_input=self.params['use_target_state_as_input'],
            num_edge_hidden_layers=self.params['num_edge_hidden_layers'],
        )
