import tensorflow as tf
import numpy as np


class SelfAttention:
    def __init__(self,
                 num_heads,
                 model_dim,
                 dropout_keep_prob):

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dropout_keep_prob = dropout_keep_prob

    def multi_head(self, batched_inputs, valid_mask=None):
        q = self._linear_projection(batched_inputs)
        qs = self._split_heads(q)
        tiled_inputs = tf.tile(tf.expand_dims(batched_inputs, axis=1), [1, self.num_heads, 1, 1])
        outputs = self._scaled_dot_product(qs, tiled_inputs, tiled_inputs, valid_mask)  # (batch, num_heads, max_contexts, value_dim)
        output = self._concat_heads(outputs)  # (batch, max_contexts, value_dim * num_heads)
        output = tf.layers.dense(output, units=self.model_dim, use_bias=False,
                                 activation=tf.nn.relu)  # (batch, max_contexts, model_dim)

        output = tf.nn.dropout(output, keep_prob=self.dropout_keep_prob)
        return output

    def _linear_projection(self, batched_inputs):
        q = tf.layers.dense(batched_inputs, units=self.model_dim * self.num_heads,
                            use_bias=False)  # (batch, max_contexts, key_dim * num_heads)
        # k = tf.layers.dense(batched_inputs, units=self.model_dim,
        #                     use_bias=False)  # (batch, max_contexts, key_dim * num_heads)
        return q

    def _split_heads(self, q):

        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            tensor = tf.reshape(tensor, [-1, tf.shape(tensor)[1], num_heads,
                                         dim])  # (batch, max_contexts, num_heads, dim)
            return tf.transpose(tensor, [0, 2, 1, 3])  # (batch, num_heads, max_contexts, dim)

        qs = split_last_dimension_then_transpose(q, self.num_heads,
                                                 self.model_dim)  # (batch, num_heads, max_contexts, key_dim)
        # ks = split_last_dimension_then_transpose(k, self.num_heads,
        #                                          self.model_dim)  # (batch, num_heads, max_contexts, key_dim)
        return qs

    def _scaled_dot_product(self, qs, ks, tiled_inputs, valid_mask):
        queries_dot_keys = tf.matmul(qs, ks, transpose_b=True)  # (batch, num_heads, max_contexts, max_contexts)
        scaled_scores = queries_dot_keys #/ ((self.model_dim // self.num_heads) ** 0.5)  # (batch, num_heads, max_contexts, max_contexts)

        if valid_mask is not None:
            mask = tf.log(tf.reshape(valid_mask, (
            tf.shape(valid_mask)[0], 1, 1, tf.shape(valid_mask)[1])))  # (batch, 1, 1, max_contexts)
            scaled_scores += mask

        attention_weights = tf.nn.softmax(scaled_scores, axis=-1)  # (batch, num_heads, max_contexts, max_contexts)
        return tf.matmul(attention_weights, tiled_inputs)  # (batch, num_heads, max_contexts, value_dim)

    def _concat_heads(self, outputs):
        # outputs: (batch, num_heads, max_contexts, value_dim)
        max_contexts = tf.shape(outputs)[2]
        tensor = tf.transpose(outputs, [0, 2, 1, 3])  # [batch, max_contexts, num_heads, value_dim // num_heads]
        return tf.reshape(tensor, [-1, max_contexts, self.model_dim * self.num_heads])


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    selfatt = SelfAttention(num_heads=2, model_dim=4, dropout_keep_prob=1.0)
    result_op = selfatt.multi_head(tf.constant(np.arange(24).reshape((2, 3, 4)), dtype=tf.float32))
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))
    result = sess.run(result_op)
    print(result.shape)
    print(result)