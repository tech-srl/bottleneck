import numpy as np
import itertools
import random
import math

from tasks.tree_dataset import TreeDataset
import common


class DictionaryLookupDataset(TreeDataset):
    def __init__(self, depth):
        super(DictionaryLookupDataset, self).__init__(depth)

    def get_combinations(self):
        # returns: an iterable of [key, permutation(leaves)]
        # number of combinations: (num_leaves!)*num_choices
        num_leaves = len(self.leaf_indices)
        num_permutations = 1000
        max_examples = 32000

        if self.depth > 3:
            per_depth_num_permutations = min(num_permutations, math.factorial(num_leaves), max_examples // num_leaves)
            permutations = [np.random.permutation(range(num_leaves)) for _ in
                            range(per_depth_num_permutations)]
        else:
            permutations = random.sample(list(itertools.permutations(range(num_leaves))),
                                         min(num_permutations, math.factorial(num_leaves)))

        return itertools.chain.from_iterable(

            zip(range(num_leaves), itertools.repeat(perm))
            for perm in permutations)

    def get_nodes_features(self, combination):
        # combination: a list of indices
        # Each leaf contains a one-hot encoding of a key, and a one-hot encoding of the value
        # Every other node is empty, for now
        selected_key, values = combination

        # The root is [one-hot selected key] + [0 ... 0]
        nodes = [
            common.one_hot(selected_key, len(self.leaf_indices)) + [0] * len(self.leaf_indices)
        ]
        for i in range(1, self.num_nodes):
            if i in self.leaf_indices:
                leaf_num = self.leaf_indices.index(i)
                node = common.one_hot(leaf_num, len(self.leaf_indices)) + common.one_hot(values[leaf_num],
                                                                                         len(self.leaf_indices))
            else:
                node = [0] * (2 * len(self.leaf_indices))
            nodes.append(node)
        return nodes

    def label(self, combination):
        selected_key, values = combination
        return int(values[selected_key])

    def get_dims(self):
        # get input and output dims
        in_dim = len(self.leaf_indices) * 2
        out_dim = len(self.leaf_indices)
        return in_dim, out_dim
