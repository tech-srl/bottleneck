import pickle
from argparse import ArgumentParser

raw_keys = ['Child', 'NextToken', 'ComputedFrom', 'LastUse', 'LastWrite', 'LastLexicalUse', 'FormalArgName', 'GuardedBy', 'GuardedByNegation', 'UsesSubtoken']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--edges", dest="edges", required=True)
    args = parser.parse_args()
    
    with open(args.edges, 'rb') as file:
        raw_edges = pickle.load(file)

    parent_to_children = {}
    child_to_parent = {}
    for s, t in raw_edges['Child']:
        if not s in parent_to_children:
            parent_to_children[s] = []
        parent_to_children[s].append(t)
        child_to_parent[t] = s
        
    cur = 0
    next_map = {}
    for s, t in raw_edges['NextToken']:
        next_map[s] = t
    prev_map = {t:s for s,t in next_map.items()}

    def get_all_next(n):
        result = []
        cur = n
        while cur in next_map:
            next_item = next_map[cur]
            result.append(next_item)
            cur = next_item
        return result

    def get_all_prev(n):
        result = []
        cur = n
        while cur in prev_map:
            prev_item = prev_map[cur]
            result.append(prev_item)
            cur = prev_item
        return result

    
    nodes = child_to_parent.keys()
    left_nodes = list(nodes)

    parent_to_descendants = {}
    def get_parent_to_descendants(p):
        desc = set()
        for c in parent_to_children[p]:
            if c in parent_to_children: # if c is a parent itself
                desc.update(get_parent_to_descendants(c))
            else:
                desc.add(c)
        return desc
                
    for p in parent_to_children.keys():
        desc = get_parent_to_descendants(p)
        parent_to_descendants[p] = desc

    roots = set()
    for n in nodes:
        cur = n
        while cur in child_to_parent:
            cur = child_to_parent[cur]
        roots.add(cur)
    
    print(raw_edges)
    