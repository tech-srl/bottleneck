from argparse import ArgumentParser
import json
import networkx as nx
import gzip
import numpy as np
import statistics

def compute_diameter(adjacency_list):
    # graph is a list of edges
    # every edge is a list: [source, type, target]
    g = nx.Graph()
    for edge_source, _, edge_target in adjacency_list:
        g.add_edge(edge_source, edge_target)
    return nx.diameter(g)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", dest="data", required=True)
    args = parser.parse_args()
    
    with gzip.open(args.data, 'r') as file:
        lines = file.readlines()
    
    objs = [json.loads(line) for line in lines]
    graphs = [o['graph'] for o in objs]

    diameters = [compute_diameter(graph) for graph in graphs]
    print('Max diameter: ', max(diameters))
    print('Mean diameter: ', np.mean(diameters))
    print('stddev: ', statistics.stdev(diameters))

    percentiles = range(10, 110, 10)
    percentile_results = np.percentile(diameters, percentiles)
    for i, res in zip(percentiles, percentile_results):
        print('Diameters - {} percentile: {}'.format(i, res))