import argparse
from pathlib import Path
import networkx as nx
from uuid import uuid4
import yaml
import json
import numpy as np
from tqdm import tqdm

from .fun import update_matrix


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', action='store',
                        help='some parameter for something'
                        )
    parser.add_argument('--direction', choices={"directed", "undirected"},
                        type=str,
                        help='whether the graph is directed or undirected'
                        )

    parser.add_argument('--num_graphs', action='store', type=int,
                        required=True)

    parser.add_argument('--num_nodes', action='store', type=int,
                        help='number of nodes in each network'
                        )

    parser.add_argument('--graph_type', action='store', type=str,
                        help='the type of networkx graph'
                        )

    parser.add_argument('--graph_params', action='store',
                        help='dictionary of graph parameters')

    parser.add_argument('--eta', type=float,
                        help='small value that is subtracted from the adjacency matrix to ensure that it '
                             'is substochastic')

    parser.add_argument('--require_connected', action='store', default=False,
                        help='All graphs must be (strongly) connected')

    parser.add_argument('--require_invertible', action='store', default=True,
                        help='Require the update matrix to be invertible')

    parser.add_argument('--require_stationary', action='store', default=True,
                        help='Require the process to be covariance stationary (max eigenvalue < 1)')

    return parser.parse_args()


def make_graph(nx_graph_type, nx_graph_params, eta, require_connected=False, require_invertible=True, require_stationary=True):
    graph_cases = dict(
        erdos_renyi = nx.erdos_renyi_graph,
        watts_strogatz = nx.watts_strogatz_graph,
        stochastic_block_model = nx.generators.stochastic_block_model,
    )
    while True:
        connected = False
        invertible = False
        stationary = False
        graph = graph_cases[nx_graph_type](**nx_graph_params)
        g_mat = update_matrix(graph, eta=eta)
        strongly_connected = nx.is_connected(graph)

        invertible = not (np.linalg.det(g_mat) == 0)

        e_vals, e_vects = np.linalg.eig(g_mat)
        stationary = (max(e_vals) < 1.0)
        if (((not require_connected) or strongly_connected)
                and ((not require_invertible) or invertible)
                and ((not require_stationary) or stationary)):
            break
    graph.graph['name'] = nx_graph_type
    graph.graph['graph_params'] = nx_graph_params
    graph.graph['id'] = str(uuid4())

    return graph


def main(out_dir, direction, num_graphs, num_nodes, graph_type, graph_params, eta,
         require_connected=False, require_invertible=False, require_stationary=False):


    graph_params = yaml.safe_load(graph_params)

    out_path = Path(out_dir) / graph_type
    out_path = out_path / '/'.join([f'{param}_{graph_params[param]}' for param in sorted(graph_params.keys())])
    out_path.mkdir(parents=True, exist_ok=True)
    graph_params_str = out_path / '/'.join([f'{param}_{graph_params[param]}' for param in sorted(graph_params.keys())])

    i = 0

    while i < num_graphs:
        i_d = str(uuid4())


        graph = make_graph(graph_type, graph_params, eta, require_connected=require_connected,
                                  require_invertible=require_invertible, require_stationary=require_stationary)


        out_loc = out_path / (graph.graph['id'] + '.yaml')
        nx.write_yaml(graph, out_loc)
        i+=1



if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
