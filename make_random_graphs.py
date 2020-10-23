import argparse
from pathlib import Path
import networkx as nx
from uuid import uuid4
import yaml
import json
import numpy as np

from .fun import update_matrix

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', action='store',
                        help='some parameter for something'
                        )
    parser.add_argument('--direction',choices={"directed","undirected"},
                        type=str,
                        help='whether the graph is directed or undirected'
                       )

    parser.add_argument('--num_graphs', action='store', type=int,
                        required=True)

    parser.add_argument('--num_nodes',action='store', type=int,
                        help = 'number of nodes in each network'
                        )

    parser.add_argument('--graph_params', action='store',
                        help = 'dictionary of graph parameters')

    parser.add_argument('--eta', type=float,
                        help = 'small value that is subtracted from the adjacency matrix to ensure that it '
                               'is substochastic')

    parser.add_argument('--require_connected', action='store', default=True,
                        help='All graphs must be (strongly) connected')

    parser.add_argument('--require_invertible', action='store', default=True,
                        help='Require the update matrix to be invertible')

    parser.add_argument('--require_stationary', action='store', default=True,
                        help='Require the process to be covariance stationary (max eigenvalue < 1)')

    return parser.parse_args()


def make_random_graph(graph_params, eta, require_connected=True, require_invertible=True, require_stationary=True):
    while True:
        strongly_connected = False
        invertible = False
        stationary = False

        graph = nx.erdos_renyi_graph(**graph_params)
        g_mat = update_matrix(graph, eta)

        strongly_connected = nx.is_connected(graph)

        invertible = not (np.linalg.det(g_mat) == 0)

        e_vals, e_vects = np.linalg.eig(g_mat)
        stationary = (max(e_vals) < 1.0)

        if (((not require_connected) or strongly_connected)
                and ((not require_invertible) or invertible)
                and ((not require_stationary) or stationary)):
            break

    return graph



def main(out_dir, direction, num_graphs, num_nodes, graph_params, eta,
         require_connected=True, require_invertible=True, require_stationary=True):
    graph_params = yaml.safe_load(graph_params)
    graphs = dict()
    i=0
    while i<num_graphs:
        i_d =str(uuid4())

        graph_params['n'] = num_nodes

        graph = make_random_graph(graph_params, eta, require_connected=require_connected,
                                  require_invertible=require_invertible, require_stationary=require_stationary)
        graphs[i_d]='\n'.join(nx.generate_graphml(graph))
        i+=1

    out_path = Path(out_dir)/'erdos_renyi'
    out_path.mkdir(parents=True, exist_ok=True)

    out_file = out_path/f'{direction}_p_{graph_params["p"]}.yaml'
    out_file.write_text(yaml.dump(graphs))

    #gs_txt = out_file.read_text()
    #gs = yaml.load(gs_txt, Loader=yaml.SafeLoader)
    #gs = {k:nx.parse_graphml(v,node_type=int) for k,v in gs.items()}


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))