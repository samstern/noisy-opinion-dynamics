import argparse
from pathlib import Path
import networkx as nx
import numpy as np
from uuid import uuid4
import yaml

from .fun import update_matrix

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', action='store',
                        help='some parameter for something'
                        )
    parser.add_argument('--direction'
                       )

    parser.add_argument('--num_graphs', action='store', type=int,
                        required=True)


    parser.add_argument('--eta', type=float,
                        help='small value that is subtracted from the adjacency matrix to ensure that it '
                             'is substochastic')

    parser.add_argument('--graph_params', action='store',
                        help = "dictionary with keys 'n' (number of nodes),"
                               " 'k' (number of nearest lattice neighbours a node is connected to) and"
                               " 'p' (rewiring probability)")

    parser.add_argument('--require_connected', action='store', default=True,
                        help='All graphs must be (strongly) connected')

    parser.add_argument('--require_invertible', action='store', default=True,
                        help='Require the update matrix to be invertible')

    parser.add_argument('--require_stationary', action='store', default=True,
                        help='Require the process to be covariance stationary (max eigenvalue < 1)')


    return parser.parse_args()


def make_graph(graph_params, eta, require_connected=True, require_invertible=True, require_stationary=True):
    while True:
        strongly_connected = False
        invertible = False
        stationary = False

        graph = nx.connected_watts_strogatz_graph(**graph_params)
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

def main(out_dir, direction, num_graphs, graph_params, eta,
    require_connected = True, require_invertible = True, require_stationary = True):
    graph_params = yaml.safe_load(graph_params)
    graphs = dict()
    for i in range(num_graphs):
        i_d =str(uuid4())
        graph = make_graph(graph_params, eta, require_connected=require_connected,
                           require_invertible=require_invertible, require_stationary=require_stationary)
        graphs[i_d]='\n'.join(nx.generate_graphml(graph))

    out_file = Path(out_dir)/'undirected_sw_graphs.yaml'
    out_file.write_text(yaml.dump(graphs))

    out_path = Path(out_dir) / 'watts_strogats'
    out_path.mkdir(parents=True, exist_ok=True)
    graph_params_str = '_'.join([f'{param}_{graph_params[param]}' for param in sorted(graph_params.keys())])
    out_file = out_path / f'{direction}_{graph_params_str}.yaml'
    out_file.write_text(yaml.dump(graphs))

    #gs_txt = out_file.read_text()
    #gs = yaml.load(gs_txt, Loader=yaml.SafeLoader)
    #gs = {k:nx.parse_graphml(v,node_type=int) for k,v in gs.items()}


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))