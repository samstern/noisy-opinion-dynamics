import argparse
from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm

from .fun import update_matrix, exp_disagreement, exp_op_div

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', action='store',
                        help='directory where the results are to be stored'
                        )

    parser.add_argument('--in_dir', action='store',
                        help='directory where the input files are held'
                        )

    return parser.parse_args()

def main(in_dir,out_dir):
    in_path = Path(in_dir)
    in_files = in_path.glob('*/*/*.yaml')

    ETA = 0.1
    sigma_sq = 1

    scale = lambda l: 1 / (1 - l ** 2)
    ss_eigen_vals = lambda ls: sum(scale(l) for l in ls) / len(ls)

    d = []
    spectra = []
    for file in tqdm(in_files):
        graph_id = file.stem
        graph = nx.read_yaml(file)

        graph_params = graph.graph['graph_params']

        graph_mat = update_matrix(graph, eta=ETA)

        eod = exp_op_div(graph_mat, sigma_sq=sigma_sq).real
        e_dis = exp_disagreement(graph_mat, sigma_sq)

        eigen_val_sequence = np.sort(np.linalg.eig(graph_mat)[0])[::-1]
        sg = eigen_val_sequence[0].real - eigen_val_sequence[1].real
        sum_scaled_eigen_vals = ss_eigen_vals(eigen_val_sequence)

        intra_cluster = graph_params['p'][0][0]
        inter_cluster = graph_params['p'][0][1]
        for index, eigen_val in enumerate(eigen_val_sequence):
            if index == 0:
                continue
            spectra.append({
                'graph_id': graph_id,
                'eigenvalue_index': index + 1,
                'eigenval': eigen_val.real,
                'scaled_eigenval': scale(eigen_val.real),
                'dis_scaled_eigenval': 1 / (1 + eigen_val.real),
                'inter_cluster': inter_cluster,
                'intra_cluster': intra_cluster,
            })

        d.append({
            'graph_id': graph_id,
            'div': eod,
            'dis': e_dis,
            'sum_scaled_eigen_vals': sum_scaled_eigen_vals,
            'acc': nx.average_clustering(graph),
            'inter_cluster': inter_cluster,
            'intra_cluster': intra_cluster,
        })
    df = pd.DataFrame(d)
    df = df.drop_duplicates()

    param_level_df = df.groupby(by=['inter_cluster', 'intra_cluster']).mean().reset_index()

    spectra_df = pd.DataFrame(spectra)

    out_path = Path(out_dir)
    pd.to_pickle(df, out_path / 'df.pkl')
    pd.to_pickle(param_level_df, out_path / 'param_level_df.pkl')
    pd.to_pickle(spectra_df, out_path / 'spectra_df.pkl')


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))