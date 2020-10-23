import argparse
from pathlib import Path
import networkx as nx
import yaml
import json
from glob import glob
import numpy as np
from tqdm import tqdm

from .fun import update_matrix
from .models import ModelFactory
from .noise import NoiseFactory

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', action='store',
                        help='directory where the results are to be stored'
                        )

    parser.add_argument('--in_dir', action='store',
                        help='directory where the input files are held'
                        )

    parser.add_argument('--graph_type', action='store',type=str,
                        help='the type of graph (random, sw)')

    parser.add_argument('--model_type', action='store', type=str,default='degroot',
                        help='the name of the model to be used')

    parser.add_argument('--noise_type', action='store', type=str, default='iidgaussian'                                          '',
                        help='the name of the noise class to be used')

    parser.add_argument('--noise_params', type=str,
                        help='a yaml-loadable string that has the parameters required to initialise a noise object' \
                             'e.g., the mean and variance')

    parser.add_argument('--model_params', action='store', type=str,
                        default=None,
                        help='a yaml-loadable string that has the parameters required to initialise a noise object')

    parser.add_argument('--direction',action='store',type=str,
                        help='whether the graph is directed or undirected')

    parser.add_argument('--graph_params', action='store', type=str,
                        help='the parameters that identify the graph over which to run simulations')

    parser.add_argument('--num_timesteps', type=int,
                        help='the number of discrete timesteps of the simulation')

    parser.add_argument('--num_samples', type=int,
                        help='the number of simulations that are run for each graph')

    parser.add_argument('--eta', type=float, default=0.0,
                        help = 'small value that is subtracted from the adjacency matrix to ensure that it '\
                               'is substochastic')

    parser.add_argument('--self_reinforcement_pct',
                        type=float,
                        default=None,
                        help='parameter controlling how much nodes are autocorrelated, should be between 0 and 1')

    parser.add_argument('--save_full',
                        type=str,
                        help='whether to save the entire simulation or only the final state of each node')


    return parser.parse_args()




def error_vector(mu, sigma_sq, num_dims):
    mu_vect = mu*np.ones(num_dims)
    covar_matrix = sigma_sq*np.identity(num_dims)
    return np.random.multivariate_normal(mu_vect,covar_matrix)

def simulate(update_mat, num_steps,
                        mu, sigma_sq):
    N = update_mat.shape[0]
    epsilons = np.zeros([N, num_steps])
    epsilons[:, 0] = error_vector(mu, sigma_sq, N)
    ys = np.zeros([N,num_steps])
    ys[:, 0] = epsilons[:, 0]
    for i in range(1,num_steps):
        epsilons[:, i] = error_vector(mu, sigma_sq, N)
        ys[:,i]= (update_mat @ ys[:,i-1]) + epsilons[:,i]
    return ys, epsilons

def main(in_dir,graph_type, direction, graph_params, out_dir, model_type, model_params, noise_type, noise_params,
         num_timesteps, num_samples, eta, self_reinforcement_pct, save_full):

    m_params = model_params
    graph_params = yaml.safe_load(graph_params)
    graph_params_str = '_'.join([f'{param}_{graph_params[param]}' for param in sorted(graph_params.keys())])

    noise_params = yaml.safe_load(noise_params)
    noise_params_str = noise_type+'_'+'_'.join([f'{param}_{noise_params[param]}' for param in sorted(noise_params.keys())])

    out_file_loc = Path(out_dir)/'simulation_results'/\
                       direction/\
                       graph_type/\
                        noise_type/\
                       f'{self_reinforcement_pct}_{noise_params_str}_{graph_params_str}'

    out_file_loc.mkdir(parents=True, exist_ok=True)

    in_path = Path(in_dir) / graph_type
    in_path = in_path / '/'.join([f'{param}_{graph_params[param]}' for param in sorted(graph_params.keys())])
    in_files = in_path.glob('*.yaml')
    graphs = {file.stem: nx.read_yaml(file) for file in in_files}
    #in_file = f'{direction}_{graph_params_str}.yaml'
    #graphs_txt = (Path(in_dir)/graph_type/in_file).read_text()
    #graphs = yaml.load(graphs_txt, Loader=yaml.FullLoader)
    #graphs = {k:nx.parse_graphml(v,node_type=int) for k,v in graphs.items()}

    print(in_path)
    for i_d, graph in tqdm(graphs.items()):

        update_mat = update_matrix(graph, self_reinforcement_pct, eta=eta)

        noise_params['N'] = update_mat.shape[0]
        noise_params['update_mat'] = update_mat

        noise_factory = NoiseFactory()
        noise = noise_factory.create_noise(noise_type, noise_params)

        #model_params = dict(
        #    update_matrix = update_mat,
        #    noise = noise,)

        if m_params is not None:
            model_params = yaml.safe_load(m_params)
            model_params_str = model_type + '_' + '_'.join(
                [f'{param}_{model_params[param]}' for param in sorted(model_params.keys())])
        else:
            model_params = dict()
        model_params['update_matrix'] = update_mat
        model_params['noise'] = noise

        if 'partition' in graph.graph.keys() and 'prejudice' in model_params.keys():
            model_params['prejudice']['params']['distribution_assignments'] = graph.graph['partition']

        model_factory = ModelFactory()
        model = model_factory.create_model(model_type, model_params)

        ys = []
        for j in range(num_samples):
            #sim_ys, sim_epsilons = simulate(update_mat, num_timesteps,
            #                                mu, sigma_sq)
            sim_ys, sim_epsilons = model.simulate(num_timesteps)

            if save_full=="False":
                sim_ys = sim_ys[:,-1]
            ys.append(sim_ys)

        out_dict = dict(
            direction=direction,
            graph_type=graph_type,
            graph_params=graph_params,
            self_reinforcement_pct=self_reinforcement_pct,
            model_type = model_type,
            model_params = model_params,
            noise_type = noise_type,
            noise_params = noise_params,
            graph_size = graph.order(),
            #graph_file_loc = in_file,
            graph_id = i_d,
            eta=eta,
            ys=ys,
            save_full=save_full

        )

        out_file = out_file_loc/f'{i_d}'

        #out_file.write_text(yaml.dump(out_dict))
        np.savez(out_file,**out_dict)


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))