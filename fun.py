import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def update_matrix(g, self_reinforcement_pct=None,eta=0):
    adj_mat = nx.to_scipy_sparse_matrix(g)
    N = adj_mat.shape[0]
    if self_reinforcement_pct is not None:
        self_ref = self_reinforcement_pct*np.eye(N)
        neighbour_ref = adj_mat*(
            ((1-self_reinforcement_pct)/adj_mat.sum(axis=1))[:,np.newaxis])
        update_mat = self_ref * neighbour_ref
    else:
        numerator = np.eye(N) + adj_mat
        denomenator = numerator.sum(axis=1)*(1+eta)
        update_mat = numerator/denomenator
    return update_mat

def exp_op_div(g_mat, sigma_sq, s=1, sigma_squared_p=None):
    """
    The expected cross-sectional variance for an undirected graph
    :param g_mat: update matrix (A)
    :param sigma_sq: variance of the i.i.d error terms
    :param s: susceptibility
    :param sigma_squared_p: the variance of the distribution of prejudices
    :return: The expected cross-sectional variance for an undirected graph
    """
    if sigma_squared_p is None:
        sigma_squared_p = sigma_sq
    N = g_mat.shape[0]
    e_vals= np.sort(np.linalg.eigvals(g_mat))[::-1]
    return (1/N)*(sigma_sq+(sigma_squared_p*(1-s)**2))*sum(1/(1-((s*e_vals)**2)))

#def exp_op_div_symmetric(g_mat, sigma_sq):
    """
    The expected cross-sectional variance for an undirected graph
    :param g_mat: update matrix (A)
    :param sigma_sq: variance of the i.i.d error terms
    :return: The expected cross-sectional variance for an undirected graph
    """
#    N = g_mat.shape[0]
#    e_vals= np.sort(np.linalg.eigvals(g_mat))[::-1]
#    return (sigma_sq/N)*sum(1/(1-(e_vals**2)))

#def exp_op_div_symmetric_2(g_mat, sigma_sq):
    """
    The expected cross-sectional variance for an undirected graph
    :param g_mat: update matrix (A)
    :param sigma_sq: variance of the i.i.d error terms
    :return: The expected cross-sectional variance for an undirected graph
    """
#    cov_mat = covar_matrix_1(g_mat, sigma_sq)
#    N = cov_mat.shape[0]
#    return (1/N)*np.trace(cov_mat)

def realised_op_div(sample):
    return np.var(sample, axis=0, ddof=1)

def realised_mean_op_div(sample):
    return np.mean(realised_op_div(sample))

"""def exp_disagreement(g_mat, sigma_sq):
    N = g_mat.shape[0]
    e_vals = np.sort(np.linalg.eigvals(g_mat))[::-1]
    coef = (1 / N) * (sigma_sq)
    sum_part = sum(1/(1 + e_val) for e_val in e_vals)
    return  coef*sum_part"""

def exp_disagreement(g_mat, sigma_sq, s=1, sigma_squared_p=None):
    """
    The expected cross-sectional variance for an undirected graph
    :param g_mat: update matrix (A)
    :param sigma_sq: variance of the i.i.d error terms
    :param s: susceptibility
    :param sigma_squared_p: the variance of the distribution of prejudices
    :return: The expected cross-sectional variance for an undirected graph
    """
    if sigma_squared_p is None:
        sigma_squared_p = sigma_sq
    N = g_mat.shape[0]
    e_vals= np.sort(np.linalg.eigvals(g_mat))[::-1]
    sum_component = sum((1-e_val)/(1-(s*e_val)**2) for e_val in e_vals)
    return (1/N)*(sigma_sq+(sigma_squared_p*(1-s)**2))*sum_component

def realised_disagreement(sample, update_mat):
    N = update_mat.shape[0]

    return (1/N)*(sample.T @ (np.eye(N)-update_mat) @ sample)




def covar_matrix_1(g_mat, sigma_sq):
    """
    The full covariance matrix
    :param g_mat: update matrix (A)
    :param sigma_sq: variance of the i.i.d error terms
    :return: The full covariance matrix
    """

    e_vals, e_vects = np.linalg.eig(g_mat)

    # the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    N = g_mat.shape[0]
    # iterating over pairs of nodes
    covar_mat = np.zeros_like(g_mat)
    for n in range(N):
        for m in range(n, N):
            unscaled = 0
            for i in range(N):
                vect_prod_component = e_vects[m, i] * e_vects[n, i]
                eigen_component = 1 / (1 - e_vals[i] ** 2)
                both_components = vect_prod_component * eigen_component
                unscaled += both_components
            scaled = sigma_sq * unscaled
            covar_mat[m, n] = scaled
            covar_mat[n, m] = scaled
    return covar_mat

def covar_matrix_2(g_mat, sigma_sq):
    """
    The cross-sectional covariance matrix derived from the probibalistic approach
    :param g_mat: update matrix (A)
    :param sigma_sq: variance of the i.i.d error terms
    :return: The full cross-sectional covariance matrix
    """
    #TODO Figure out why the two methods give different results.
    N = g_mat.shape[0]
    geom_series = np.linalg.inv(np.eye(N)-(g_mat@g_mat.T))
    cross_sect_covar = sigma_sq*geom_series
    return cross_sect_covar


def results_df(ts_files, graphs):
    """

    :param ts_files:
    :param graphs:
    :return:
    """
    realised_op_divs = defaultdict(list)
    expected_op_divs = defaultdict(lambda: defaultdict(list))
    final_states = defaultdict(lambda: defaultdict(list))
    final_states_pooled_var = defaultdict(lambda: defaultdict(list))

    scale = lambda l: 1 / (1 - l ** 2)
    ss_eigen_vals = lambda ls: sum(scale(l) for l in ls) / len(ls)
    largest_eigenval = lambda update_mat: np.sort(np.linalg.eig(update_mat)[0])[-1]
    second_largest_eigenval = lambda update_mat: np.sort(np.einalg.eig(update_mat)[0])[-2]
    spectral_gap = lambda update_mat: largest_eigenval(update_mat) - second_largest_eigenval(update_mat)
    ave_clustering_coef = lambda g: nx.average_clustering(g)

    # spectral_gaps = defaultdict(list)

    d = []
    spectra = []
    seen_graphs = set()

    # ts_files = list(file_loc.glob('*/*.npz'))
    for file in tqdm(ts_files):
        npz_file = np.load(file, allow_pickle=True)
        graph_params = npz_file['graph_params'].item()
        graph_id = str(npz_file['graph_id'])
        eta = npz_file['eta'].item()

        ##TODO: in future all npz files should have noise_params and noise_type keys and should throw an error if they don't
        if ('noise_params' in npz_file) and ('noise_type' in npz_file):
            noise_params = npz_file['noise_params'].item()
            noise_type = npz_file['noise_type'].item()
        else:
            noise_params = {'mu': npz_file['error_params'][0],
                            'sigma_squared': npz_file['error_params'][1]}
            noise_type = ""
            graph_params = {'p': float(graph_params[-3:])}
        sigma_sq = noise_params['sigma_squared']
        graph_params_str = '_'.join(f'{key}_{graph_params[key]}' for key in sorted(graph_params))
        try:
            graph = graphs[graph_params_str][graph_id]
        except KeyError:
            graph = graphs[graph_id]

        graph_mat = noise_params.pop('update_mat',None)
        if graph_mat is None:
            graph_mat = update_matrix(graph, eta=eta)

        eod = exp_op_div(graph_mat, sigma_sq=sigma_sq).real
        e_dis = exp_disagreement(graph_mat, sigma_sq)
        #sg = spectral_gap(graph_mat)
        eigen_val_sequence = np.sort(np.linalg.eig(graph_mat)[0])[::-1]
        sg = eigen_val_sequence[0].real-eigen_val_sequence[1].real
        sum_scaled_eigen_vals = ss_eigen_vals(eigen_val_sequence)

        # largest_eig = largest_eigenval(graph_mat)
        # second_eig = second_largest_eigenval(graph_mat)
        acc = ave_clustering_coef(graph)

        if graph_id not in seen_graphs:
            for index, eigen_val in enumerate(eigen_val_sequence):
                if index == 0:
                    continue
                spectra.append({
                    'graph_id': graph_id,
                    'eigenvalue_index': index + 1,
                    'eigenval': eigen_val.real,
                    'scaled_eigenval': scale(eigen_val.real),
                    'dis_scaled_eigenval': 1/(1+eigen_val.real),
                    **graph_params
                })
        seen_graphs.add(graph_id)


        final_states_list = []  # [graph_params][graph_id].append(sample[:,-1])
        for sample_num, sample in enumerate(npz_file['ys']):
            if 'save_full' in npz_file:
                if npz_file['save_full'].item() == 'True':
                    final_states_list.append(sample[:, -1])
                elif npz_file['save_full'].item() == 'False':
                    final_states_list.append(sample)
            else:
                final_states_list.append(sample[:, -1])
        final_states = np.concatenate(final_states_list)

        d.append({
            'graph_id': graph_id,
            'div': eod,
            'dis': e_dis,
            'type': 'expected',
            'sum_scaled_eigen_vals': sum_scaled_eigen_vals,
            'spectral_gap': sg,
            'acc': acc,
            'noise_type': 'expected',
            **noise_params,
            **graph_params,
        })
        d.append({
            'graph_id': graph_id,
            'div': realised_op_div(final_states),
            'dis': np.median([realised_disagreement(final_state, graph_mat) for final_state in final_states_list]),
            'type': 'realised',
            'sum_scaled_eigen_vals': sum_scaled_eigen_vals,
            'spectral_gap': sg,
            'acc': acc,
            'noise_type': noise_type,
            **noise_params,
            **graph_params,
        })
    df = pd.DataFrame(d)
    df = df.drop_duplicates()

    spectra_df = pd.DataFrame(spectra)
    return df, spectra_df