import numpy as np

class Noise(object):
    def __init__(self, mu, sigma_squared, N):
        self.mu_vect = mu * np.ones(N)
        self.sigma_squared = sigma_squared * np.eye(N)

    def generate(self):
        pass

    def generate_first(self):
        """noise vector for time t=0"""
        return np.random.multivariate_normal(self.mu_vect, self.sigma_squared)

class IIDGaussianNoise(Noise):
    """
    i.i.d gaussian noise
    """
    def __init__(self, mu, sigma_squared, N, **kwargs):
        self.mu_vect = mu * np.ones(N)
        self.sigma_squared = sigma_squared * np.eye(N)

    def generate(self, **kwargs):
        return np.random.multivariate_normal(self.mu_vect, self.sigma_squared)


class GlobalUniquenessGaussianNoise(Noise):
    """
    gaussian noise where sigma_squared_i increases the closer i's opinion is to the consensus
    """
    def __init__(self, mu, sigma_squared, N, beta, **kwargs):
        self.mu_vect = mu * np.ones(N)
        self.sigma_squared = sigma_squared * np.eye(N)
        self.beta = beta

    def generate(self, ys_t, **kwargs):
        current_consensus = np.mean(ys_t)
        diff = np.power(ys_t - current_consensus, 2)
        variance = self.sigma_squared * np.diag(np.exp(-self.beta * diff))
        return np.random.multivariate_normal(self.mu_vect, variance)


class LocalUniquenessGaussianNoise(Noise):
    """
    gaussian noise where sigma_squared_i increases the closer i's opinion is to its neighbours
    """
    def __init__(self, mu, sigma_squared, update_mat, beta, **kwargs):
        N = update_mat.shape[0]
        self.update_mat = update_mat
        self.mu_vect = mu * np.ones(N)
        self.sigma_squared = sigma_squared * np.eye(N)
        self.beta = beta

    def generate(self, ys_t, **kwargs):
        diff = np.exp(-self.beta * np.square(ys_t.reshape(-1, 1) - ys_t))
        variance = self.sigma_squared * np.diag(np.sum(np.multiply(self.update_mat, diff), axis=1).A1)
        return np.random.multivariate_normal(self.mu_vect, variance)

class GaussianMixtureNoise(Noise):
    """
    mixture of gaussians
    """
    def __init__(self, mus, sigma_squareds, distribution_assignments=None):

        assert(len(mus)==len(distribution_assignments))
        assert(len(sigma_squareds)==len(distribution_assignments))

        self.num_communities = len(mus)
        self.mu_vects = [mus[i] * np.ones(len(distribution_assignments[i])) for i in range(self.num_communities)]
        self.sigma_squareds = [sigma_squareds[i] * np.eye(len(distribution_assignments[i])) for i in range(self.num_communities)]

    def generate(self):
        return np.concatenate(
            [np.random.multivariate_normal(self.mu_vects[i], self.sigma_squareds[i])\
             for i in range(self.num_communities)])


class NoiseFactory:
    def create_noise(self, noise_type, noise_params):
        if noise_type.lower() == 'iidgaussian':
            return IIDGaussianNoise(**noise_params)
        elif noise_type.lower()  in {'local_unique', 'local','local unique'}:
            return LocalUniquenessGaussianNoise(**noise_params)
        elif noise_type.lower() in {'global_unique', 'global','golobal unique'}:
            return GlobalUniquenessGaussianNoise(**noise_params)
        elif noise_type.lower() in {'mixture'}:
            return GaussianMixtureNoise(**noise_params)
        else:
            raise ValueError(noise_type)