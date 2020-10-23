import numpy as np

from .noise import NoiseFactory, IIDGaussianNoise

class Model(object):
    """

    """

    def simulate(self):
        pass



class FriedkinJohnsonModel(Model):
    """

    """
    def __init__(self, update_matrix, s, prejudice, noise=None, sigma_squared_p=None):
        """

        :param update_matrix: the matrix that governs interations between agents (i.e., the sub-stochastic adjacency matrix)
        :param s: a scalar susceptability of the agents
        :param prejudice: a vector of prejudices or a dictionary that describes how the prejudices should be asigned
        :param noise: an op_div_simulation.noise.Noise object that determines the noise of the system. If none then assumes that noise is 0
        :param prejudice_var: (optional) the variance of the prejudice. If None then estimate using the prejudice input argument.
        """
        self.update_matrix = update_matrix
        self.N = update_matrix.shape[0]
        self.s = s

        if type(prejudice) == dict:
            factory = NoiseFactory()
            prejudice_generator = factory.create_noise(prejudice['type'],prejudice['params'])
            self.prejudice = prejudice_generator.generate()
        else:
            self.prejudice = prejudice

        if noise is None:
            self.noise = IIDGaussianNoise(0,0,self.N)
        else:
            self.noise = noise

        if sigma_squared_p is None:
            self.sigma_squared_p = np.var(self.prejudice)
        else:
            self.sigma_squared_p = sigma_squared_p

    def simulate(self, num_steps):
        ys = np.zeros([self.N, num_steps])
        epsilons = np.zeros([self.N, num_steps])
        epsilons[:, 0] = self.noise.generate_first()
        ys[:, 0] = epsilons[:, 0]


        for i in range(1, num_steps):
            noise_payload = dict(
                ys_t = ys[:,i-1],
                update_matrix = self.update_matrix
            )
            epsilons[:, i] = self.noise.generate(**noise_payload)
            ys[:, i] = ((self.s * self.update_matrix) @ ys[:, i - 1].T) + (1 - self.s) * self.prejudice + epsilons[:, i]
        return ys, epsilons

    def exp_op_div(self):
        e_vals = np.sort(np.linalg.eigvals(self.update_matrix))[::-1]
        return (1 / self.N) * (self.noise.sigma_sq + (self.sigma_squared_p * (1 - self.s) ** 2)) * sum(1 / (1 - ((self.s * e_vals) ** 2)))

class DeGrootModel(FriedkinJohnsonModel):
    """
    The deGroot model is just a special case of the FJ model with susceptibility 1.
    """
    def __init__(self, update_matrix, noise, **kwargs):
        super(DeGrootModel, self).__init__(
            update_matrix=update_matrix,
            s=1,
            prejudice=0,
            noise=noise,
            sigma_squared_p=0)



class ModelFactory:
    def create_model(self, model_type, model_params):
        if model_type.lower() == 'degroot':
            return DeGrootModel(**model_params)
        elif model_type.lower()  in {'friedkin-johnson', 'fj'}:
            return FriedkinJohnsonModel(**model_params)
        else:
            raise ValueError(model_type)