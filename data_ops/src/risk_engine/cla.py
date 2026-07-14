import numpy as np
import pandas as pd

class customBaseOptimizer:
    """
    Instance variables:

    - ``n_assets`` - int
    - ``tickers`` - str list
    - ``weights`` - np.ndarray

    Public methods:

    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, n_assets, tickers = None):
        self.n_assets = n_assets
        if tickers is None:
            self.tickers = list(range(n_assets))
        else:
            self.ticker  = tickers
        self._risk_free_rate = None
        self.weights = None
    
    def _make_output_weights(self, weights = None):
        if weights is None:
            weights = self.weights
        
        return collections.OrderedDict(zip(self.tickers))
    
    def set_weights(self, input_weights):
        pass


class customCLA(BaseOptimizer):
    def __init__(self, expected_returns, cov_matrix, weight_bounds=(0, 1)):
        """
        :param expected_returns: expected returns for each asset. Set to None if
                                 optimising for volatility only.
        :type expected_returns: pd.Series, list, np.ndarray
        """
        self.mean = np.array(expected_returns).reshape((len(expected_returns), 1))
        self.expected_returns = self.mean.reshape((len(self.mean),))
        self.cov_matrix = np.asarray(cov_matrix)

        if len(weight_bounds) == len(self.mean) and not isinstance(
            weight_bounds[0], (float, int)
        ):
            self.lB = np.array([b[0] for b in weight_bounds]).reshape(-1, 1)
            self.uB = np.array([b[1] for b in weight_bounds]).reshape(-1, 1)
        else:
            if isinstance(weight_bounds[0], (float, int)):
                self.lB = np.ones(self.mean.shape) * weight_bounds[0]
            else:
                self.lB = np.array(weight_bounds[0]).reshape(self.mean.shape)
            if isinstance(weight_bounds[0], (float, int)):
                self.uB = np.ones(self.mean.shape) * weight_bounds[1]
            else:
                self.uB = np.array(weight_bounds[1]).reshape(self.mean.shape)

        self.w = []  # solution
        self.ls = []  # lambdas
        self.g = []  # gammas
        self.f = []  # free weight

        self.frontier_values = None

        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        else:
            tickers = list(range(len(self.mean)))
        super().__init__(len(tickers), tickers)
    
    @staticmethod
    def _infnone(x):
        return float('-inf') if x is None else x
    
    def _init_algo(self):
        a = np.zeros((self.mean.shape[0]), dtype=[('id', int), ('mu', float)])
        b = [self.mean[i][0] for i in range(self.mean.shape[0])]
        a[:] = list(zip(list(range(self.mean.shape[0])), b))
        b = np.sort(a, order = 'mu')
        i, w = b.shape[0], np.copy(self.lB)
        while sum(w)<1:
            i -=1
            w[b[i][0]] = self.uB[b[i][0]]
        w[b[i][0]] +=1 - sum(w)
        return [b[i][0]], w

    def _reduce_matrix()
        pass
    def _get_matrices(self, f):
        covarF = self._reduce_matrix(self.cov_matrix, f, f)
        pass
    def _solve(self):
        f, w = self._init_algo()
        self.w.append(np.copy(w))
        self.ls.append(None)
        self.g.append(None)
        self.f.append(f[:])

        while True:
            l_in = None
            if len(f)>1:
                covarF, covarFB, meanF, wB = self._get_matrices(f)
        pass