from pymoo.core.problem import Problem
import torch
import numpy as np


class PymooProblem(Problem):
    def __init__(self, var, bounds, acq_function, q):
        super().__init__(n_var=var, n_obj=1, n_constr=0, xl=bounds[0], xu=bounds[1])
        self.acquisition_function = acq_function
        self.q = q

    def _evaluate(self, x, out, *args, **kwargs):
        # recover q dimension to evaluate acq
        x = x.reshape((x.shape[0], self.q, x.shape[1] // self.q))
        # x has to be converted to Tensor
        x = torch.from_numpy(x)
        acq_evaluation = -self.acquisition_function(x).detach()
        # acq_evaluation will be a tensor. It has to be converted to numpy
        out["F"] = np.asarray([acq_evaluation.numpy()]).T
