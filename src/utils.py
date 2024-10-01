import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, PosteriorMean
from botorch.generation.gen import get_best_candidates
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.likelihoods import PairwiseLogitLikelihood, PairwiseProbitLikelihood
from botorch.optim.optimize import optimize_acqf, optimize_acqf_discrete
import torch
from torch import Tensor
from torch.distributions import Bernoulli, Normal, Gumbel
from typing import Optional, Callable
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP
from src.models.composite_model import CompositePreferentialGP
from src.models.variational_preferential_gp import VariationalPreferentialGP
from src.models.composite_variational_preferential_gp import CompositeVariationalPreferentialGP
from src.pymoo_problem import PymooProblem
from gpytorch.mlls.variational_elbo import VariationalELBO

import pymoo
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize

import os

class FuncProblem(pymoo.core.problem.Problem):
    def __init__(self, var, n_obj, bounds, attribute_func):
        super().__init__(n_var=var, n_obj=n_obj, n_constr=0, xl=bounds[0], xu=bounds[1])
        self.attribute_function = attribute_func

    def _evaluate(self, x, out, *args, **kwargs):
        # x has to be converted to Tensor
        x = torch.from_numpy(x)
        attrib_evaluation = -self.attribute_function(x).detach()
        # attrib_evaluation will be a tensor. It has to be converted to numpy
        out["F"] = attrib_evaluation.numpy()


def fit_model(
    queries: Tensor,
    responses: Tensor,
    attribute_func: Callable,
    model_type: str,
    likelihood: Optional[str] = "logit",
):
    if model_type == "pairwise_gp":
        datapoints, comparisons = training_data_for_pairwise_gp(queries, responses)

        if likelihood == "probit":
            likelihood_func = PairwiseProbitLikelihood()
        else:
            likelihood_func = PairwiseLogitLikelihood()
        model = PairwiseGP(
            datapoints,
            comparisons,
            likelihood=likelihood_func,
            jitter=1e-4,
        )

        mll = PairwiseLaplaceMarginalLogLikelihood(
            likelihood=model.likelihood, model=model
        )
        fit_gpytorch_mll(mll)
        model = model.to(device=queries.device, dtype=queries.dtype)
    elif model_type == "pairwise_kernel_variational_gp":
        model = PairwiseKernelVariationalGP(queries, responses)
    elif model_type == "composite_preferential_gp":
        model = CompositePreferentialGP(queries, responses, attribute_func)
    elif model_type == "variational_preferential_gp":
        model = VariationalPreferentialGP(queries, responses, attribute_func)
        model.train()
        model.likelihood.train()
        mll = VariationalELBO(
            likelihood=model.likelihood,
            model=model,
            num_data=2 * model.num_data,
        )
        mll = fit_gpytorch_mll(mll)
        model.eval()
        model.likelihood.eval()
    elif model_type == "composite_variational_preferential_gp":
        n_mips = int(os.environ.get("MIPS", "0"))
        print(f"add {n_mips} monotonity inducing points")

        model = CompositeVariationalPreferentialGP(queries, responses, attribute_func, num_monotonity_inducing_points=n_mips)
        model.train()
        model.likelihood.train()
        mll = VariationalELBO(
            likelihood=model.likelihood,
            model=model.utility_model,
            num_data=2 * model.num_data,
        )
        mll = fit_gpytorch_mll(mll)
        model.eval()
        model.likelihood.eval()
    return model


def generate_initial_data(
    num_queries: int,
    batch_size: int,
    input_dim: int,
    attribute_func,
    utility_func,
    comp_noise_type,
    comp_noise,
    seed: int = None,
):
    # generates initial data
    queries = generate_random_queries(num_queries, batch_size, input_dim, seed)
    attribute_vals, utility_vals = get_attribute_and_utility_vals(
        queries, attribute_func, utility_func
    )
    responses = generate_responses(utility_vals, comp_noise_type, comp_noise)
    return queries, attribute_vals, utility_vals, responses


def generate_random_queries(
    num_queries: int, batch_size: int, input_dim: int, seed: int = None
):
    # generate `num_queries` queries each constituted by `batch_size` points chosen uniformly at random
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        queries = torch.rand([num_queries, batch_size, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        queries = torch.rand([num_queries, batch_size, input_dim])
    return queries


def get_attribute_and_utility_vals(queries, attribute_func, utility_func):
    attribute_vals = attribute_func(queries)
    utility_vals = utility_func(attribute_vals)
    return attribute_vals, utility_vals


def generate_responses(utility_vals, noise_type, noise_level):
    # generate simulated comparisons based on true underlying objective
    corrupted_utility_vals = corrupt_vals(utility_vals, noise_type, noise_level)
    responses = torch.argmax(corrupted_utility_vals, dim=-1)
    return responses


def corrupt_vals(vals, noise_type, noise_level):
    # corrupts (attribute or utility) values to simulate noise in the DM's responses
    if noise_type == "noiseless":
        corrupted_vals = vals
    elif noise_type == "probit":
        normal = Normal(torch.tensor(0.0), torch.tensor(noise_level))
        noise = normal.sample(sample_shape=vals.shape)
        corrupted_vals = vals + noise
    elif noise_type == "logit":
        gumbel = Gumbel(torch.tensor(0.0), torch.tensor(noise_level))
        noise = gumbel.sample(sample_shape=vals.shape)
        corrupted_vals = vals + noise
    elif noise_type == "constant":
        corrupted_vals = vals.clone()
        n = vals.shape[0]
        for i in range(n):
            coin_toss = Bernoulli(noise_level).sample().item()
            if coin_toss == 1.0:
                corrupted_vals[i, 0] = vals[i, 1]
                corrupted_vals[i, 1] = vals[i, 0]
    return corrupted_vals


def training_data_for_pairwise_gp(queries, responses):
    num_queries = queries.shape[0]
    batch_size = queries.shape[1]
    datapoints = []
    comparisons = []
    for i in range(num_queries):
        best_item_id = batch_size * i + responses[i]
        comparison = [best_item_id]
        for j in range(batch_size):
            datapoints.append(queries[i, j, :].unsqueeze(0))
            if j != responses[i]:
                comparison.append(batch_size * i + j)
        comparisons.append(torch.tensor(comparison).unsqueeze(0))

    datapoints = torch.cat(datapoints, dim=0)
    comparisons = torch.cat(comparisons, dim=0)
    return datapoints, comparisons


def maximize_multi_objective(
    bounds: Tensor,
    attribute_func,
    n_attrib: int,
    steps: int,
    pop: int,
) -> Tensor:
    # maximize attributes ~> minimize negative attributes: FuncProblem flips sign of attributes
    problem = FuncProblem(var=bounds.shape[-1], n_obj=n_attrib, bounds=bounds.numpy(), attribute_func=attribute_func)

    algorithm = NSGA2(pop_size=pop)

    termination = pymoo.termination.get_termination("n_gen", steps)
    res = pymoo_minimize(problem, algorithm, termination)
    assert res.X.ndim == 2
    assert res.F.ndim == 2
    xs = torch.from_numpy(res.X)
    fs = torch.from_numpy(res.F)

    return xs, -fs

def optimize_acqf_and_get_suggested_query(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    batch_initial_conditions: Optional[Tensor] = None,
    batch_limit: Optional[int] = 2,
    init_batch_limit: Optional[int] = 30,
    use_f_grad: bool = False,
) -> Tensor:
    """Optimizes the acquisition function, and returns the candidate solution."""

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    dim = bounds.shape[1]
    aug_bounds = torch.cat([bounds for _ in range(batch_size)], dim=-1)
    assert aug_bounds.shape == (2, batch_size * dim)

    if not use_f_grad:
        with torch.no_grad():
            # maximize acq ~> minimize negative acq: PymooProblem flips sign of acq
            problem = PymooProblem(
                var=aug_bounds.shape[-1], bounds=aug_bounds.numpy(), acq_function=acq_func, q=batch_size,
            )
            algorithm = CMAES(restarts=4)
            res = pymoo_minimize(problem, algorithm, verbose=False)
            new_x = res.X
            new_a = res.F
            new_x = torch.from_numpy(new_x)
            new_a = -torch.from_numpy(new_a)
            if batch_size > 1:
                new_x = new_x.reshape(
                    batch_size,
                    new_x.shape[-1] // batch_size,
                )

    else:
        # we can compute gradients for the objectives 
        candidates, acq_values = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            batch_initial_conditions=batch_initial_conditions,
            options={
                "batch_limit": batch_limit,
                "init_batch_limit": init_batch_limit,
                "maxiter": 100,
                "nonnegative": False,
                "method": "L-BFGS-B",
            },
            return_best_only=False,
        )
        with torch.no_grad():
            candidates = candidates.detach()
            new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
            new_a = acq_func(new_x)

    return new_x, new_a

def batch(func, arg, batch_size):
    split = arg.split(batch_size, 0)
    return torch.cat([func(s) for s in split], 0)

def optimize_acqf_and_get_suggested_query_discrete_brute_force(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    candidates: Tensor,
) -> Tensor:
    n_candidates, dim = candidates.shape

    if batch_size == 1:
        qs = candidates.reshape(n_candidates, 1, dim)
        acq_vals = acq_func.forward(qs)
        ix_max = torch.argmax(acq_vals)
        return candidates[ix_max, :], acq_vals[ix_max]

    elif batch_size == 2:
        ijs = torch.combinations(torch.arange(n_candidates), r=batch_size, with_replacement=False)

        # acq takes (batch, q, dim)
        ls = candidates[ijs[:,0], :]
        rs = candidates[ijs[:,1], :]
        qs = torch.stack([ls, rs], dim=-2)
        assert qs.shape == (n_candidates * (n_candidates + 1) // 2 - n_candidates, 2, dim)
        print(f"maximizing over {n_candidates} candidates ~> {qs.shape[0]} queries ({qs.shape[0] * batch_size * dim * 8 / 1024**2:.1f} MiB queries)")

        acq_vals = batch(acq_func.forward, qs[:, None, :], 1_000)
        ix_max_a = torch.argmax(acq_vals)
        ix_max = ijs[ix_max_a, :]
        assert ix_max.shape == (2,)
        return candidates[ix_max, :], acq_vals[ix_max_a]
    
    else:
        assert False, f"Only for batch size 1 and 2. Got {batch_size}."

def optimize_acqf_and_get_suggested_query_discrete_sequiential(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    candidates: Tensor,
) -> Tensor:
    """
    uses sequential conditioning 
    returns all intermediate acqf values
    """

    # uses sequiential conditioning
    choices, a = optimize_acqf_discrete(
        acq_func,
        batch_size,
        candidates,
        max_batch_size=1024,
    )
    # return acq of all intermediate menus
    return choices, a

def compute_posterior_mean_maximizer(
    model: Model,
    model_type,
    input_dim: int,
) -> Tensor:
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 4 * input_dim
    raw_samples = 120 * input_dim

    use_f_grad = model_has_grad_x(model)

    post_mean_func = PosteriorMean(model=model)
    max_post_mean_func, _ = optimize_acqf_and_get_suggested_query(
        acq_func=post_mean_func,
        bounds=standard_bounds,
        batch_size=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        use_f_grad=use_f_grad,
    )
    return max_post_mean_func


# TODO clean up
def model_has_grad_x(model: Model):
    if type(model).__name__ in ['PairwiseKernelVariationalGP', 'VariationalPreferentialGP']:
        return  True
    elif type(model).__name__ in ['CompositePreferentialGP', 'CompositeVariationalPreferentialGP']:
        return  False
    else:
        print(f"WARN: unkown model type {type(model).__name__}")
        return  False
