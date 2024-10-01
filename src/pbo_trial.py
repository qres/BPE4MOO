#!/usr/bin/env python3

from typing import Callable, Dict, Optional, Literal

import numpy as np
import os
import sys
import time
import torch
from botorch.models.model import Model
from botorch.sampling import SobolQMCNormalSampler
from torch import Tensor
from botorch.acquisition import PosteriorMean

from src.acquisition_functions.emov import (
    ExpectedMaxObjectiveValue,
    qExpectedMaxObjectiveValue,
)
from src.acquisition_functions.thompson_sampling import gen_thompson_sampling_query
from src.utils import (
    fit_model,
    generate_initial_data,
    generate_random_queries,
    get_attribute_and_utility_vals,
    generate_responses,
    optimize_acqf_and_get_suggested_query,
    maximize_multi_objective,
    compute_posterior_mean_maximizer,
    optimize_acqf_and_get_suggested_query_discrete_brute_force,
    optimize_acqf_and_get_suggested_query_discrete_sequiential,
    model_has_grad_x,
)


# this function runs a single trial of a given problem
# see more details about the arguments in experiment_manager.py
def pbo_trial(
    problem: str,
    attribute_func: Callable,
    utility_func: Callable,
    input_dim: int,
    num_attributes: int,
    comp_noise_type: str,
    comp_noise: float, # only for computing the DM responses. Noise level is learned by the model.
    algo: str,
    batch_size: int,
    num_init_queries: int,
    num_algo_iter: int,
    trial: int,
    restart: bool,
    model_type: str,
    ignore_failures: bool,
    algo_params: Optional[Dict] = None,
    # mode: str = 'PLMOO',
    mode: Literal['PLMOO', 'MOO+PL'] = 'PLMOO',
    num_paretro_iter: int = 0, # only 'MOO+PL'
    num_paretro_pop: Optional[int] = None, # only 'MOO+PL'
) -> None:
    if mode == 'PLMOO':
        algo_id = f"{algo}__{model_type}__{mode}"
    elif mode == 'MOO+PL':
        algo_id = f"{algo}__{model_type}__{mode}__p{num_paretro_pop},i{num_paretro_iter}"

    # get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = (
        project_path + "/experiments/results/" + problem + "/" + algo_id + "/"
    )

    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])


    counters = {}
    _conter_name = None
    def set_counter(name):
        nonlocal _conter_name
        _conter_name = name
        counters[name] = counters.get(name, [])
        counters[name].append(0)
    def counted_attribute_func(*args, **kwargs):
        assert len(args) == 1
        shape_without_dim = args[0].shape[:-1]
        counters[_conter_name][-1] += shape_without_dim.numel()
        return attribute_func(*args, **kwargs)

    if restart:
        # check if training data is available
        try:
            # current available iterations
            file = results_folder + "history_" + str(trial) + ".npz"
            print(f"Loading previous data: {file}")
            with np.load(file, allow_pickle=True) as hist:
                queries = torch.from_numpy(hist['queries'])
                attribute_vals = torch.from_numpy(hist['attributes'])
                utility_vals = torch.from_numpy(hist['utilities'])
                responses = torch.from_numpy(hist['responses'])
                xs_paretro = torch.from_numpy(hist['xs_moo'])
                fs_paretro = torch.from_numpy(hist['fs_moo'])
                runtimes = list(hist['runtimes'])
                max_post_mean_x = list(hist['max_post_mean_x'])
                max_post_mean_f = list(hist['max_post_mean_f'])
                max_post_mean_u = list(hist['max_post_mean_u'])
                max_post_mean_query_x = list(hist['max_post_mean_query_x'])
                max_post_mean_query_f = list(hist['max_post_mean_query_f'])
                max_post_mean_query_u = list(hist['max_post_mean_query_u'])
                max_utility_vals_within_queries = list(hist['max_utilities_at_queries'])
                post_mean_at_queries = list([])
                post_var_at_queries = list([])
                menus = list(hist['menus'])
                counters = hist['counters'][()]

            # fit model
            set_counter('fit')
            t0 = time.time()
            model = fit_model(
                queries,
                responses,
                attribute_func=counted_attribute_func,
                model_type=model_type,
                likelihood=comp_noise_type,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            iteration = len(max_utility_vals_within_queries) - 1
            print(f"Restarting experiment from iteration {iteration}.")

        except Exception as e:
            print(f"Failed: Restarting experiment from available data. ({e})")
            print("  Starting from first interation.")
            iteration = None

    if not restart or iteration is None:
        # initial data
        set_counter('init')
        queries, attribute_vals, utility_vals, responses = generate_initial_data(
            num_queries=num_init_queries,
            batch_size=batch_size,
            input_dim=input_dim,
            attribute_func=counted_attribute_func,
            utility_func=utility_func,
            comp_noise_type=comp_noise_type,
            comp_noise=comp_noise,
            seed=trial,
        )

        # fit model
        set_counter('init_fit')
        t0 = time.time()
        model = fit_model(
            queries,
            responses,
            attribute_func=counted_attribute_func,
            model_type=model_type,
            likelihood=comp_noise_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # historical utility values at the maximum of the posterior mean
        set_counter('init_opt_max_post')
        posterior_mean_maximizer_x = compute_posterior_mean_maximizer(
            model=model, model_type=model_type, input_dim=input_dim
        )
        set_counter('init_max_post')
        posterior_mean_maximizer_f = counted_attribute_func(posterior_mean_maximizer_x)
        posterior_mean_maximizer_u = utility_func(posterior_mean_maximizer_f).item()
        max_post_mean_x = [posterior_mean_maximizer_x.numpy()]
        max_post_mean_f = [posterior_mean_maximizer_f.numpy()]
        max_post_mean_u = [posterior_mean_maximizer_u]

        # evaluate max posterior mean at data points
        set_counter('init_opt_max_post_query')
        posterior_mean_maximizer_query_x, _ = optimize_acqf_and_get_suggested_query_discrete_brute_force(PosteriorMean(model=model), standard_bounds, 1, queries.reshape(-1, input_dim)) # flatten all query points
        set_counter('init_max_post_query')
        posterior_mean_maximizer_query_f = counted_attribute_func(posterior_mean_maximizer_query_x)
        posterior_mean_maximizer_query_u = utility_func(posterior_mean_maximizer_query_f).item()
        max_post_mean_query_x = [posterior_mean_maximizer_query_x.numpy()]
        max_post_mean_query_f = [posterior_mean_maximizer_query_f.numpy()]
        max_post_mean_query_u = [posterior_mean_maximizer_query_u]

        # evaluate model at queries
        set_counter('init_model_at_query')
        with torch.no_grad():
            model_at_queries = model.posterior(queries.reshape(-1, input_dim))
            post_mean_at_queries = [model_at_queries.mean]
            post_var_at_queries = [model_at_queries.variance]

        # historical max utility values within queries and runtimes
        max_utility_val_within_queries = utility_vals.max().item()
        max_utility_vals_within_queries = [max_utility_val_within_queries]

        # historical menus
        menus = []

        # historical acquisition runtimes
        runtimes = []

        iteration = 0

    if mode == "MOO+PL":
        set_counter('moo')
        xs_paretro, fs_paretro = maximize_multi_objective(standard_bounds, counted_attribute_func, num_attributes, num_paretro_iter, num_paretro_pop)
        print(f"Paretro points: {xs_paretro.shape[0]}")
    else:
        xs_paretro, fs_paretro = None, None

    # optimization loop
    while iteration < num_algo_iter:
        t0_step = time.time()

        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + algo_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # new suggested query
        t0 = time.time()
        set_counter('opt_query')
        if mode == 'PLMOO':
            new_query, _ = get_new_suggested_query(
                algo=algo,
                model=model,
                batch_size=batch_size,
                input_dim=input_dim,
                algo_params=algo_params,
            )
        elif mode == 'MOO+PL':
            with torch.no_grad():
                new_query, _ = get_new_suggested_query_discrete_brute_force(
                    algo=algo,
                    model=model,
                    batch_size=batch_size,
                    input_dim=input_dim,
                    candidates=xs_paretro,
                )
        assert new_query.shape == (1, 2, input_dim)
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # get response at new query
        set_counter('query')
        new_attribute_vals, new_utility_vals = get_attribute_and_utility_vals(new_query, counted_attribute_func, utility_func)
        new_responses = generate_responses(
            new_utility_vals,
            noise_type=comp_noise_type,
            noise_level=comp_noise,
        )

        # update training data
        queries = torch.cat((queries, new_query))
        attribute_vals = torch.cat([attribute_vals, new_attribute_vals], 0)
        utility_vals = torch.cat([utility_vals, new_utility_vals], 0)
        responses = torch.cat((responses, new_responses))

        # fit model
        t0 = time.time()
        set_counter('fit')
        model = fit_model(
            queries,
            responses,
            attribute_func=counted_attribute_func,
            model_type=model_type,
            likelihood=comp_noise_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # compute and append current utility value at the maximum of the posterior mean
        set_counter('opt_max_post')
        t0 = time.time()
        posterior_mean_maximizer_x = compute_posterior_mean_maximizer(
            model=model, model_type=model_type, input_dim=input_dim
        )
        set_counter('max_post')
        posterior_mean_maximizer_f = counted_attribute_func(posterior_mean_maximizer_x)
        posterior_mean_maximizer_u = utility_func(posterior_mean_maximizer_f).item()
        max_post_mean_x.append(posterior_mean_maximizer_x.numpy())
        max_post_mean_f.append(posterior_mean_maximizer_f.numpy())
        max_post_mean_u.append(posterior_mean_maximizer_u)
        print(f"Utility value at the maximum of the posterior mean: {posterior_mean_maximizer_u}")
        t1 = time.time()
        post_max_time = t1 - t0

        # evaluate max posterior mean at data points
        set_counter('opt_max_post_query')
        t0 = time.time()
        posterior_mean_maximizer_query_x, _ = optimize_acqf_and_get_suggested_query_discrete_brute_force(PosteriorMean(model=model), standard_bounds, 1, queries.reshape(-1, input_dim)) # flatten all query points
        posterior_mean_maximizer_query_f = counted_attribute_func(posterior_mean_maximizer_query_x)
        set_counter('max_post_query')
        posterior_mean_maximizer_query_u = utility_func(posterior_mean_maximizer_query_f).item()
        max_post_mean_query_x.append(posterior_mean_maximizer_query_x.numpy())
        max_post_mean_query_f.append(posterior_mean_maximizer_query_f.numpy())
        max_post_mean_query_u.append(posterior_mean_maximizer_query_u)
        t1 = time.time()
        post_max_query_time = t1 - t0

        # evaluate model at queries
        set_counter('model_at_query')
        with torch.no_grad():
            model_at_queries = model.posterior(queries.reshape(-1, input_dim))
            post_mean_at_queries.append(model_at_queries.mean)
            post_var_at_queries.append(model_at_queries.variance)

        # append current max utility val within queries
        max_utility_val_within_queries = utility_vals.max().item()
        max_utility_vals_within_queries.append(max_utility_val_within_queries)
        print(f"Max utility value within queries: {max_utility_val_within_queries}")

        # compute suggested menue
        menue_sizes = [1, 4, 16]
        menus.append({})
        set_counter('menus')

        t0 = time.time()
        if iteration < 10 or iteration % 5 == 0:
            if mode == 'MOO+PL' and len(menue_sizes) > 0:
                # optimizes sequentially -> only do it once
                m = np.max(menue_sizes)
                print(f"Prepare menue {m}")
                with torch.no_grad():
                    full_menu_xs, full_menu_a = get_new_suggested_query_discrete_sequiential(
                        algo="EUBO",
                        model=model,
                        batch_size=m,
                        input_dim=input_dim,
                        candidates=xs_paretro,
                    )
                    assert full_menu_xs.shape == torch.Size([1, m, input_dim])
                    assert len(full_menu_a) == m

            for m in menue_sizes:
                print(f"Compute menue {m}")
                if mode == 'PLMOO':
                    xs, a = get_new_suggested_query(
                        algo="EUBO",
                        model=model,
                        batch_size=m,
                        input_dim=input_dim,
                        algo_params=algo_params,
                    )
                elif mode == 'MOO+PL':
                    xs, a = full_menu_xs[:,:m], full_menu_a[m-1]
                    print(f"{xs}, {a}")
            
                f = attribute_func(xs)
                u = utility_func(f)

                menus[-1][m] = dict(
                    # optimized values
                    x = xs.numpy(),
                    a = a.numpy(),
                    # function and utility values at points
                    f = f.numpy(),
                    u = u.numpy(),
                    u_max = u.numpy().max(),
                )
        t1 = time.time()
        menu_time = t1 - t0

        t1_step = time.time()
        total_time = t1_step - t0_step
        print(f"{iteration}: Times [s]: {total_time:.1e} | {acquisition_time:.1e} {model_training_time:.1e} {post_max_time:.1e} {post_max_query_time:.1e} {menu_time:.1e}")

        # save data
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        np.savez(
            results_folder + "history_" + str(trial) + ".npz",
            queries=queries.numpy(),
            attributes=attribute_vals.numpy(),
            utilities=utility_vals.numpy(),
            responses=responses.numpy(),
            xs_moo=xs_paretro.numpy() if xs_paretro is not None else None,
            fs_moo=fs_paretro.numpy() if fs_paretro is not None else None,
            runtimes=runtimes,
            max_post_mean_x=max_post_mean_x,
            max_post_mean_f=max_post_mean_f,
            max_post_mean_u=max_post_mean_u,
            max_post_mean_query_x=max_post_mean_query_x,
            max_post_mean_query_f=max_post_mean_query_f,
            max_post_mean_query_u=max_post_mean_query_u,
            max_utilities_at_queries=max_utility_vals_within_queries,
            post_mean_at_queries=np.array([a.numpy() for a in post_mean_at_queries], dtype=object),
            post_var_at_queries=np.array([a.numpy() for a in post_var_at_queries], dtype=object),
            menus=menus,
            counters=counters,
        )

    print(f"Done: {results_folder}")


def get_new_suggested_query(
    algo: str,
    model: Model,
    batch_size,
    input_dim: int,
    algo_params: Optional[Dict] = None,
) -> Tensor:
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 2 * input_dim * batch_size
    raw_samples = 60 * input_dim * batch_size
    batch_initial_conditions = None

    if algo == "Random":
        return generate_random_queries(
            num_queries=1, batch_size=batch_size, input_dim=input_dim
        )
    elif algo == "ANALYTIC_EUBO":
        acquisition_function = ExpectedMaxObjectiveValue(model=model)
    elif algo == "EUBO":
        sampler = SobolQMCNormalSampler(sample_shape=64)
        acquisition_function = qExpectedMaxObjectiveValue(model=model, sampler=sampler)
    elif algo == "TS":
        return gen_thompson_sampling_query(
            model, batch_size, standard_bounds, 2 * input_dim, 60 * input_dim
        )

    use_f_grad = model_has_grad_x(model)

    new_query, new_a = optimize_acqf_and_get_suggested_query(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        use_f_grad=use_f_grad,
    )

    new_query = new_query.unsqueeze(0)
    return new_query, new_a


def get_new_suggested_query_discrete_brute_force(
    algo: str,
    model: Model,
    batch_size,
    input_dim: int,
    candidates: Tensor,
) -> Tensor:
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])

    if algo == "Random":
        assert False
    elif algo == "ANALYTIC_EUBO":
        acquisition_function = ExpectedMaxObjectiveValue(model=model)
    elif algo == "EUBO":
        sampler = SobolQMCNormalSampler(sample_shape=64)
        acquisition_function = qExpectedMaxObjectiveValue(model=model, sampler=sampler)
    elif algo == "TS":
        assert False

    new_query, new_a = optimize_acqf_and_get_suggested_query_discrete_brute_force(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=batch_size,
        candidates=candidates,
    )

    new_query = new_query.unsqueeze(0)
    return new_query, new_a

def get_new_suggested_query_discrete_sequiential(
    algo: str,
    model: Model,
    batch_size,
    input_dim: int,
    candidates: Tensor,
) -> Tensor:
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])

    if algo == "Random":
        assert False
    elif algo == "ANALYTIC_EUBO":
        acquisition_function = ExpectedMaxObjectiveValue(model=model)
    elif algo == "EUBO":
        sampler = SobolQMCNormalSampler(sample_shape=64)
        acquisition_function = qExpectedMaxObjectiveValue(model=model, sampler=sampler)
    elif algo == "TS":
        assert False

    new_query, new_a = optimize_acqf_and_get_suggested_query_discrete_sequiential(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=batch_size,
        candidates=candidates,
    )

    new_query = new_query.unsqueeze(0)
    return new_query, new_a
