import os
import sys

import argparse
def to_bool(s):
    if s.lower() in ['true', '1']:
        return True
    if s.lower() in ['false', '0']:
        return False
    raise ValueError(f'Not a bool: {s}')
def to_int_or_None(s):
    if s.lower() in ['', '-', 'none']:
        return None
    return int(s)
def str_or_None(s):
    if s.lower() in ['', '-', 'none']:
        return None
    return s
def float_as_str(s):
    _ = float(s)
    return s
parser = argparse.ArgumentParser()
parser.add_argument('first_trial', type=int)
parser.add_argument('last_trial', type=int, nargs='?')
parser.add_argument('--out-dir', type=str_or_None, default=None)
parser.add_argument('--restart', type=to_bool, default=True, help="Restart from existing data.")
parser.add_argument('--prob', type=str, default="DTLZ2(5,4)")
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--mode', choices=['PLMOO', 'MOO+PL', 'analytic'], default='PLMOO')
parser.add_argument('--n-threads', type=to_int_or_None, default=None)
parser.add_argument('--preference-model', choices=['comp-pref-gp', 'pw-vari-gp', 'pw-gp', 'vari-pref-gp', 'comp-vari-pref-gp'], default='comp-pref-gp', help="""
    Controls the model for the preference: g(x) vs g(f).
    The next query is always optimized over x
    'comp-pref-gp': x->f(x)->g(f(x)) [evaluates f]
    'pw-vari-gp':   x->g(x)          [no evaluations of f]
    'pw-gp':        x->g(x)          [no evaluations of f]
""")
parser.add_argument('--paretro-iter', type=int, default=250)
parser.add_argument('--paretro-pop', type=to_int_or_None, default=400)
parser.add_argument('--utility', choices=['lin', 'lin5', 'exp2', 'exp-2', 'min', 'min2', 'p3-0.2', 'p4-0.2', 'softmin', 'RF-VLMOP3-0.3', 'ackley'], default='lin')
parser.add_argument('--noise-type', choices=['logit'], default='logit')
parser.add_argument('--noise', type=str, default="1e-4", help="""
    Either:
        float: Set the noise parameter directly.
        float@float: to set error rate for top points: E.g. 0.3@0.01 for 30%% error rate at the top 1%% of points.
    Default is close to noise-free preferences.
""")
args = parser.parse_args()

if args.out_dir is None:
    args.out_dir = f"{args.prob}_{args.utility}_{args.noise_type}({args.noise})"

if args.n_threads is not None:
    # set befrore we import torch and numpy
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import torch
if args.n_threads is not None:
    torch.set_num_threads(args.n_threads)



import torch

from botorch.settings import debug
from botorch.test_functions.multi_objective import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ7
from botorch.test_functions.synthetic import Ackley
from pymoo.problems import get_problem

from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager
from src.get_noise_level import get_noise_level


# Objective function
def VLMOP3(dim, num_objectives, negate=False):
    assert dim == 2
    assert num_objectives == 3
    # rescale x from [0,1]^2 to [-3,3]^3
    trafo = lambda x: x*6 - 3
    return lambda x: (-1 if negate else 1) * attribute_func_VLMOP3(trafo(x))

def nI(dim, num_objectives, negate=False):
    assert dim == num_objectives
    return lambda x: (-1 if negate else 1) * -x

def WFG3(dim, num_objectives, negate=False, k_arg={}):
    wfg3 = get_problem("WFG3", n_var=dim, n_obj=num_objectives, **k_arg)
    def inner(x):
        if x.ndim == 1:
            return (-1 if negate else 1) * torch.from_numpy(wfg3.evaluate(x.numpy()))
        else:
            s = x.shape[:-1]
            return (-1 if negate else 1) * torch.from_numpy(wfg3.evaluate(x.numpy().reshape(-1, dim)).reshape(*s, num_objectives))
    return inner

def WFG4(dim, num_objectives, negate=False, k_arg={}):
    wfg4 = get_problem("WFG4", n_var=dim, n_obj=num_objectives, **k_arg)
    def inner(x):
        if x.ndim == 1:
            return (-1 if negate else 1) * torch.from_numpy(wfg4.evaluate(x.numpy()))
        else:
            s = x.shape[:-1]
            return (-1 if negate else 1) * torch.from_numpy(wfg4.evaluate(x.numpy().reshape(-1, dim)).reshape(*s, num_objectives))
    return inner

def WFG3k8(dim, num_objectives, negate=False, k_arg={}):
    return WFG3(dim, num_objectives, negate=negate, k_arg=dict(k=8))
def WFG4k8(dim, num_objectives, negate=False, k_arg={}):
    return WFG4(dim, num_objectives, negate=negate, k_arg=dict(k=8))

import re
re_prob = re.compile(r"(?P<prob>nI|DTLZ[1-57]|VLMOP3|WFG[34](k8)?)\((?P<dx>[1-9][0-9]*),(?P<df>[1-9][0-9]*)\)")
if m := re_prob.match(args.prob):
    input_dim = int(m['dx'])
    num_attributes = int(m['df'])
    attribute_func = eval(m['prob'])(dim=input_dim, num_objectives=num_attributes, negate=True)
    # negate=True: Multi-objective maximization problem
    #   utility should increase monotonically: high utility for large objectives
    #   monotonity inducing points prefere large objectives
    #
    # negate=False: Multi-objective minimization problem
    #   utility should decrease monotonically: high utility for small objectives
    #   monotonity inducing points prefere small objectives
else:
    assert False, f"Unkown problem: {args.prob}"

def attribute_func_VLMOP3(X):
    # [-3, 3]^2 -> IR^3
    assert X.shape[-1] == 2
    x1 = X[...,0]
    x2 = X[...,1]
    # Minimizatin problem
    f = torch.stack([
        0.5 * (x1**2 + x2**2) + torch.sin(x1**2 + x2**2),
        (3*x1 - 2*x2 + 4)**2 / 8 + (x1 - x2 + 1)**2 / 27 + 15,
        1 / (x1**2 + x2**2 + 1) - 1.1 * torch.exp(-x1**2 - x2**2),
    ], axis=-1)
    return f

def attribute_func(X, attribute_func=attribute_func):
    return attribute_func(X)


# must be monotonically increasing
def utility_func(Y):
    if args.utility == 'ref':
        reference_vector = attribute_func(torch.tensor([0.0, 0.5, 1.0] + [0.5] * (num_attributes - 3))[:num_attributes])
        return -torch.square(Y - reference_vector).sum(dim=-1)
    # optima are extrema
    elif args.utility == 'lin':
        return Y.sum(dim=-1)
    elif args.utility == 'lin5':
        return (Y[..., ::2].sum(dim=-1) + 5 * Y[..., 1::2].sum(dim=-1))
    elif args.utility == 'exp2':
        return torch.log((torch.exp(Y)**2.0).sum(dim=-1))
    # optimum is centered but very flat
    elif args.utility == 'exp-2':
        return -torch.log((torch.exp(Y)**-2.0).sum(dim=-1))
    # optimum is slightly off center but very flat (p3) / flat (p4)
    elif args.utility in ['p3-0.2', 'p4-0.2']:
        reference_vector = torch.zeros(num_attributes)
        p,f,r = {
            'p3-0.2': (3, 1, 0.2),
            'p4-0.2': (4, 1, 0.2),
        }[args.utility]
        reference_vector[1::2] = 0.0
        reference_vector[ ::2] = r
        Y = Y.clone()
        Y[..., 1::2] *= 1.0
        Y[...,  ::2] *= f
        return -(torch.abs(Y - reference_vector)**p).sum(dim=-1)**(1/p)
    # optimum is centered and sharp
    elif args.utility == 'min':
        return Y.min(dim=-1)[0]
    # optimum is slightly off center and sharp
    elif args.utility == 'min2':
        Y = Y.clone()
        Y[..., 1::2] *= 2
        return Y.min(dim=-1)[0]

    elif args.utility == 'softmin':
        t = 4
        return -torch.log(torch.sum(torch.exp(-t*Y), dim=-1)) / 4

    elif args.utility in [f'RF-VLMOP3-{t}' for t in ['0.3']]:
        # used in https://arxiv.org/pdf/1911.05934 with theta in [0.1, 0.5]
        # t=0.1; plt.contour(np.mean((1 - np.exp(-t*np.mgrid[0:10:100j, 0:10:100j]))/t, axis=0))
        theta = {f'RF-VLMOP3-{t}': float(t) for t in ['0.3']}[args.utility]
        return torch.mean((1 - torch.exp(-theta * Y)) / theta, axis=-1)

    elif args.utility == 'ackley':
        Y_unnorm = (4.0 * Y) - 2.0 # [0,1] -> [-2,2]
        ackley = Ackley(dim=input_dim)
        return -ackley.evaluate_true(Y_unnorm)


    # DTLZ2 front is a circle segment
    # https://www.wolframalpha.com/input?i=plot+x%5E2+%2B+y%5E2+%3D+1%2C+log%28exp%28x%29%5E2+%2B+exp%28y%29%5E2%29+%3D+-0.7%2C+log%28exp%28x%29%5E2+%2B+exp%28y%29%5E2%29+%3D+-0.6%2C+-log%28exp%28x%29%5E-2+%2B+exp%28y%29%5E-2%29+%3D+-2.2%2C+-log%28exp%28x%29%5E-2+%2B+exp%28y%29%5E-2%29+%3D+-2.1%2C+-sqrt%28max%280%2C-x%29%5E2+%2B+max%280%2C-y%29%5E2%29+%3D+-0.5


# Algos
algo = "EUBO"
model_type = {
    'comp-pref-gp': "composite_preferential_gp",
    'pw-vari-gp': "pairwise_kernel_variational_gp",
    'vari-pref-gp': "variational_preferential_gp",
    'comp-vari-pref-gp': "composite_variational_preferential_gp",
    'pw-gp': "pairwise_gp",
}[args.preference_model]

if args.mode == 'analytic':

    try:
        print(f"Generate Pareto front")
        from pymoo.problems import get_problem
        if m['prob'].endswith('k8'):
            prob = get_problem(m['prob'][:-len('k8')], n_var=input_dim, n_obj=num_attributes, k=8)
        else:
            prob = get_problem(m['prob'], n_var=input_dim, n_obj=num_attributes)
        pf = -prob.pareto_front() # negate if negate=True
        ps = prob.pareto_set()
    except Exception as e:
        print(f"Failed generating Pareto front: {e}")
        pf = None
        ps = None

    # optimize analytically
    from scipy.optimize import minimize
    import numpy as np

    def objective_np(X):
        return -utility_func(attribute_func(torch.from_numpy(X))).numpy()
    def objective(X):
        return utility_func(attribute_func(X))

    method = 'botorch'
    method = 'scipy'
    restarts = 10000

    file = f'{args.out_dir}/{args.prob}_{args.utility}.npz'

    if method == 'botorch':
        from botorch.optim.optimize import optimize_acqf
        print(f"running {restarts} optimizations with botorch")
        x, u = optimize_acqf(
            objective,
            torch.tensor([(0.0, 1.0)] * input_dim).T,
            1,
            restarts,
            raw_samples=restarts,
        )
        f = attribute_func(x).numpy()[0]
        x = x.numpy()[0]
        u = u.numpy()[0]

    elif method == 'scipy':
        print(f"running {restarts} optimizations with scipy")
        try:
            prev = np.load(file, allow_pickle=True)
            first = prev['restarts']
            print(f"loaded {first} previous results")
            if first >= restarts and os.environ.get('EXTRA_INIT') is None and os.environ.get('EXTRA_INIT_2') is None and os.environ.get('EXTRA_MOO') is None:
                print(f"previous results have more restarts. Abort.")
                sys.exit()
        except Exception as e:
            print(f"No old results: {e}")
            prev = dict()
            first = 0


        if (e := os.environ.get('EXTRA_INIT')) is not None:
            # for p in "WFG3(6,3)" "WFG4(6,3)" "WFG3(12,6)" "WFG3k8(14,9)" "WFG4(12,6)" "WFG4k8(14,9)"; for u in "p3-0.2"; env EXTRA_INIT="results_pcsgs-ga-nomenu/""$p""_""$u""_logit(1e-4)/ results_pcsgs-ga-nomenu/""$p""_""$u""_logit(0.3@0,01)/" python3 dtlz2_runner.py 0 --restart 0 --steps 0 --mode analytic --prob "$p" --utility "$u" --out-dir analytic; end; end; wait
            ress_init = []
            best_init = None

            import pathlib
            xs = np.ones((0, input_dim))
            us = np.ones((0,))
            for ei in e.split(' '):
                print(f"load from {ei}")
                for hist in pathlib.Path(ei).glob("**/history*npz"):
                    print(f"  {hist}")
                    with np.load(hist) as h:
                        xs = np.append(xs, h["queries"][:,0,:], axis=0)
                        xs = np.append(xs, h["queries"][:,1,:], axis=0)
                        us = np.append(us, h["utilities"][:,0], axis=0)
                        us = np.append(us, h["utilities"][:,1], axis=0)
            k = 500
            print(f"get best {k} from {us.shape[0]}")
            ixs = np.argsort(us)[-k:]
            xs_best = xs[ixs, :]
            for r, init in enumerate(xs_best):
                res = minimize(objective_np, init, bounds=[(0.0, 1.0)] * input_dim)
                ress_init.append(res)
                if np.isfinite(res.fun):
                    if (best_init is None or res.fun < best_init.fun):
                        print(f"{-r:>5}: {res.fun:.16e}")
                        best_init = res
                else:
                    print(f"{-r}: got nan result")

            f_init = attribute_func(torch.from_numpy(best_init.x)).numpy()
            x_init = best_init.x
            u_init = -best_init.fun
            print(x_init)
            print(f_init)
            print(u_init)
        else:
            x_init = prev.get('x_init')
            f_init = prev.get('f_init')
            u_init = prev.get('u_init')
            if np.asarray(x_init)[()] is None:
                x_init = None
                f_init = None
                u_init = None


        if (e := os.environ.get('EXTRA_INIT_2')) is not None:
            # for p in "WFG3(6,3)" "WFG4(6,3)" "WFG3(12,6)" "WFG3k8(14,9)" "WFG4(12,6)" "WFG4k8(14,9)"; for u in "p3-0.2" softmin lin; env EXTRA_INIT="results_pcsgs-ga-nomenu/""$p""_""$u""_logit(1e-4)/ results_pcsgs-ga-nomenu/""$p""_""$u""_logit(0.3@0,01)/" python3 dtlz2_runner.py 0 --restart 0 --steps 0 --mode analytic --prob "$p" --utility "$u" --out-dir analytic; end; end; wait
            ress_init_2 = []
            best_init_2 = None

            import pathlib
            print(f"load from {e}")
            xs_init_2 = None
            fs_init_2 = -np.loadtxt(e, skiprows=1, delimiter=',')
            us_init_2 = utility_func(torch.from_numpy(fs_init_2)).numpy()

            ix = np.argmax(us_init_2)
            x_init_2 = None
            f_init_2 = fs_init_2[ix,:]
            u_init_2 = us_init_2[ix]
            print(x_init_2)
            print(f_init_2)
            print(u_init_2)
        else:
            x_init_2 = prev.get('x_init_2')
            f_init_2 = prev.get('f_init_2')
            u_init_2 = prev.get('u_init_2')
            if np.asarray(f_init_2)[()] is None:
                x_init_2 = None
                f_init_2 = None
                u_init_2 = None


        if (e := os.environ.get('EXTRA_MOO')) is not None:
            from pymoo.optimize import minimize as pymoo_minimize
            from pymoo.util.ref_dirs import get_reference_directions
            from pymoo.algorithms.moo.nsga3 import NSGA3
            print("Run MOO")
            ref_dirs = get_reference_directions("energy", num_attributes, 1000, seed=1)
            algorithm = NSGA3(ref_dirs, pop_size=10_000)
            sys.path.append('../src')
            from utils import FuncProblem
            bounds = np.array([(0.0, 1.0)] * input_dim).T
            prob_ = FuncProblem(var=input_dim, n_obj=num_attributes, bounds=bounds, attribute_func=attribute_func)
            res = pymoo_minimize(prob_, algorithm, seed=1, termination=('n_gen', 150), verbose=True)
            xs_moo = res.X
            fs_moo = -res.F

            print(fs_moo.shape)

            us_moo = utility_func(torch.from_numpy(fs_moo)).numpy()
            best = np.argmax(us_moo)

            x_moo = xs_moo[best,:]
            f_moo = fs_moo[best,:]
            u_moo = us_moo[best]
            print(x_moo)
            print(f_moo)
            print(u_moo)
        else:
            x_moo = prev.get('x_moo')
            f_moo = prev.get('f_moo')
            u_moo = prev.get('u_moo')
            xs_moo = prev.get('xs_moo')
            fs_moo = prev.get('fs_moo')
            us_moo = prev.get('us_moo')
            if np.asarray(x_moo)[()] is None:
                x_moo = None
                f_moo = None
                u_moo = None
                xs_moo = None
                fs_moo = None
                us_moo = None



        if (e := os.environ.get('EXTRA_CMA')) is not None:
            from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
            sys.path.append('../src')
            from utils import FuncProblem
            from pymoo.optimize import minimize as pymoo_minimize
            bounds = np.array([(0.0, 1.0)] * input_dim).T
            prob_ = FuncProblem(var=input_dim, n_obj=num_attributes, bounds=bounds, attribute_func=objective_np)
            algorithm = CMAES(restarts=4)
            res = pymoo_minimize(prob_, algorithm, verbose=False, seed=r)
            x_cma = res.X.squeeze()
            f_cma = attribute_func(torch.from_numpy(x_cma)).numpy()
            u_cma = res.F.squeeze()
            print(x_cma)
            print(f_cma)
            print(u_cma)
        else:
            x_cma = prev.get('x_cma')
            f_cma = prev.get('f_cma')
            u_cma = prev.get('u_cma')
            if np.asarray(x_cma)[()] is None:
                x_cma = None
                f_cma = None
                u_cma = None


        ress_rand = []
        best_rand = None
        if first < restarts:
            for r in range(first, restarts):
                np.random.seed(r)
                init = np.random.rand(input_dim)
                res = minimize(objective_np, init, bounds=[(0.0, 1.0)] * input_dim)
                ress_rand.append(res)
                if np.isfinite(res.fun):
                    if (best_rand is None or res.fun < best_rand.fun):
                        print(f"{r:>5}: {res.fun:.16e}")
                        best_rand = res
                else:
                    print(f"{r}: got nan result")
            f_rand = attribute_func(torch.from_numpy(best_rand.x)).numpy()
            x_rand = best_rand.x
            u_rand = -best_rand.fun

        if best_rand is None or (prev.get("u") is not None and prev["u"] >= u_rand):
            if best_rand is not None:
                print(f"Previous results were better {prev['u']} >= {u_rand}")
            else:
                print(f"Keep previous results")
            x_rand = prev["x"]
            f_rand = prev["f"]
            u_rand = prev["u"]

        # get best of all approaches
        print("All")
        print([prev.get("u"), u_rand, u_init, u_init_2, u_moo])

        x = x_rand
        f = f_rand
        u = u_rand
        if u_init is not None and u_init >= u:
            x = x_init
            f = f_init
            u = u_init
        if u_init_2 is not None and u_init_2 >= u:
            x = x_init_2
            f = f_init_2
            u = u_init_2
        if u_moo is not None and u_moo >= u:
            x = x_moo
            f = f_moo
            u = u_moo
        if u_cma is not None and u_cma >= u:
            x = x_cma
            f = f_cma
            u = u_cma

    print("Opt")
    print(x)
    print(f)
    print(u)

    np.savez(file, problem=args.prob, utility=args.utility, method=method,
        x=x, f=f, u=u,
        # individual results
        x_rand=x_rand, f_rand=f_rand, u_rand=u_rand, restarts=restarts,
        x_init=x_init, f_init=f_init, u_init=u_init,
        x_moo=x_moo, f_moo=f_moo, u_moo=u_moo,
        xs_moo=xs_moo, fs_moo=fs_moo, us_moo=us_moo,
        x_cma=x_cma, f_cma=f_cma, u_cma=u_cma,
        pf=pf, ps=ps,
    )
    print(file)
    sys.exit()

# parse response noise
if '@' in args.noise:
    error_rate, top_proportion = [float(f) for f in args.noise.split('@')]

    print(f"Optimize noise for error rate {error_rate} in top {top_proportion}")
    def objective(X):
        return utility_func(attribute_func(X))
    args.noise, _ = get_noise_level(objective, input_dim, args.noise_type, error_rate, top_proportion, 10000)
else:
    args.noise = float(args.noise)
print(f"Noise parameter={args.noise}")

# Run experiment
first_trial = args.first_trial
last_trial = args.last_trial if args.last_trial is not None else args.first_trial

experiment_manager(
    problem=args.out_dir,
    attribute_func=attribute_func,
    utility_func=utility_func,
    input_dim=input_dim,
    num_attributes=num_attributes,
    comp_noise_type=args.noise_type,
    comp_noise=args.noise,
    algo=algo,
    model_type=model_type,
    batch_size=2,
    num_init_queries=2 * (input_dim + 1),
    num_algo_iter=args.steps,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=args.restart,
    mode=args.mode,
    num_paretro_iter=args.paretro_iter,
    num_paretro_pop=args.paretro_pop,
)
