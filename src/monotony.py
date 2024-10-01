import torch
from botorch.utils.sampling import draw_sobol_samples


def gen_monotonity_inducing_points(n, bounds, seed=0, opt='max'):
    assert bounds.ndim == 2
    assert bounds.shape[0] == 2
    dim = bounds.shape[1]

    mips     = draw_sobol_samples(bounds=bounds, n=n, q=1, seed=seed)
    mips_dir = draw_sobol_samples(bounds=torch.tensor([[0.0]*dim,[1.0]*dim]), n=n, q=1, seed=seed+1)
    assert mips.shape == (n, 1, dim)
    mips = mips[:,0,:]
    mips_dir = mips_dir[:,0,:]
    assert mips.shape == (n, dim)

    mips_normalized = (mips - bounds[0,:]) / (bounds[1,:] - bounds[0,:])
    #
    #   f2
    #   │       ╎
    #   │       ╎ x < .
    #   ├╴╴╴╴╴╴╴x╶╶╶╶╶╶╶
    #   │ . < x ╎
    #   │       ╎
    #   └───────┴─────── f1
    #
    hv_smaller = torch.prod(mips_normalized, axis=1)
    hv_larger = torch.prod(1-mips_normalized, axis=1)
    p_other_smaller = hv_smaller / (hv_larger + hv_smaller)
    assert p_other_smaller.shape == (n,)
    rand = draw_sobol_samples(bounds=torch.tensor([[0.0],[1.0]]), n=n, q=1, seed=seed+2)
    assert rand.shape == (n, 1, 1)
    rand = rand[:,0,0]

    other_smaller = rand < p_other_smaller

    mips_other = mips + mips_dir * torch.where(other_smaller[:,None], -(mips - bounds[0,:]), bounds[1,:] - mips)

    # response: argmax(u(f_mip), u(f_other))
    if opt == 'max':
        # maximization problem
        return mips, mips_other, 1-other_smaller.int()
    else:
         # minimization problem
        return mips, mips_other, other_smaller.int()