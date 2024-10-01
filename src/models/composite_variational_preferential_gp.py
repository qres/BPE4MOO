from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal
import torch
from torch import Tensor

from src.monotony import gen_monotonity_inducing_points
from src.models.variational_preferential_gp import VariationalPreferentialGP
from gpytorch.likelihoods import Likelihood


class CompositeVariationalPreferentialGP(Model):
    def __init__(
        self,
        queries: Tensor,
        responses: Tensor,
        attribute_func: Tensor,
        num_monotonity_inducing_points: int = 0,
        growth_factor: float = 0.25,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        assert queries.shape[1] == 2
        input_dim = queries.shape[2]

        self.queries = queries
        self.responses = responses
        self.attribute_func = attribute_func

        attributes = attribute_func(self.queries)

        maxs = torch.max(torch.max(attributes, dim=1)[0], dim=0)[0]
        mins = torch.min(torch.min(attributes, dim=1)[0], dim=0)[0]

        if num_monotonity_inducing_points > 0:
            # generate random points in enlarged domain
            bounds = torch.stack([mins - growth_factor*(maxs - mins), maxs + growth_factor*(maxs - mins)])

            mip0, mip1, mip_responses = gen_monotonity_inducing_points(num_monotonity_inducing_points, bounds, seed=0)
            mip_queries = torch.stack([mip0, mip1], 1)
            assert attributes.shape[1:] == mip_queries.shape[1:], f"{queries.shape} {mip_queries.shape}"

            self.mip_queries = mip_queries
            self.mip_responses = mip_responses

            attributes_mips = torch.cat([attributes, mip_queries])
            responses_mips = torch.cat([self.responses, mip_responses])

        else:
            self.mip_queries = None
            self.mip_responses = None

            attributes_mips = attributes
            responses_mips = self.responses

        # normalize data
        if normalize:
            self.normalize_delta = mins
            self.normalize_scale = (maxs - mins)
            self.normalize = lambda Y: (Y - self.normalize_delta) / self.normalize_scale
            self.normalize_inv = lambda Y: Y * self.normalize_scale + self.normalize_delta
        else:
            self.normalize_delta = 0
            self.normalize_scale = 1
            self.normalize = lambda Y: Y
            self.normalize_inv = lambda Y: Y
        attributes_mips = self.normalize(attributes_mips)

        self.utility_model = VariationalPreferentialGP(
            attributes_mips, responses_mips
        )

    def posterior(self, X: Tensor, posterior_transform=None) -> MultivariateNormal:
        Y = self.attribute_func(X)
        Y = self.normalize(Y)
        return self.utility_model.posterior(Y)

    @property
    def likelihood(self) -> Likelihood:
        return self.utility_model.likelihood

    @property
    def num_data(self) -> int:
        return self.utility_model.num_data

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1
