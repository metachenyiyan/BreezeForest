from __future__ import print_function
import copy
import torch
from torch import nn
from torch.func import jacrev, vmap
from model.BreezeForest import BreezeForest
from model.tools import sigmoid


class MultiBF(torch.nn.Module):
    """
    Mixture of BreezeForest distributions.

    Models p(x) = sum_k pi_k * p_k(x), where each p_k is a BreezeForest
    normalizing flow and pi_k are jointly trained mixture weights.

    Training objective (per sample):
        log p(x) = logsumexp_k( log pi_k + log |det J_k(x)| )

    Sampling:
        1. k ~ Categorical(pi)
        2. z ~ Uniform(0.01, 0.99)^dim
        3. x = f_k^{-1}(z)
    """

    def __init__(self, n_components, dim, shapes, **bf_kwargs):
        """
        :param n_components: number of mixture components (K)
        :param dim: number of dimensions
        :param shapes: layer shapes for BreezeForest (deep-copied per component)
        :param bf_kwargs: additional kwargs passed to each BreezeForest
        """
        super(MultiBF, self).__init__()
        self.n_components = n_components
        self.dim = dim

        self.components = nn.ModuleList([
            BreezeForest(dim=dim, shapes=copy.deepcopy(shapes), **bf_kwargs)
            for _ in range(n_components)
        ])

        # Learnable mixture logits (uniform initialization -> equal weights)
        self.mixture_logits = nn.Parameter(torch.zeros(n_components))

    def get_mixture_log_weights(self):
        """Return log pi_k via log-softmax for numerical stability."""
        return torch.log_softmax(self.mixture_logits, dim=0)

    def get_mixture_weights(self):
        """Return pi_k via softmax."""
        return torch.softmax(self.mixture_logits, dim=0)

    def forward(self, x):
        """Forward pass through all components. Used for ActiNorm initialization."""
        for bf in self.components:
            bf.forward(x)

    def _per_sample_log_det(self, bf, x):
        """
        Compute per-sample log|det J| using finite difference approximation.

        :param bf: a BreezeForest component
        :param x: input tensor (batch_size, dim)
        :return: per-sample log-determinant tensor (batch_size,)
        """
        bf.batch_example = x
        epsilons = bf.epsilon

        x_deltas = torch.cat([
            (x - epsilons).view(1, -1, x.size(1)),
            (x + epsilons).view(1, -1, x.size(1))
        ], dim=0)

        breeze_list = []
        y = bf.forward(x, breeze_list)
        x_deltas = bf.breeze_forward(x_deltas, breeze_list)

        du_dx = (x_deltas[1] - x_deltas[0]) / (2 * epsilons)
        du_dx = torch.abs(du_dx * bf.dim_mask + 1 - bf.dim_mask).clamp(min=0.001)

        # Sum log|du/dx| over dimensions -> per-sample scalar
        return torch.sum(torch.log(du_dx), dim=1)  # (batch_size,)

    def _per_sample_log_det_exact(self, bf, x):
        """
        Compute per-sample exact log|det J| using torch.func.jacrev.

        :param bf: a BreezeForest component
        :param x: input tensor (batch_size, dim)
        :return: per-sample log-determinant tensor (batch_size,)
        """
        bf.batch_example = x

        def single_forward(x_single):
            x_single = x_single.unsqueeze(0)
            breeze_list = []
            y_single = bf.forward(x_single, breeze_list)
            return y_single.squeeze(0)

        try:
            jacobian_fn = vmap(jacrev(single_forward))
            jacobians = jacobian_fn(x)
            sign, log_det = torch.linalg.slogdet(jacobians)
        except Exception as e:
            print(f"Warning: vmap failed ({e}), falling back to loop-based computation")
            log_dets = []
            for i in range(x.shape[0]):
                jac = jacrev(single_forward)(x[i])
                _, ld = torch.linalg.slogdet(jac)
                log_dets.append(ld)
            log_det = torch.stack(log_dets)

        return log_det  # (batch_size,)

    def train_forward(self, x, exact=False):
        """
        Compute mixture log-likelihood with log-sum-exp over components.

        log p(x) = logsumexp_k( log pi_k + log |det J_k(x)| )

        :param x: input tensor (batch_size, dim)
        :param exact: if True, use exact Jacobian via jacrev
        :return: mean log p(x) over batch (scalar, negate for loss)
        """
        log_pi = self.get_mixture_log_weights()  # (K,)

        det_fn = self._per_sample_log_det_exact if exact else self._per_sample_log_det

        component_log_probs = []
        for k, bf in enumerate(self.components):
            per_sample_ld = det_fn(bf, x)  # (batch_size,)
            component_log_probs.append(log_pi[k] + per_sample_ld)

        # (K, batch_size) -> logsumexp over K -> (batch_size,)
        stacked = torch.stack(component_log_probs, dim=0)
        log_prob = torch.logsumexp(stacked, dim=0)

        return torch.mean(log_prob)

    def inverse_map(self, n_samples, max_gap=1e-3, decay_ratio=1.0):
        """
        Generate samples from the mixture distribution.

        1. Sample component k ~ Categorical(pi)
        2. For each k, sample z ~ Uniform(0.01, 0.99)^dim
        3. x = f_k^{-1}(z) via bisection-based inverse_map

        :param n_samples: number of samples to generate
        :param max_gap: bisection precision
        :param decay_ratio: bisection decay ratio
        :return: generated samples (n_samples, dim)
        """
        weights = self.get_mixture_weights().detach()

        component_indices = torch.multinomial(weights, n_samples, replacement=True)

        results = torch.zeros(n_samples, self.dim)

        for k in range(self.n_components):
            mask = (component_indices == k)
            n_k = mask.sum().item()
            if n_k == 0:
                continue

            z = torch.rand(n_k, self.dim) * 0.98 + 0.01
            x_k = self.components[k].inverse_map(
                z, max_gap=max_gap, decay_ratio=decay_ratio
            )
            results[mask] = x_k

        return results

    def explain(self):
        """Print mixture weights and component details."""
        print("Mixture weights:", self.get_mixture_weights().detach())
        for k, bf in enumerate(self.components):
            print(f"\n--- Component {k} ---")
            bf.explain()


if __name__ == "__main__":
    # Quick sanity check
    mbf = MultiBF(
        n_components=3,
        dim=2,
        shapes=[[1, 4, 8, 1]],
        sap_w=0.5,
        inc_mode="no strict"
    )

    # ActiNorm init
    x = torch.randn(100, 2)
    with torch.no_grad():
        mbf.forward(x)

    # Training step
    optimizer = torch.optim.Adam(mbf.parameters(), lr=0.01)

    x = torch.randn(50, 2)
    log_prob = mbf.train_forward(x)
    loss = -log_prob
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"log_prob: {log_prob.item():.4f}")
    print(f"mixture weights: {mbf.get_mixture_weights().detach()}")

    # Sampling
    with torch.no_grad():
        samples = mbf.inverse_map(n_samples=200)
        print(f"samples shape: {samples.shape}")
        print(f"sample range: [{samples.min().item():.2f}, {samples.max().item():.2f}]")
