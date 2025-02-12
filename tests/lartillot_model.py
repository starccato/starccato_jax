import jax.numpy as jnp
import jax.random as random
from jax import jit
from jax.random import PRNGKey
from jax.scipy.stats import norm


class LartillotModel:
    def __init__(self, p: float, v: float):
        self.p = p
        self.v = v

    @property
    def lnZ(self):
        return jnp.log((self.v / (1 + self.v)) ** (self.p / 2))

    def log_likelihood(self, theta: jnp.ndarray) -> float:
        v, p = self.v, self.p
        return lnl(p, v, theta)

    def log_prior(self, theta: jnp.ndarray) -> float:
        return jnp.sum(norm.logpdf(theta, loc=0, scale=1))

    def simulate_posterior_samples(
        self, n, rng: PRNGKey, beta=1.0
    ) -> jnp.ndarray:
        """
        Use beta=1.0 for the posterior samples.
        Beta!=1.0 for the power posterior (or transitional distributions)
        """
        p, v = self.p, self.v
        mean = jnp.zeros(p)
        cov_matrix = v / (v + beta) * jnp.eye(p)
        return random.multivariate_normal(
            rng, mean=mean, cov=cov_matrix, shape=(n,)
        )

    def generate_lnl_chains(self, n: int, betas, rng: PRNGKey):
        lnl_chains = []
        for beta in betas:
            theta = self.simulate_posterior_samples(n, rng, beta)
            # assert theta.shape == (n, self.p)
            lnl = self.log_likelihood(theta)
            # assert lnl.shape == (n,), lnl.shape
            lnl_chains.append(lnl)

        return jnp.array(lnl_chains)


@jit
def lnl(p: int, v: float, theta: jnp.ndarray):
    return -jnp.sum(theta**2 / (2 * v), axis=1) - (p / 2) * jnp.log(
        2 * jnp.pi * v
    )
