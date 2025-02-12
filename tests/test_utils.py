import matplotlib.pyplot as plt

from starccato_jax.sampler.utils import beta_spaced_samples


def test_beta_samples(outdir):
    samples = beta_spaced_samples(100, 0.3, 1)
    plt.plot(samples)
    plt.savefig(f"{outdir}/test_beta_samples.png")
