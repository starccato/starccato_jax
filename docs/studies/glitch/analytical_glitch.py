import numpy as np

from scipy.stats import uniform, loguniform


def generate_sine_gaussian_blip(f0=300, Q=9, amplitude=1e-21,
                                duration=0.1, sample_rate=4096):
    """
    Generate a sine-Gaussian waveform to model blip glitches

    Parameters:
    f0 (float): Central frequency in Hz (typical blip range: 20-1000 Hz)
    Q (float): Quality factor (∼9 for blip glitches)
    amplitude (float): Peak amplitude of the glitch
    duration (float): Total waveform duration in seconds
    sample_rate (int): Sampling frequency in Hz

    Returns:
    tuple: (time_array, waveform)
    """
    t = np.arange(-duration / 2, duration / 2, 1 / sample_rate)
    tau = Q / (np.pi * f0)  # Characteristic decay time
    envelope = amplitude * np.exp(-t ** 2 / (2 * tau ** 2))
    waveform = envelope * np.sin(2 * np.pi * f0 * t)
    return t, waveform


def sample_blip_params(num_samples=1000):
    """
    Sample parameters for blip glitches from specified distributions.

    Parameters:
    num_samples (int): Number of samples to generate.

    Returns:
    dict: Dictionary containing sampled parameters.
    """
    f0_samples = uniform.rvs(loc=20, scale=980, size=num_samples)
    Q_samples = loguniform.rvs(5, 15, size=num_samples)
    amplitude_samples = loguniform.rvs(1e-22, 1e-20, size=num_samples)

    return {
        "f0": f0_samples,
        "Q": Q_samples,
        "amplitude": amplitude_samples
    }



def generate_blip_waveforms(num_samples=1000, duration=0.1, sample_rate=4096):
    """
    Generate multiple blip waveforms based on sampled parameters.

    Parameters:
    num_samples (int): Number of waveforms to generate.
    duration (float): Duration of each waveform in seconds.
    sample_rate (int): Sampling frequency in Hz.

    Returns:
    list: List of tuples containing (time_array, waveform) for each sample.
    """
    params = sample_blip_params(num_samples)
    t = np.arange(-duration / 2, duration / 2, 1 / sample_rate)
    nt = len(t)
    waveforms = np.zeros((num_samples, nt))

    for i in range(num_samples):
        waveforms[i] = generate_sine_gaussian_blip(
            f0=params["f0"][i],
            Q=params["Q"][i],
            amplitude=params["amplitude"][i],
            duration=duration,
            sample_rate=sample_rate
        )[1]
    return waveforms


def plot_credible_intervals(data, ci=0.9):
    """
    Plot the credible intervals for the generated blip waveforms.

    Parameters:
    data (array): The data to plot.
    ci (float): Credible interval to plot.
    """
    import matplotlib.pyplot as plt

    lower_bound = np.percentile(data, (1 - ci) / 2 * 100, axis=0)
    upper_bound = np.percentile(data, (1 + ci) / 2 * 100, axis=0)
    x = np.arange(len(lower_bound))
    # plt.fill_between(x, lower_bound, upper_bound, alpha=0.5)
    for i in range(data.shape[0]):
        plt.plot(data[i], color="gray", alpha=0.1, lw=0.1)

    # plt.plot(data, label="Data")
    plt.title(f"Credible Interval: {ci*100}%")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Example usage
    waveforms = generate_blip_waveforms(num_samples=1000)
    plot_credible_intervals(waveforms, ci=0.9)
