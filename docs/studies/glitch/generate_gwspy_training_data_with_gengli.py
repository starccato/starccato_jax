"""

1. Create an empty bank.
2. Generate a random glitch and add it as the first template.
3. Set max_fails = 1000
4. Set overlap_threshold = 0.2
5. while attempts < max_fails:
   a. Sample new glitch from GAN
   b. is overlap(new_glitch, bank) > overlap_threshold?
        YES: attempts += 1, repeat step 5
        NO: attempts = 0, add new_glitch to bank, repeat step 5
6. Cant add new glitch with overlap < overlap_threshold, saving bank

"""

import os
import pickle

import gengli
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from tqdm import tqdm


def match(glitch_bank, glitch):
    """
    Compute the match between a bank of glitches and a candidate glitch.

    The function normalizes each glitch (both from the bank and the candidate)
    and computes the cross-correlation via FFT. For each template in the bank,
    the maximum absolute value of the inverse FFT of the product of the FFTs is
    returned as the match value.

    Parameters
    ----------
    glitch_bank : np.ndarray
        Array of shape (N, D) where each row is a glitch template.
    glitch : np.ndarray
        1D array representing a candidate glitch.

    Returns
    -------
    np.ndarray
        Array of shape (N,) with the maximum absolute correlation values for
        each template in the bank.
    """

    def sigma_squared(g):
        return np.sum(np.square(g), axis=-1)

    assert (
        glitch.ndim == 1
    ), "The candidate glitch must be a one-dimensional array"

    # Normalize glitches
    glitch_bank = (glitch_bank.T / np.sqrt(sigma_squared(glitch_bank))).T
    glitch = glitch / np.sqrt(sigma_squared(glitch))

    # FFT cross-correlation
    correlation = np.multiply(
        np.conj(np.fft.fft(glitch_bank)), np.fft.fft(glitch)
    )
    correlation = np.fft.ifft(correlation, axis=-1)
    match_values = np.max(np.abs(correlation), axis=-1)

    return match_values


def save_bank(bank, mm_threshold, checkpoint_dir="banks"):
    """
    Save the bank for a given min_match threshold to a checkpoint file.

    Parameters
    ----------
    bank : np.ndarray
        The glitch bank array.
    mm_threshold : float
        The minimum match threshold corresponding to this bank.
    checkpoint_dir : str, optional
        Directory to save the bank files.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir, f"bank_mm_{mm_threshold:.2f}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(bank, f)


def load_bank(mm_threshold, checkpoint_dir="banks"):
    """
    Load the bank for a given min_match threshold from a checkpoint file, if it exists.

    Parameters
    ----------
    mm_threshold : float
        The minimum match threshold corresponding to the desired bank.
    checkpoint_dir : str, optional
        Directory where the bank files are stored.

    Returns
    -------
    np.ndarray or None
        The loaded bank if the file exists, otherwise None.
    """
    filename = os.path.join(checkpoint_dir, f"bank_mm_{mm_threshold:.2f}.pkl")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            bank = pickle.load(f)
        print(
            f"Loaded existing bank for mm_threshold = {mm_threshold:.2f} with {bank.shape[0]} templates."
        )
        return bank
    else:
        return None


def generate_bank(
    min_match,
    empty_loops=100,
    srate=4096.0,
    seed_bank=None,
    checkpoint_dir="banks",
):
    """
    Generate a template bank of glitch signals using a stochastic algorithm.
    This version supports checkpointing so that progress is saved to disk.

    Parameters
    ----------
    min_match : float
        The minimum match threshold required between a candidate and existing templates.
    empty_loops : int, optional
        Number of consecutive iterations without adding a new template before termination.
    srate : float, optional
        The sampling rate used to generate glitches.
    seed_bank : np.ndarray, optional
        A starting bank of glitches. Its last dimension must match the expected glitch length.
    checkpoint_dir : str, optional
        Directory to save checkpoint files.

    Returns
    -------
    np.ndarray
        Array of glitch templates of shape (N, D), where N is the number of templates.
    """
    g = gengli.glitch_generator("L1")
    # Try to load an existing bank if available.
    bank = load_bank(min_match, checkpoint_dir=checkpoint_dir)
    if bank is None:
        if seed_bank is None:
            bank = g.get_glitch(srate=srate)[None, :]
        elif isinstance(seed_bank, np.ndarray):
            expected_length = g.get_len_glitch(srate)
            assert (
                seed_bank.shape[-1] == expected_length
            ), "Seed bank glitch length does not match expected length"
            bank = seed_bank
        else:
            raise ValueError("Seed bank must be a numpy array or None!")

    empty_count = 0
    desc_str = f"Generating bank: min_match = {min_match} | Bank size: {{}} | Fails until quit"
    pbar = tqdm(
        total=empty_loops,
        desc=desc_str.format(bank.shape[0]),
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    )

    highest_empty_count = 0

    while empty_count < empty_loops:
        try:
            proposal = g.get_glitch(srate=srate)
            m = match(bank, proposal)
            if np.max(m) < min_match:
                bank = np.insert(bank, 0, proposal, axis=0)
                empty_count = 0  # Reset the counter on successful addition.
                pbar.set_description(desc_str.format(bank.shape[0]))
            else:
                empty_count += 1
                if highest_empty_count < empty_count:
                    highest_empty_count = empty_count
                    pbar.update(1)
            # Save checkpoint every 50 iterations (or any interval you prefer).
            if pbar.n % 50 == 0:
                save_bank(bank, min_match, checkpoint_dir=checkpoint_dir)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Saving progress...")
            save_bank(bank, min_match, checkpoint_dir=checkpoint_dir)
            break
    pbar.close()
    # Save one final time when done.
    save_bank(bank, min_match, checkpoint_dir=checkpoint_dir)
    return bank


def bank_scaling(
    min_match_list,
    empty_loops=100,
    checkpoint_dir="banks",
    savefile=None,
    load=False,
):
    """
    Determine and plot how the size of the template bank scales with the minimum match threshold.
    This function also uses checkpointing for each bank.

    Parameters
    ----------
    min_match_list : list or array-like
        List of minimum match thresholds to test.
    empty_loops : int, optional
        Number of consecutive iterations without a new template to terminate bank generation.
    checkpoint_dir : str, optional
        Directory where individual bank checkpoints are saved.
    savefile : str, optional
        File path to save the overall scaling data via pickle.
    load : bool, optional
        If True, load previously saved scaling data from the specified savefile.

    Returns
    -------
    dict
        Dictionary containing the minimum match values ('min_match_list') and corresponding
        bank sizes ('bank_size').
    """
    if load and savefile is not None and os.path.exists(savefile):
        with open(savefile, "rb") as filehandler:
            scaling_dict = pickle.load(filehandler)
    else:
        scaling_dict = {
            "min_match_list": np.array(min_match_list),
            "bank_size": [],
        }
        for mm in min_match_list:
            bank = generate_bank(
                mm, empty_loops, checkpoint_dir=checkpoint_dir
            )
            scaling_dict["bank_size"].append(bank.shape[0])
        scaling_dict["bank_size"] = np.array(scaling_dict["bank_size"])
        if savefile is not None:
            with open(savefile, "wb") as filehandler:
                pickle.dump(scaling_dict, filehandler)

    return scaling_dict

phases = np.exp(1j * 2 * np.pi * np.random.rand(n//2 - 1))
def plot_template_bank(bank, num_samples=5):
    """
    Plot a sample of glitch templates from the template bank for diagnostic purposes.

    Randomly selects a subset of glitches from the bank and plots them in the time domain.

    Parameters
    ----------
    bank : np.ndarray
        Array of glitch templates with shape (N, D).
    num_samples : int, optional
        Number of random templates to plot.

    Returns
    -------
    None
    """
    N = bank.shape[0]
    if num_samples > N:
        num_samples = N
    indices = np.random.choice(N, num_samples, replace=False)
    plt.figure(figsize=(10, 6))
    for idx in indices:
        plt.plot(bank[idx], label=f"Template {idx}")
    plt.xlabel("Time index")
    plt.ylabel("Amplitude")
    plt.title("Sample Glitch Templates from the Bank")
    plt.legend()
    plt.grid(True)
    plt.show()


# if __name__ == '__main__':
#     min_match_list = [0.5, 0.6, 0.7, 0.8, 0.9]
#     scaling_data = bank_scaling(
#         min_match_list, empty_loops=500,
#         checkpoint_dir="banks",
#         savefile='bank_scaling.pkl',
#         load=False
#     )


# LOAD bank scaling data


def load_scaling_data(filename):
    """
    Load the scaling data from a pickle file.

    Parameters
    ----------
    filename : str
        Path to the pickle file containing the scaling data.

    Returns
    -------
    dict
        Dictionary containing the minimum match values and corresponding bank sizes.
    """
    with open(filename, "rb") as filehandler:
        scaling_dict = pickle.load(filehandler)
    return scaling_dict


def plot_scaling_data(scaling_dict):
    x = scaling_dict["min_match_list"]
    y = scaling_dict["bank_size"]

    # sort x and y based on x
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    # Fit power-law relationship
    log_x = np.log(x)
    log_y = np.log(y)
    res = linregress(log_x, log_y)
    y_pred = np.exp(res.intercept) * x**res.slope

    plt.figure(figsize=(4, 3))
    plt.plot(x, y_pred, "r--")
    plt.plot(
        x, y, color="r", marker="o", markersize=5, lw=0
    )  # , label=f'Data (power={res.slope:.2f})')

    plt.xlabel(r"$\text{Minimum Match between Templates}$")
    plt.ylabel(r"$N_{\mathrm{templates}}$")
    plt.yscale("log")
    plt.xscale("log")

    plt.xlim(0.4, 1.0)
    # dont use SF notation for x ticks log scale
    plt.xticks(
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    )

    plt.title("Scaling of Template Bank Size")
    plt.savefig("bank_scaling.png", bbox_inches="tight")


if __name__ == "__main__":
    generate_bank(
        min_match=0.75,
        empty_loops=5000,
        srate=4096.0,
        seed_bank=None,
        checkpoint_dir="banks",
    )
