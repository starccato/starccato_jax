# pip install gengli

import numpy as np
import matplotlib.pyplot as plt
import gengli
from tqdm import tqdm
import pickle
import os
from scipy.stats import linregress


######################################################

def match(glitch1, glitch2):
    "Match between whithened glitches (in TD)"
    sigmasq = lambda g: np.sum(np.square(g), axis=-1)
    assert glitch2.ndim == 1, "glitch2 must be one-dimensional"

    # normalizing glitches
    glitch1 = (glitch1.T / np.sqrt(sigmasq(glitch1))).T  # (N,)/(N,D)
    glitch2 = (glitch2.T / np.sqrt(sigmasq(glitch2))).T  # (N,)

    match = np.multiply(np.conj(np.fft.fft(glitch1)), np.fft.fft(glitch2))  # (N,D)
    match = np.fft.ifft(match, axis=-1)  # (N,D)
    match = np.max(np.abs(match), axis=-1)
    # match = np.sum(np.multiply(glitch1, glitch2), axis =-1).real/glitch2.shape[0] #(N,) #overlap!!

    return match


def generate_bank(MM, empty_loops=100, srate=4096., seed_bank=None):
    """
    Generates a bank of glitches with a given minimum match MM, via a stochastic algorithm.
    The iteration is terminated after empty_loops loops.
    If seed_bank is given,
    """
    nothing_new = 0
    g = gengli.glitch_generator('L1')

    if seed_bank is None:
        bank = g.get_glitch(srate=srate)[None, :]
    elif isinstance(seed_bank, np.ndarray):
        assert seed_bank.shape[-1] == g.get_len_glitch(srate)
        bank = seed_bank
    else:
        raise ValueError("Seed bank must be a numpy array!")

    # Initializing the loop!
    def generator():
        while True:
            yield

    desc_str = 'Generating glitch bank: MM = {} - Bank size: {}'
    pbar = tqdm(generator(), desc=desc_str.format(MM, bank.shape[0]))

    # Loop to add templates
    for i in pbar:
        try:
            proposal = g.get_glitch(srate=srate)
            m = match(bank, proposal)
            if np.max(m) < MM:
                bank = np.insert(bank, 0, proposal, axis=0)
                nothing_new = 0
                pbar.set_description(desc_str.format(MM, bank.shape[0]))
            else:
                nothing_new += 1
            if nothing_new >= empty_loops: break
        except KeyboardInterrupt:
            print("Exiting the bank generation loop")
            break

    return bank


def bank_scaling(MM_list, empty_loops=100, savefile=None, load=False):
    if load:
        with open(savefile, 'rb') as filehandler:
            scaling_dict = pickle.load(filehandler)
    else:
        scaling_dict = dict(MM_list=np.array(MM_list), bank_size=[])

        for MM in MM_list:
            scaling_dict['bank_size'].append(generate_bank(MM, empty_loops).shape[0])

        scaling_dict['bank_size'] = np.array(scaling_dict['bank_size'])

        if isinstance(savefile, str):
            with open(savefile, 'wb') as filehandler:
                pickle.dump(scaling_dict, filehandler)

    # plotting part
    plt.figure()
    x, y = 1 - scaling_dict['MM_list'], scaling_dict['bank_size']
    p = plt.scatter(x, y)

    res = linregress(np.log(x), np.log(y))
    m, q = res.slope, res.intercept
    y_ = lambda x_: x_ * m + q
    y_pred = np.exp(y_(np.log(x)))
    plt.plot(x, y_pred, '--')

    plt.xlabel(r"$1-MM$")
    plt.ylabel(r"$N_{templates}$")
    plt.xscale('log')
    plt.yscale('log')

    plt.show()


######################################################
if __name__ == '__main__':
    # generate a single bank
    bank = generate_bank(0.8, 1000)
    print("The glitch bank has {} templates".format(bank.shape[0]))

    # compute the scaling relation of the number of templates as a function of the minimum match MM
    MM_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    bank_scaling(MM_list, empty_loops=500, savefile='out.pkl', load=False)











