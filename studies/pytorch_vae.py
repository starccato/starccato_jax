"""This loads the signal data from the raw simulation outputs from Richers et al (20XX) ."""
import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# from ..defaults import BATCH_SIZE
from ..logger import logger

BATCH_SIZE = 32

# _ROOT_URL = "https://raw.githubusercontent.com/starccato/data/main/training"
SIGNALS_CSV = "https://github.com/starccato/data/raw/refs/heads/main/training/richers_1764.csv"
PARAMETERS_CSV = "https://github.com/starccato/data/blob/main/training/richers_1764_parameters.csv"
TIME_CSV = f"../data/training/richers_1764_times.csv"


class Data(Dataset):
    def __init__(self, batch_size=BATCH_SIZE, frac=1, train=True, indices=None):
        ### read data from csv files
        self.parameters = pd.read_csv(PARAMETERS_CSV)
        self.signals = pd.read_csv(SIGNALS_CSV).astype("float32").T
        self.signals.index = [i for i in range(len(self.signals.index))]

        assert (
            self.signals.shape[0] == self.parameters.shape[0],
            "Signals and parameters must have the same number of rows (the number of signals)",
        )

        if frac < 1:
            init_shape = self.signals.shape
            n_signals = int(frac * self.signals.shape[0])
            # keep n_signals random signals columns
            self.signals = self.signals.sample(n=n_signals, axis=0)
            self.parameters = self.parameters.iloc[self.signals.index, :]
            logger.info(
                f"Frac of TrainingData being used {init_shape} -> {self.signals.shape}"
            )

        # remove unusual parameters
        keep_idx = self.parameters["beta1_IC_b"] > 0
        self.parameters = self.parameters[keep_idx]
        self.parameters = self.parameters["beta1_IC_b"]
        self.signals = self.signals[keep_idx]
        self.signals = self.signals.values.T

        ### flatten signals and take last 256 timestamps
        temp_data = np.empty(shape=(256, 0)).astype("float32")

        for i in range(0, self.signals.shape[1]):
            signal = self.signals[:, i]
            signal = signal.reshape(1, -1)

            cut_signal = signal[:, int(len(signal[0]) - 256): len(signal[0])]
            temp_data = np.insert(
                temp_data, temp_data.shape[1], cut_signal, axis=1
            )

        self.signals = temp_data

        if indices is not None:
            if train:
                self.signals = self.signals[:, indices]
                self.parameters = self.parameters.iloc[indices]
                self.indices = indices
            else:
                self.signals = self.signals[:, indices]
                self.parameters = self.parameters.iloc[indices]
                self.indices = indices

        self.batch_size = batch_size
        self.mean = self.signals.mean()
        self.std = np.std(self.signals, axis=None)
        self.scaling_factor = 5
        self.max_value = abs(self.signals).max()
        self.ylim_signal = (self.signals[:, :].min(), self.signals[:, :].max())

    def __str__(self):
        return f"TrainingData: {self.signals.shape}"

    def __repr__(self):
        return self.__str__()

    @property
    def raw_signals(self):
        return pd.read_csv(SIGNALS_CSV).astype("float32").T.values

    def summary(self):
        """Display summary stats about the data"""
        str = f"Signal Dataset mean: {self.mean:.3f} +/- {self.std:.3f}\n"
        str += f"Signal Dataset scaling factor (to match noise in generator): {self.scaling_factor}\n"
        str += f"Signal Dataset max value: {self.max_value}\n"
        str += f"Signal Dataset shape: {self.signals.shape}\n"
        logger.info(str)

    def standardize(self, signal):
        standardized_signal = (signal - self.mean) / self.std
        standardized_signal = standardized_signal / self.scaling_factor
        return standardized_signal

    def normalise(self, signal):
        normalised_signal = signal / self.max_value
        return normalised_signal

    ### overloads ###
    def __len__(self):
        return self.signals.shape[1]

    @property
    def shape(self):
        return self.signals.shape

    def get_indices(self):
        return self.indices

    def __getitem__(self, idx):
        signal = self.signals[:, idx]
        signal = signal.reshape(1, -1)
        parameters = self.parameters.iloc[idx]

        normalised_signal = self.normalise(signal)

        return normalised_signal, parameters

    def get_loader(self) -> DataLoader:
        return DataLoader(
            self, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def get_signals_iterator(self):
        return next(iter(self.get_loader()))

    def plot_waveforms(
            self, fname=None, normalised=False
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes = axes.flatten()
        signal_iterator = self.get_signals_iterator()

        # Plot each signal on a separate subplot
        for i, ax in enumerate(axes):
            x = np.arange(signal_iterator.size(dim=2))
            y = signal_iterator[i, :, :].flatten()

            if normalised:
                # y = y * self.scaling_factor
                # y = y * self.std + self.mean
                y = y * self.max_value

            ax.plot(x, y, color="blue")

            ax.axvline(x=53, color="black", linestyle="--", alpha=0.5)
            ax.grid(True)
            ax.set_ylim((-4, 2))
            if normalised:
                ax.set_ylim(self.ylim_signal)

            # Add axis titles
            ax.set_ylabel("distance * strain (cm)")
            ax.set_xlabel("n (timestamps)")
            ax.set_xlim(min(x), max(x))

            # parameters = signal_iterator[i, :].numpy()[0]
            # parameters_with_names = f'{parameter_names[0]}: {parameters[0]:.6f}\n{parameter_names[1]}: {parameters[1]:.2f}, {parameter_names[2]}: {parameters[2]:.2f}'
            # ax.set_xlabel(f'Parameters:\n{parameters_with_names}')

        fig.suptitle("Waveforms")
        if normalised:
            fig.suptitle("Normalised Waveforms")

        for i in range(407, 8 * 4):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        return fig, axes


