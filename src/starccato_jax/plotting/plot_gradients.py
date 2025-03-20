import os

import matplotlib.pyplot as plt
import numpy as np


def plot_gradients(grad_dict, fname: str = None):
    fig, axes = plt.subplots(
        3, 1, figsize=(5, 6), sharex=True, gridspec_kw={"hspace": 0}
    )  # No vertical space
    grad_dict_new = {}
    # replace any fc*_logvar with fc*[logvar]
    for key in grad_dict.keys():
        if "_logvar" in key:
            grad_dict_new[key.replace("_logvar", "[logvar]")] = grad_dict[key]
        elif "_mean" in key:
            grad_dict_new[key.replace("_mean", "[mean]")] = grad_dict[key]
        else:
            grad_dict_new[key] = grad_dict[key]

    grad_dict = grad_dict_new
    keys = list(grad_dict.keys())

    # Extract unique layer names
    layer_names = []
    for key in keys:
        layer_names.append(get_layer(key))
    layer_names = list(set(layer_names))

    # Define color and linestyle maps
    layer_color_map = {layer: f"C{i}" for i, layer in enumerate(layer_names)}
    linestyle_map = {"kernel": "-", "bias": "--"}

    categories = ["encoder", "decoder", "global"]

    handles, labels = [], []

    for i, category in enumerate(categories):
        ax = axes[i]

        for key, values in grad_dict.items():
            if category in key:
                # Identify layer (fc1, fc2, etc.)
                layer = next(
                    (
                        fc
                        for fc in reversed(layer_names)
                        if fc == get_layer(key)
                    ),
                    "fc1",
                )
                # Identify kernel/bias/logvar type
                line_type = next(
                    (lt for lt in ["kernel", "bias"] if lt in key), "kernel"
                )

                color = layer_color_map[layer]
                linestyle = linestyle_map.get(line_type, "-")

                (line,) = ax.plot(
                    values,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.85,
                    lw=2.5,
                    label=f"{layer.upper()} {line_type.capitalize()}",
                )

                # Collect handles and labels for shared legend
                if f"{layer.upper()} {line_type.capitalize()}" not in labels:
                    handles.append(line)
                    labels.append(f"{layer.upper()} {line_type.capitalize()}")

        # Move title inside the plot
        ax.text(
            0.02,
            0.85,
            category.capitalize(),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )

        # Set log scale for better visualization
        ax.set_yscale("log")

    # Set common labels
    axes[-1].set_xlabel("Epochs")
    axes[1].set_ylabel("Gradient Norm")

    # Set x-axis limits
    max_len = max(len(v) for v in grad_dict.values())
    for ax in axes:
        ax.set_xlim(0, max_len - 1)

    # Place a single legend outside the figure
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=False,
    )

    # Adjust layout to fit legend
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    plt.show()


def get_layer(string):
    layer = ""
    if "coder" in string:
        layer = string.split("coder_")[1]
        layer = "_".join(layer.split("_")[:-1])
    return layer
