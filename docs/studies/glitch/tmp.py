"""
Script to generate and plot several (white) blip glitches
"""
import sys

# sys.path.insert(0, '..')
import gengli
import matplotlib.pyplot as plt
import numpy as np

############################################

#########
# Glitch generation
#########

N_glitches = 3
srate = 4096.0

# initializing the generator
g = gengli.glitch_generator("L1")

# (whithened) glitch @ 4096Hz
glitches = g.get_glitch(N_glitches, srate=srate, fhigh=250)

t_grid = np.linspace(0, glitches.shape[-1] / srate, glitches.shape[-1])


# Generating some anomalous glitches
g.initialize_benchmark_set(4)
anomalous_glitches = g.get_glitch_confidence_interval(
    [80, 100], N_glitches, srate=srate
)


#########
# Plotting part
#########

plt.figure()
plt.plot(t_grid, glitches.T)

plt.title("Raw glitches", fontsize=20)
plt.xlabel(r"$t (s)$", fontsize=18)
plt.ylabel(r"$g_W$", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.tight_layout()

plt.figure()
plt.plot(t_grid, anomalous_glitches.T)

plt.title("Anomalous glitches", fontsize=20)
plt.xlabel(r"$t (s)$", fontsize=18)
plt.ylabel(r"$g_W$", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.tight_layout()

plt.show()
