from starccato_sampler.pp_test import pp_test

if __name__ == "__main__":
    pp_test(
        result_regex="out_mcmc/inj_*/inference.nc",
        credible_levels_fpath="credible_levels.npy",
        plot_fname="pp_plot.png",
        include_title=True,
    )
