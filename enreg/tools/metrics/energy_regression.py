import os
import numpy as np
import awkward as ak
from omegaconf import DictConfig
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from enreg.tools.visualization import base as b
from enreg.tools import general as g

hep.style.use(hep.styles.CMS)

# TODO: Add later the possibility to check also full tau p4.

def plot_tau_en_ratio_dependence(x_values, y_values, y_err, label, ax):
    # plt.fill_between(
    #     x=x_values,
    #     y1=y_values-y_err,
    #     y2=y_values+y_err,
    #     label=label,
    #     alpha=0.4
    # )
    plt.errorbar(
        x=x_values,
        y=y_values,
        yerr=y_err,
        color='k',
        linestyle='',
        label=label,
        marker='.',
        capsize=5
    )
    plt.grid()
    plt.legend(prop={"size": 30})
    plt.title(r"$\tau_h$ energy reconstruction" , loc="left")
    plt.ylabel(r"$\frac{E^{RECO}}{E^{GEN}}$", fontsize=30)
    plt.xlabel(r"$E_{vis}^{GEN}$", fontsize=30)
    plt.ylim([0, 5])
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)


def plot_energy_regression(algorithm_info, cfg):
    plot_tau_en_ratio(algorithm_info, cfg)
    ratio_distribution(algorithm_info, cfg)


def prepare_violin_plot_data(
    sample_data: ak.Array,
    cfg: DictConfig
):
    reco_gen_E_ratio = g.reinitialize_p4(sample_data.tau_p4s).energy / sample_data.gen_jet_tau_vis_energy
    gen_vis_tau_E = sample_data.gen_jet_tau_vis_energy
    bin_edges = np.array(cfg.metrics.regression.ratio_plot.bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_gen_tau_energies = np.digitize(gen_vis_tau_E, bins=bin_edges)  # Biggest idx is overflow
    ratio_values = [reco_gen_E_ratio[binned_gen_tau_energies == bin_idx].to_numpy() for bin_idx in range(1, len(bin_edges))]
    return ratio_values, bin_centers


def plot_ratio_violin_plot(
    x_values: np.array,
    y_values: np.array,
    label: str,
    cfg: DictConfig
):
    fig, ax = plt.subplots(figsize=(16,9))
    ax.violinplot(y_values, positions=x_values, widths=10)
    plt.grid()
    # plt.legend(prop={"size": 20})
    plt.title(label , fontsize=18, loc="center", fontweight="bold", style="italic", family="monospace")
    plt.ylabel(r"$\left(\frac{p_{T, reco}}{p_{T, gen}}\right)$", fontsize=20)
    plt.xlabel(r"$p_T^{GEN}$ [GeV]", fontsize=20)
    plt.ylim((0, 30))
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)
    output_path = os.path.join(cfg.output_dir, "violin_plot.png")
    plt.savefig(output_path, bbox_inches='tight')


def ratio_distribution(algorithm_info: dict, cfg: DictConfig):
    for algorithm_name, algorithm_values in algorithm_info.items():
        for dataset_name, dataset_values in algorithm_values.items():
            for sample_name, sample_data in dataset_values.items():
                label = f"{algorithm_name}: {sample_name}"
                ratio_values, bin_centers = prepare_violin_plot_data(sample_data=sample_data, cfg=cfg)
                plot_ratio_violin_plot(x_values=bin_centers, y_values=ratio_values, label=label, cfg=cfg)
                resolutions, bin_centers = prepare_resolution_plot_data(sample_data, 'IQR', cfg)
                plot_ATLAS_resolution(x_values=bin_centers, y_values=resolutions, cfg=cfg)


def plot_tau_en_ratio(algorithm_info: dict, cfg: DictConfig):
    fig, ax = plt.subplots(figsize=(16,9))
    for algorithm_name, algorithm_values in algorithm_info.items():
        for dataset_name, dataset_values in algorithm_values.items():
            for sample_name, sample_data in dataset_values.items():
                label = f"{algorithm_name}: {sample_name}"
                ratio_means, ratio_std, bin_centers = prepare_tau_en_ratio_data(sample_data, 'IQR', cfg)
                plot_tau_en_ratio_dependence(x_values=bin_centers, y_values=ratio_means, y_err=ratio_std, label=label, ax=ax)
    output_path = os.path.join(os.path.expandvars(cfg.output_dir), "reco_gen_ratio.pdf")
    plt.savefig(output_path)


def prepare_tau_en_ratio_data(
    sample_data: ak.Array,
    resolution_type: str,
    cfg: DictConfig
):
    reco_gen_E_ratio = g.reinitialize_p4(sample_data.tau_p4s).energy / sample_data.gen_jet_tau_vis_energy
    gen_vis_tau_E = sample_data.gen_jet_tau_vis_energy
    bin_edges = np.array(cfg.metrics.regression.ratio_plot.bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_gen_tau_energies = np.digitize(gen_vis_tau_E, bins=bin_edges)  # Biggest idx is overflow
    ratio_values = [reco_gen_E_ratio[binned_gen_tau_energies == bin_idx].to_numpy() for bin_idx in range(1, len(bin_edges))]
    if resolution_type == 'std':
        ratio_means = np.array([np.mean(ratio) for ratio in ratio_values])
        ratio_std = np.array([np.std(ratio) for ratio in ratio_values])
    elif resolution_type == 'IQR':
        ratio_means = np.array([np.median(ratio) for ratio in ratio_values])
        ratio_std = np.array([(np.quantile(ratio, 0.75) - np.quantile(ratio, 0.25)) for ratio in ratio_values])
    return ratio_means, ratio_std, bin_centers


def prepare_resolution_plot_data(
    sample_data: ak.Array,
    resolution_type: str,
    cfg: DictConfig
):
    """ Prepares the data for the resolution plotting

    Args:
        sample_data : ak.Array
            The input data
        resolution_type : str
            Either 'std' or 'IQR'.
        cfg : DictConfig
            Configuration

    Returns:
        resolutions : np.array
            The resulting resolutions
        gen_tau_pt : np.array
            The bin centers for the resolutions
    """
    gen_vis_tau_pt = g.reinitialize_p4(sample_data.gen_jet_tau_p4s).pt
    reco_gen_pt_ratio = g.reinitialize_p4(sample_data.tau_p4s).pt / g.reinitialize_p4(sample_data.gen_jet_tau_p4s).pt
    bin_edges = np.array(cfg.metrics.regression.ratio_plot.bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_gen_tau_pt = np.digitize(gen_vis_tau_pt, bins=bin_edges)
    ratio_values = [reco_gen_pt_ratio[binned_gen_tau_pt == bin_idx].to_numpy() for bin_idx in range(1, len(bin_edges))]
    if resolution_type == 'std':
        resolutions = np.array([np.std(ratio) / np.mean(ratio) for ratio in ratio_values])
    elif resolution_type == 'IQR':
        resolutions = np.array(
            [(np.quantile(ratio, 0.75) - np.quantile(ratio, 0.25)) / np.median(ratio) for ratio in ratio_values]
        )
    resolutions *= 100
    return resolutions, bin_centers



def plot_ATLAS_resolution(x_values, y_values, cfg):
    fig, ax = plt.subplots(figsize=(16,9))
    ATLAS_baseline = {
        "x": [30, 50, 68, 87, 105, 125, 145, 165, 182, 203, 222, 241, 260],
        "y": [14, 11.1, 9.1, 8.1, 7.4, 7.0, 6.5, 6.3, 6.0, 6.05, 5.8, 5.3, 5.35],
        "marker": ".",
        "color": "k"
    }
    ATLAS_BRT = {
        "x": [30, 50, 68, 87, 105, 125, 145, 165, 182, 203, 222, 241, 260],
        "y": [7.1, 6.8, 6.5, 6.0, 5.9, 5.4, 5.39, 5.4, 5.0, 5.3, 5.2, 5.28, 5.9],
        "marker": "^",
        "color": "r"
    }
    x_range = (20, 270)
    y_range = (0, 20)
    plt.plot(ATLAS_baseline["x"], ATLAS_baseline["y"], marker=ATLAS_baseline["marker"], color=ATLAS_baseline["color"], label="[ATLAS] Baseline")
    plt.plot(ATLAS_BRT["x"], ATLAS_BRT["y"], marker=ATLAS_BRT["marker"], color=ATLAS_BRT["color"], label="[ATLAS] BRT")
    plt.plot(x_values, y_values, marker="*", color='green', label="HPS")
    plt.grid()
    plt.legend(prop={"size": 20})
    plt.title("response" , fontsize=18, loc="center", fontweight="bold", style="italic", family="monospace")
    plt.ylabel(r"$p_T$ resolution [%]", fontsize=20)
    plt.xlabel(r"$p_T^{GEN}$ [GeV]", fontsize=20)
    plt.ylim((0, 30))
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)
    output_path = os.path.join(cfg.output_dir, "atlas_plot.png")
    plt.savefig(output_path, bbox_inches='tight')
