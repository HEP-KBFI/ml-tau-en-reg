import os
import json
import mplhep as hep
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import matplotlib.colors as colors
from enreg.tools import general as g
from enreg.tools.visualization import base as b

hep.style.use(hep.styles.CMS)


def plot_E_gen_distribution(algorithm_info, cfg):
    output_path = os.path.join(cfg.output_dir, 'gen_pt_distribution.png')
    algo_name = list(algorithm_info.keys())[0]
    dataset = 'test' if 'test' in algorithm_info[algo_name] else algorithm_info[algo_name][0]
    gen_pts = []
    sample_names = []
    for sample_name, sample_data in algorithm_info[algo_name][dataset].items():
        gen_pts.append(g.reinitialize_p4(sample_data.gen_jet_tau_p4s).pt)
        sample_names.append(sample_name)
    plot_multiple_histograms(
        entries=gen_pts,
        labels=sample_names,
        output_path=output_path,
        bin_edges=np.array(cfg.metrics.regression.ratio_plot.bin_edges),
        x_label=r"$p_T^{gen}$",
        y_label=r"Entries",
        hatches=['//', '\\\\'],
        colors=['b', 'r']
    )

def plot_multiple_histograms(
    entries: list[np.array],
    labels: list[str],
    output_path: str,
    bin_edges : list[float],
    n_bins: int = 24,
    figsize: tuple = (12, 12),
    y_label: str = "",
    x_label: str = "",
    title: str = "",
    integer_bins : bool = False,
    hatches : list[str] = ["//"],
    colors : list[str] = ["blue"],
):
    fig, ax = plt.subplots(figsize=figsize)
    for entry, label, hatch, color in zip(entries, labels, hatches, colors):
        hist, bin_edges = np.histogram(entry, bins=bin_edges)
        hep.histplot(hist, bin_edges, label=label, hatch=hatch, color=color)
    plt.xlabel(x_label, fontdict={"size": 20})
    plt.ylabel(y_label, fontdict={"size": 20})
    plt.grid(True, which="both")
    plt.title(title, loc="left")
    plt.legend()
    plt.savefig(output_path)#, format="pdf")
    plt.close("all")


def plot_energy_regression(algorithm_info, cfg):
    plotting_input = get_plotting_input(algorithm_info, cfg)
    for algorithm, properties in cfg.metrics.regression.algorithms.items():
        if properties.load_from_json:
            with open(properties.json_metrics_path, 'rt') as in_file:
                plotting_input.update(json.load(in_file)[algorithm])
    json_output_path = output_path = os.path.join(cfg.output_dir, 'plotting_data.json')
    g.save_to_json(plotting_input, json_output_path)
    plot_E_gen_distribution(algorithm_info, cfg)
    plot_mean(plotting_input, cfg, resolution_type='IQR', variable='pt')
    plot_mean(plotting_input, cfg, resolution_type='std', variable='pt')
    plot_resolution(plotting_input, cfg, resolution_type='std', variable='pt')
    plot_resolution(plotting_input, cfg, resolution_type='IQR', variable='pt')
    plot_distribution_bin_wise(plotting_input, cfg, variable='pt')
    plot_ATLAS_resolution(
        plotting_input['HPS']['test']['ZH_Htautau']["pt_bin_centers"],
        plotting_input['HPS']['test']['ZH_Htautau']["pt_resolution_w_std"],
        plotting_input['HPS']['test']['ZH_Htautau']["pt_resolution_w_IQR"],
        plotting_input['ParticleTransformer']['test']['ZH_Htautau']["pt_bin_centers"],
        plotting_input['ParticleTransformer']['test']['ZH_Htautau']["pt_resolution_w_std"],
        plotting_input['ParticleTransformer']['test']['ZH_Htautau']["pt_resolution_w_IQR"],
        "ZH_Htautau",
        cfg
    )
    plot_ATLAS_resolution(
        plotting_input['HPS']['test']['Z_Ztautau']["pt_bin_centers"],
        plotting_input['HPS']['test']['Z_Ztautau']["pt_resolution_w_std"],
        plotting_input['HPS']['test']['Z_Ztautau']["pt_resolution_w_IQR"],
        plotting_input['ParticleTransformer']['test']['Z_Ztautau']["pt_bin_centers"],
        plotting_input['ParticleTransformer']['test']['Z_Ztautau']["pt_resolution_w_std"],
        plotting_input['ParticleTransformer']['test']['Z_Ztautau']["pt_resolution_w_IQR"],
        "Z_Ztautau",
        cfg
    )

def plot_distribution_bin_wise(plotting_input, cfg, variable, figsize=(12,12)):
    x_label = r"$\frac{p_T^{reco}}{p_T^{gen}}$" if variable == 'pt' else r"$\frac{E_{vis}^{reco}}{E_{vis}^{gen}}$"
    y_label = "Entries"
    for sample in cfg.comparison_samples:
            for dataset in cfg.comparison_datasets:
                for algorithm in plotting_input.keys():
                    output_dir = os.path.join(cfg.output_dir, sample, dataset, algorithm)
                    os.makedirs(output_dir, exist_ok=True)
                    bin_contents = plotting_input[algorithm][dataset][sample][f"{variable}_ratio_values"]
                    bin_centers = plotting_input[algorithm][dataset][sample][f"{variable}_bin_centers"]
                    for bin_content, bin_center in zip(bin_contents, bin_centers):
                        output_path = os.path.join(output_dir, f"{variable}_bin_{bin_center}.png")
                        fig, ax = plt.subplots(figsize=figsize)
                        hist, bin_edges = np.histogram(bin_content, bins=24)
                        hep.histplot(hist, bin_edges)
                        plt.xlabel(x_label, fontdict={"size": 20})
                        plt.ylabel(y_label, fontdict={"size": 20})
                        plt.grid(True, which="both")
                        plt.title(f"Bin @ {bin_center}", loc="left")
                        # plt.legend()
                        plt.savefig(output_path, bbox_inches='tight')
                        plt.close("all")


def plot_resolution(
    plotting_input,
    cfg,
    resolution_type='IQR',
    variable='pt',
    figsize=(16,9)
):
    x_label = r"$p_T^{gen}$" if variable == 'pt' else r"$E_{vis}^{gen}$"
    y_label = r"$\frac{IQR(reco/gen)}{q_{50}(reco/gen)}$" if resolution_type =='IQR' else r"$\frac{\sigma(reco/gen)}{\mu(reco/gen)}$"
    for sample in cfg.comparison_samples:
            fig, ax = plt.subplots(figsize=figsize)
            for dataset in cfg.comparison_datasets:
                for algorithm in plotting_input.keys():
                    plotting_data = plotting_input[algorithm][dataset][sample]
                    x_values = plotting_data[f"{variable}_bin_centers"]
                    y_values = plotting_data[f"{variable}_resolution_w_{resolution_type}"]
                    plt.plot(
                        x_values,
                        y_values,
                        marker=cfg.metrics.regression.algorithms[algorithm].marker,
                        color=cfg.metrics.regression.algorithms[algorithm].color,
                        label=f"{dataset}: {algorithm}")
            plt.xlabel(x_label, fontdict={"size": 20})
            plt.ylabel(y_label, fontdict={"size": 20})
            plt.grid(True, which="both")
            plt.title(f"{sample} resolution", loc="left")
            plt.legend()
            output_path = os.path.join(cfg.output_dir, f"{sample}_{resolution_type}_resolution_{variable}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close("all")


def plot_mean(
    plotting_input,
    cfg,
    resolution_type='IQR',
    variable='pt',
    figsize=(16,9)
):
    mean_type = 'median' if resolution_type =='IQR' else 'mean'
    x_label = r"$p_T^{gen}$" if variable == 'pt' else r"$E_{vis}^{gen}$"
    y_label = r"$q_{50}(\frac{reco}{gen})$" if resolution_type =='IQR' else r"$\mu(\frac{reco}{gen})$"
    for sample in cfg.comparison_samples:
            fig, ax = plt.subplots(figsize=figsize)
            for dataset in cfg.comparison_datasets:
                for algorithm in plotting_input.keys():
                    plotting_data = plotting_input[algorithm][dataset][sample]
                    x_values = plotting_data[f"{variable}_bin_centers"]
                    y_values = plotting_data[f"{variable}_ratio_{mean_type}s"]
                    plt.plot(
                        x_values,
                        y_values,
                        marker=cfg.metrics.regression.algorithms[algorithm].marker,
                        color=cfg.metrics.regression.algorithms[algorithm].color,
                        label=f"{dataset}: {algorithm}")
            plt.xlabel(x_label, fontdict={"size": 20})
            plt.ylabel(y_label, fontdict={"size": 20})
            plt.grid(True, which="both")
            plt.title(f"{sample} {mean_type}", loc="left")
            plt.legend()
            output_path = os.path.join(cfg.output_dir, f"{sample}_{mean_type}_ratio_{variable}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close("all")

# def prepare_violin_plot_data(
#     sample_data: ak.Array,
#     cfg: DictConfig
# ):
#     reco_gen_E_ratio = g.reinitialize_p4(sample_data.tau_p4s).energy / sample_data.gen_jet_tau_vis_energy
#     gen_vis_tau_E = sample_data.gen_jet_tau_vis_energy
#     bin_edges = np.array(cfg.metrics.regression.ratio_plot.bin_edges)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     binned_gen_tau_energies = np.digitize(gen_vis_tau_E, bins=bin_edges)  # Biggest idx is overflow
#     ratio_values = [reco_gen_E_ratio[binned_gen_tau_energies == bin_idx].to_numpy() for bin_idx in range(1, len(bin_edges))]
#     return ratio_values, bin_centers


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


# def ratio_distribution(algorithm_info: dict, cfg: DictConfig):
#     for algorithm_name, algorithm_values in algorithm_info.items():
#         for dataset_name, dataset_values in algorithm_values.items():
#             for sample_name, sample_data in dataset_values.items():
#                 label = f"{algorithm_name}: {sample_name}"
#                 ratio_values, bin_centers = prepare_violin_plot_data(sample_data=sample_data, cfg=cfg)
#                 plot_ratio_violin_plot(x_values=bin_centers, y_values=ratio_values, label=label, cfg=cfg)
#                 resolutions, bin_centers = prepare_resolution_plot_data(sample_data, 'IQR', cfg)
#                 plot_ATLAS_resolution(x_values=bin_centers, y_values=resolutions, cfg=cfg)


def get_plotting_input(algorithm_info: dict, cfg: DictConfig):
    algorithms = {}
    for algorithm_name, algorithm_values in algorithm_info.items():
        datasets = {}
        for dataset_name, dataset_values in algorithm_values.items():
            samples = {}
            for sample_name, sample_data in dataset_values.items():
                gen_tau_p4s = g.reinitialize_p4(sample_data.gen_jet_tau_p4s)
                gen_jet_p4s = g.reinitialize_p4(sample_data.gen_jet_p4s)
                pred_tau_p4s = g.reinitialize_p4(sample_data.tau_p4s)
                gen_pt_mask = gen_jet_p4s.pt > 15
                # ratio_mask = np.abs(gen_tau_p4s.pt/gen_jet_p4s.pt - 1) < 0.2
                # prediction_mask = pred_tau_p4s.pt > 1
                sample_data = sample_data[gen_pt_mask]# * ratio_mask * prediction_mask]
                label = f"{algorithm_name}: {sample_name}"
                pt_ratio_means, pt_ratio_std, pt_bin_centers, pt_ratio_values = prepare_tau_pt_ratio_data(
                    sample_data=sample_data, resolution_type='std', cfg=cfg)
                pt_ratio_medians, pt_ratio_IQR, pt_bin_centers, pt_ratio_values = prepare_tau_pt_ratio_data(
                    sample_data=sample_data, resolution_type='IQR', cfg=cfg)
                samples[sample_name] = {
                    "pt_ratio_means": pt_ratio_means,
                    "pt_ratio_values": pt_ratio_values,
                    "pt_ratio_std": pt_ratio_std,
                    "pt_resolution_w_std": pt_ratio_std/pt_ratio_means,
                    "pt_ratio_medians": pt_ratio_medians,
                    "pt_ratio_IQR": pt_ratio_IQR,
                    "pt_resolution_w_IQR": pt_ratio_IQR/pt_ratio_medians,
                    "pt_bin_centers": pt_bin_centers,
                }
            datasets[dataset_name] = samples
        algorithms[algorithm_name] = datasets
    return algorithms


def prepare_tau_en_ratio_data(
    sample_data: ak.Array,
    resolution_type: str,
    cfg: DictConfig
):
    if 'tau_vis_energy' not in sample_data.fields:
        reco_gen_E_ratio = g.reinitialize_p4(sample_data.tau_p4s).energy / sample_data.gen_jet_tau_vis_energy
    else:
        reco_gen_E_ratio = (sample_data.tau_vis_energy + g.reinitialize_p4(sample_data.reco_jet_p4s).energy) / sample_data.gen_jet_tau_vis_energy
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
    return ratio_means, ratio_std, bin_centers, ratio_values


def prepare_tau_pt_ratio_data(
    sample_data: ak.Array,
    resolution_type: str,
    cfg: DictConfig
):
    if 'tau_pt' not in sample_data.fields:
        reco_gen_pt_ratio = g.reinitialize_p4(sample_data.tau_p4s).pt / g.reinitialize_p4(sample_data.gen_jet_tau_p4s).pt
    else:
        reco_gen_pt_ratio = sample_data.tau_pt / g.reinitialize_p4(sample_data.gen_jet_tau_p4s).pt
    gen_vis_tau_pt = g.reinitialize_p4(sample_data.gen_jet_tau_p4s).pt
    bin_edges = np.array(cfg.metrics.regression.ratio_plot.bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_gen_tau_pt = np.digitize(gen_vis_tau_pt, bins=bin_edges)  # Biggest idx is overflow
    ratio_values = [reco_gen_pt_ratio[binned_gen_tau_pt == bin_idx].to_numpy() for bin_idx in range(1, len(bin_edges))]
    if resolution_type == 'std':
        ratio_means = np.array([np.mean(ratio) for ratio in ratio_values])
        ratio_std = np.array([np.std(ratio) for ratio in ratio_values])
    elif resolution_type == 'IQR':
        ratio_means = np.array([np.median(ratio) for ratio in ratio_values])
        ratio_std = np.array([(np.quantile(ratio, 0.75) - np.quantile(ratio, 0.25)) for ratio in ratio_values])
    return ratio_means, ratio_std, bin_centers, ratio_values


# def prepare_resolution_plot_data(
#     sample_data: ak.Array,
#     resolution_type: str,
#     cfg: DictConfig
# ):
#     """ Prepares the data for the resolution plotting

#     Args:
#         sample_data : ak.Array
#             The input data
#         resolution_type : str
#             Either 'std' or 'IQR'.
#         cfg : DictConfig
#             Configuration

#     Returns:
#         resolutions : np.array
#             The resulting resolutions
#         gen_tau_pt : np.array
#             The bin centers for the resolutions
#     """
#     gen_vis_tau_pt = g.reinitialize_p4(sample_data.gen_jet_tau_p4s).pt
#     reco_gen_pt_ratio = g.reinitialize_p4(sample_data.tau_p4s).pt / g.reinitialize_p4(sample_data.gen_jet_tau_p4s).pt
#     bin_edges = np.array(cfg.metrics.regression.ratio_plot.bin_edges)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     binned_gen_tau_pt = np.digitize(gen_vis_tau_pt, bins=bin_edges)
#     ratio_values = [reco_gen_pt_ratio[binned_gen_tau_pt == bin_idx].to_numpy() for bin_idx in range(1, len(bin_edges))]
#     if resolution_type == 'std':
#         resolutions = np.array([np.std(ratio) / np.mean(ratio) for ratio in ratio_values])
#     elif resolution_type == 'IQR':
#         resolutions = np.array(
#             [(np.quantile(ratio, 0.75) - np.quantile(ratio, 0.25)) / np.median(ratio) for ratio in ratio_values]
#         )
#     resolutions *= 100
#     return resolutions, bin_centers



def plot_ATLAS_resolution(HPS_x_values, HPS_y_values_std, HPS_y_values_iqr, PT_x_values, PT_y_values_std, PT_y_values_iqr, sample, cfg):
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
    HPS_x_values_mask = HPS_x_values >= 20
    PT_x_values_mask = PT_x_values >= 20
    plt.plot(HPS_x_values[HPS_x_values_mask], HPS_y_values_std[HPS_x_values_mask] * 100, marker="*", color='green', ls='--', label="HPS w/ STD")
    plt.plot(HPS_x_values[HPS_x_values_mask], HPS_y_values_iqr[HPS_x_values_mask] * 100, marker="*", color='green', ls='-', label="HPS w/ IQR")
    plt.plot(PT_x_values[PT_x_values_mask], PT_y_values_std[PT_x_values_mask] * 100, marker="*", color='blue', ls='--', label="PT w/ STD")
    plt.plot(PT_x_values[PT_x_values_mask], PT_y_values_iqr[PT_x_values_mask] * 100, marker="*", color='blue', ls='-', label="PT w/ IQR")
    plt.grid()
    plt.legend(prop={"size": 20})
    plt.title(sample , fontsize=18, loc="center", fontweight="bold", style="italic", family="monospace")
    plt.ylabel(r"$p_T$ resolution [%]", fontsize=20)
    plt.xlabel(r"$p_T^{GEN}$ [GeV]", fontsize=20)
    plt.ylim((0, 30))
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)
    output_path = os.path.join(cfg.output_dir, f"{sample}_atlas_plot.png")
    plt.savefig(output_path, bbox_inches='tight')
