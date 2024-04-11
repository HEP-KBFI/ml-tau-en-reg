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

def plot_median_and_iqr(plotting_input, cfg):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharey='row', sharex='col', figsize=(16,9))
    ax1.plot([30, 180], [1, 1], ls='--', c='k')
    ax2.plot([30, 180], [1, 1], ls='--', c='k')
    for algorithm in plotting_input.keys():
        ax1.plot(
            plotting_input[algorithm]['test']['ZH_Htautau']['pt_bin_centers'],
            plotting_input[algorithm]['test']['ZH_Htautau']['pt_ratio_medians'],
            label=algorithm
        )
        ax2.plot(
            plotting_input[algorithm]['test']['Z_Ztautau']['pt_bin_centers'],
            plotting_input[algorithm]['test']['Z_Ztautau']['pt_ratio_medians'],
            label=algorithm
        )
        ax3.plot(
            plotting_input[algorithm]['test']['ZH_Htautau']['pt_bin_centers'],
            plotting_input[algorithm]['test']['ZH_Htautau']['pt_ratio_IQR'],
            label=algorithm
        )
        ax4.plot(
            plotting_input[algorithm]['test']['Z_Ztautau']['pt_bin_centers'],
            plotting_input[algorithm]['test']['Z_Ztautau']['pt_ratio_IQR'],
            label=algorithm
        )
    ax1.title.set_text(r'ee $\rightarrow$ ZH (H $\rightarrow \tau\tau$)')
    ax2.title.set_text(r'ee $\rightarrow$ Z (Z $\rightarrow \tau\tau$)')
    ax1.set_ylabel(r'$q_{50}(reco/gen)$')
    ax3.set_ylabel(r"$q_{75} - q_{25}(reco/gen)$")
    ax3.set_xlabel(r'$p_T^{gen}$')
    ax4.set_xlabel(r'$p_T^{gen}$')
    ax1.set_xlim(30, 180)
    ax2.set_xlim(30, 180)
    ax3.set_xlim(30, 180)
    ax4.set_xlim(30, 180)
    ax1.set_ylim(0.96, 1.04)
    ax3.set_ylim(0, 0.1)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    plt.legend()
    output_path = os.path.join(cfg.output_dir, f"median_and_iqr.png")
    plt.savefig(output_path, bbox_inches='tight')


def to_bh(data, bins, cumulative=False):
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    if cumulative:
        h1[:] = np.sum(h1.values()) - np.cumsum(h1)
    return h1


def plot_energy_regression(algorithm_info, cfg):
    plotting_input = get_plotting_input(algorithm_info, cfg)
    for algorithm, properties in cfg.metrics.regression.algorithms.items():
        if properties.load_from_json:
            with open(properties.json_metrics_path, 'rt') as in_file:
                plotting_input.update(json.load(in_file)[algorithm])
    json_output_path = output_path = os.path.join(cfg.output_dir, 'plotting_data.json')
    g.save_to_json(plotting_input, json_output_path)
    plot_median_and_iqr(plotting_input, cfg)
    plot_bins(plotting_input, cfg)
    


def get_plotting_input(algorithm_info: dict, cfg: DictConfig):
    algorithms = {}
    for algorithm_name, algorithm_values in algorithm_info.items():
        datasets = {}
        for dataset_name, dataset_values in algorithm_values.items():
            samples = {}
            for sample_name, sample_data in dataset_values.items():
                gen_tau_p4s = g.reinitialize_p4(sample_data.gen_jet_tau_p4s)
                gen_jet_p4s = g.reinitialize_p4(sample_data.gen_jet_p4s)
                pred_tau_pts = sample_data.tau_pt
                gen_pt_mask = gen_jet_p4s.pt > 15
                sample_data = sample_data[gen_pt_mask]
                label = f"{algorithm_name}: {sample_name}"
                pt_ratio_medians, pt_ratio_IQR, pt_bin_centers, pt_ratio_values = prepare_tau_pt_ratio_data(
                    sample_data=sample_data, cfg=cfg)
                samples[sample_name] = {
                    "pt_ratio_values": pt_ratio_values,
                    "pt_ratio_medians": pt_ratio_medians,
                    "pt_ratio_IQR": pt_ratio_IQR,
                    "pt_resolution_w_IQR": pt_ratio_IQR/pt_ratio_medians,
                    "pt_bin_centers": pt_bin_centers,
                }
            datasets[dataset_name] = samples
        algorithms[algorithm_name] = datasets
    return algorithms


def prepare_tau_pt_ratio_data(
    sample_data: ak.Array,
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
    ratio_medians = np.array([np.median(ratio) for ratio in ratio_values])
    ratio_iqr = np.array([(np.quantile(ratio, 0.75) - np.quantile(ratio, 0.25)) for ratio in ratio_values])
    return ratio_medians, ratio_iqr, bin_centers, ratio_values


def plot_bins(plotting_input, cfg):
    fig, rows = plt.subplots(nrows=3, ncols=4, sharex='col', figsize=(16,9))
    i = 0
    bins = np.linspace(0.5, 1.5, 25)
    bin_edges = cfg.metrics.regression.ratio_plot.bin_edges
    bin_titles = [f"[{bin_edges[i]}, {bin_edges[i+1]}]" for i in range(len(bin_edges) - 1)]
    for row in rows:
        for ax in row:
            plot_data = plotting_input['ParticleTransformer']['test']['ZH_Htautau']['pt_ratio_values'][i]
            histo = hep.histplot(to_bh(plot_data, bins=bins), ax=ax, density=True)
            ax.set_title(bin_titles[i], fontsize=12)
            ax.set_xlim(0.5, 1.5)
            print(ax)
            i += 1
    output_path = os.path.join(cfg.output_dir, f"bin_contents.png")
    plt.savefig(output_path, bbox_inches='tight')
