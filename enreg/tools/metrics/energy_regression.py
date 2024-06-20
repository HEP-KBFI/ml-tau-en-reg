import os
import json
import mplhep as hep
import numpy as np
import awkward as ak
import matplotlib
import boost_histogram as bh
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import matplotlib.colors as colors
from enreg.tools import general as g
from enreg.tools.visualization import base as b

hep.style.use(hep.styles.CMS)

def plot_median_and_iqr(plotting_input, cfg):
    plot_median_iqr(plotting_input, cfg, "z_test", "iqr")
    plot_median_iqr(plotting_input, cfg, "z_test", "median")
    plot_median_iqr(plotting_input, cfg, 'zh_test', "median")
    plot_median_iqr(plotting_input, cfg, 'zh_test', "iqr")


def plot_median_iqr(plotting_input, cfg, dataset, y_data):
    if dataset == 'zh_test':
        title = r'ee $\rightarrow$ ZH (H $\rightarrow \tau\tau$)'
    elif dataset == 'z_test':
        title = r'ee $\rightarrow$ Z (Z $\rightarrow \tau\tau$)'
    else:
        raise ValueError(f"No {dataset} found")

    if y_data == "median":
        y_key = 'pt_ratio_medians'
        y_label = r'$p_T\ scale\ (q_{50})$'
        plt.plot([30, 200], [1, 1], ls='--', c='k')
        y_lim = (0.96, 1.04)
    elif y_data == 'iqr':
        y_key = 'pt_resolution_w_IQR'
        y_lim = (0, 0.1)
        y_label = r"$p_T\ resol.\ (q_{75} - q_{25})/q_{50}$"
    else:
        raise ValueError(f"{y_data} not found")

    plt.title(r'ee $\rightarrow$ ZH (H $\rightarrow \tau\tau$)')
    for algorithm in plotting_input.keys():
        plt.plot(
            plotting_input[algorithm][dataset]['pt_bin_centers'],
            plotting_input[algorithm][dataset][y_key],
            label=algorithm, marker="o"
        )
    plt.ylabel(y_label)
    plt.xlabel(r'$p_T^{gen}$')
    plt.xlim(20, 200)
    plt.ylim(y_lim[0], y_lim[1])
    plt.grid()
    plt.legend()
    output_path = os.path.join(cfg.output_dir, f"{dataset}_{y_data}.pdf")
    plt.savefig(output_path)
    plt.close('all')



def to_bh(data, bins, cumulative=False):
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    if cumulative:
        h1[:] = np.sum(h1.values()) - np.cumsum(h1)
    return h1


def plot_energy_regression(sample_data, algorithm_info, cfg):
    plotting_input = get_plotting_input(sample_data, algorithm_info, cfg)
    for algorithm, properties in cfg.metrics.regression.algorithms.items():
        if properties.load_from_json:
            with open(properties.json_metrics_path, 'rt') as in_file:
                plotting_input.update(json.load(in_file)[algorithm])
    json_output_path = output_path = os.path.join(cfg.output_dir, 'plotting_data.json')
    g.save_to_json(plotting_input, json_output_path)
    plot_median_and_iqr(plotting_input, cfg)
    for algorithm in plotting_input.keys():
        for sample in cfg.comparison_samples:
            plot_bins(plotting_input, algorithm, sample, cfg)

    for sample_name, sample_data in sample_data.items():
        gen_tau_p4s = g.reinitialize_p4(sample_data.gen_jet_tau_p4s)
        gen_jet_p4s = g.reinitialize_p4(sample_data.gen_jet_p4s)
        reco_jet_p4s = g.reinitialize_p4(sample_data.reco_jet_p4s)
        for algorithm_name in algorithm_info[sample_name].fields:
            pred_tau_pts = algorithm_info[sample_name][algorithm_name]
            plot_2d_histogram(
                x_entries=np.array(gen_tau_p4s.pt),
                y_entries=np.array(gen_jet_p4s.pt),
                x_label=r"$\tau\ p_T^{gen}$",
                y_label=r"$jet\ p_T^{gen}$",
                title="",
                cfg=cfg,
                out_filename=f"genTau_vs_genJet_{sample_name}_pt.pdf",
            )
            plot_2d_histogram(
                x_entries=np.array(gen_tau_p4s.pt),
                y_entries=np.array(reco_jet_p4s.pt),
                x_label=r"$jet\ p_T^{gen}$",
                y_label=r"$jet\ p_T^{reco}$",
                title="",
                cfg=cfg,
                out_filename=f"genTau_vs_recoJet_{sample_name}_pt.pdf",
            )
            plot_2d_histogram(
                x_entries=np.array(gen_tau_p4s.pt),
                y_entries=np.array(pred_tau_pts),
                x_label=r"$\tau\ p_T^{gen}$",
                y_label=r"$\tau\ p_T^{reco}$",
                title=f"{algorithm_name}:{sample_name}",
                cfg=cfg,
                out_filename=f"genTau_vs_predTau_pt_{algorithm_name}_{sample_name}.pdf",
            )
            plot_true_v_pred_1D_histo(
                true=np.array(gen_tau_p4s.pt),
                pred=np.array(pred_tau_pts),
                x_label="pT",
                y_label="Number entries",
                true_label="Truth",
                pred_label="Prediction",
                title=f"{algorithm_name}:{sample_name}",
                cfg=cfg,
                out_filename=f"true_pred_pt_distributions_{algorithm_name}_{sample_name}.pdf",
            )
    plot_overall_ratio(plotting_input, cfg)


def get_plotting_input(sample_info: dict, algorithm_info: dict, cfg: DictConfig):
    samples = {}
    for sample_name, sample_data in sample_info.items():
        gen_tau_p4s = g.reinitialize_p4(sample_data.gen_jet_tau_p4s)
        gen_jet_p4s = g.reinitialize_p4(sample_data.gen_jet_p4s)

        gen_pt_mask = gen_tau_p4s.pt > 15
        sample_data = sample_data[gen_pt_mask]

        algorithms = {}
        for algorithm_name in algorithm_info[sample_name].fields:
            algorithm_data = algorithm_info[sample_name][algorithm_name]
            pred_tau_pts = algorithm_data[gen_pt_mask]

            label = f"{algorithm_name}: {sample_name}"
            pt_ratio_medians, pt_ratio_IQR, pt_bin_centers, pt_ratio_values = prepare_tau_pt_ratio_data(
                sample_name=sample_name,
                sample_data=sample_data,
                tau_pt=pred_tau_pts,
                cfg=cfg
            )
            algorithms[algorithm_name] = {
                "pt_ratio_values": pt_ratio_values,
                "pt_ratio_medians": pt_ratio_medians,
                "pt_ratio_IQR": pt_ratio_IQR,
                "pt_resolution_w_IQR": pt_ratio_IQR/pt_ratio_medians,
                "pt_bin_centers": pt_bin_centers,
            }
        samples[sample_name] = algorithms

    algos = list(samples[sample_name].keys())
    by_algo = {}
    for algo in algos:
        by_algo[algo] = {sn: samples[sn][algo] for sn in samples.keys()}
    return by_algo


def prepare_tau_pt_ratio_data(
    sample_name: str,
    sample_data: ak.Array,
    tau_pt: ak.Array,
    cfg: DictConfig
):
    reco_gen_pt_ratio = tau_pt / g.reinitialize_p4(sample_data.gen_jet_tau_p4s).pt
    gen_vis_tau_pt = g.reinitialize_p4(sample_data.gen_jet_tau_p4s).pt
    bin_edges = np.array(cfg.metrics.regression.ratio_plot.bin_edges[sample_name])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_gen_tau_pt = np.digitize(gen_vis_tau_pt, bins=bin_edges)  # Biggest idx is overflow
    ratio_values = [reco_gen_pt_ratio[binned_gen_tau_pt == bin_idx].to_numpy() for bin_idx in range(1, len(bin_edges))]
    ratio_medians = np.array([np.median(ratio) for ratio in ratio_values])
    ratio_iqr = np.array([(np.quantile(ratio, 0.75) - np.quantile(ratio, 0.25)) for ratio in ratio_values])
    return ratio_medians, ratio_iqr, bin_centers, ratio_values


def plot_bins(plotting_input, algorithm, sample, cfg):
    fig, rows = plt.subplots(nrows=3, ncols=4, sharex='col', figsize=(16,9))
    plt.title(f"{algorithm}: {sample}")
    bins = np.linspace(0.5, 1.5, 101)
    bin_edges = list(cfg.metrics.regression.ratio_plot.bin_edges[sample])
    bin_titles = [f"$p_{{\\tau,true}} \in [{bin_edges[i]}, {bin_edges[i+1]}]\ GeV$" for i in range(len(bin_edges) - 1)]
    axes = rows.flatten()
    if len(bin_edges) - 1 > len(axes):
        raise Exception("more bins than axes: {} > {}".format(len(bin_edges), len(axes))) 
    for i in range(len(bin_edges)-1):
        ax = axes[i]
        plot_data = plotting_input[algorithm][sample]['pt_ratio_values'][i]
        histo = hep.histplot(to_bh(plot_data, bins=bins), ax=ax, density=True)
        ax.set_title(bin_titles[i], fontsize=12)
        ax.set_xlim(0.5, 1.5)
        ax.set_xlabel("$q$", fontsize=12)

    #delete unused axes
    for iax in range(len(axes)):
        if iax >= len(bin_edges)-1:
            axes[iax].set_title("")
    output_path = os.path.join(cfg.output_dir, f"bin_contents_{algorithm}_{sample}.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def plot_2d_histogram(
    x_entries,
    y_entries,
    x_label,
    y_label,
    title,
    cfg,
    out_filename,
    b_min=0,
    b_max=220,
    n_bins=221,
    log_color=True
):
    b = np.linspace(b_min, b_max, n_bins)
    plt.hist2d(
        x_entries,
        y_entries,
        bins=(b,b),
        norm=matplotlib.colors.LogNorm() if log_color else None,
        cmap="Blues",
    )
    plt.colorbar()
    plt.plot([b_min, b_max],[b_min, b_max], color="black", ls="--", lw=1.0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    output_path = os.path.join(cfg.output_dir, out_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def plot_true_v_pred_1D_histo(
    true,
    pred,
    x_label,
    y_label,
    true_label,
    pred_label,
    title,
    cfg,
    out_filename,
    b_min=0,
    b_max=220,
    n_bins=111,
):
    b = np.linspace(b_min, b_max, n_bins)
    plt.hist(pred, bins=b, histtype='step', label=pred_label)
    plt.hist(true, bins=b, histtype='step', label=true_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    output_path = os.path.join(cfg.output_dir, out_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def plot_overall_ratio(plotting_input, cfg):
    be = np.linspace(0, 2, 100)
    for sample in cfg.comparison_samples:
        fig = plt.figure()
        ax = plt.axes()
        for algorithm in plotting_input.keys():
            all_values = []
            for b_values in plotting_input[algorithm][sample]['pt_ratio_values']:
                all_values.extend(b_values)
            plt.hist(all_values, bins=be, label=algorithm, histtype='step')
        plt.axvline(x=1, ymin=0, ymax=1e5, color="black")
        plt.yscale('log')
        plt.legend(loc=1)
        output_path = os.path.join(cfg.output_dir, f"ratioplot_{sample}.pdf")
        plt.xlabel("$response\ q=p_{T}^{reco}/p_{T}^{gen}$")
        ax.set_ylim(top=ax.get_ylim()[1]*10)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close('all')
