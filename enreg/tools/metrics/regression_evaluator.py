import os
import json
import numpy as np
import mplhep as hep
import boost_histogram as bh
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from enreg.tools.general import NpEncoder
hep.style.use(hep.styles.CMS)


def plot_regression_confusion_matrix(
        y_true: np.array,
        y_pred: np.array,
        left_bin_edge: float = 0.0,
        right_bin_edge: float = 1.0,
        n_bins: int = 24,
        figsize: tuple = (12, 12),
        cmap: str = "Greys",
        y_label: str = "Predicted",
        x_label: str = "Truth",
        title: str = "Confusion matrix",
):
    """Plots the confusion matrix for the regression task. Although confusion
    matrix is in principle meant for classification task, the problem can be
    solved by binning the predictions and truth values.

    Args:
        y_true : np.array
            The array containing the truth values with shape (n,)
        y_pred : np.array
            The array containing the predicted values with shape (n,)
        left_bin_edge : float
            [default: 0.0] The smallest value
        right_bin_edge : float
            [default: 1.0] The largest value
        n_bins : int
            [default: 24] The number of bins the values will be divided into
            linearly. The number of bin edges will be n_bin_edges = n_bins + 1
        figsize : tuple
            [default: (12, 12)] The size of the figure that will be created
        cmap : str
            [default: "Greys"] Name of the colormap to be used for the
            confusion matrix
        y_label : str
            [default: "Predicted"] The label for the y-axis
        x_label : str
            [default: "Truth"] The label for the x-axis
        title : str
            [default: "Confusion matrix"] The title for the plot

    """
    bin_edges = np.linspace(left_bin_edge, right_bin_edge, num=n_bins + 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.label_outer()
    bin_counts = np.histogram2d(y_true, y_pred, bins=[bin_edges, bin_edges])[0]
    im = ax.pcolor(bin_edges, bin_edges, bin_counts.T, cmap=cmap, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect("equal")
    ax.set_ylabel(f"{y_label}")
    ax.set_xlabel(f"{x_label}")
    ax.set_title(title, fontsize=18, loc="center", fontweight="bold", style="italic", family="monospace")
    return fig, ax


def IQR(ratios: np.array) -> np.array:
    return np.quantile(ratios, 0.75) - np.quantile(ratios, 0.25)


def to_bh(data, bins, cumulative=False):
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    if cumulative:
        h1[:] = np.sum(h1.values()) - np.cumsum(h1)
    return h1


def calculate_bin_centers(edges: np.array) -> np.array:
    bin_widths = np.array([edges[i + 1] - edges[i] for i in range(len(edges) - 1)])
    bin_centers = []
    for i in range(len(edges) - 1):
        bin_centers.append(edges[i] + (bin_widths[i] / 2))
    return np.array(bin_centers), bin_widths / 2


class RegressionEvaluator:
    def __init__(
            self,
            prediction: np.array,
            truth: np.array,
            cfg: DictConfig,
            sample_name: str,
            algorithm: str,
    ):
        self.cfg = cfg
        self.prediction = prediction
        self.truth = truth
        self.gen_tau_pt = truth
        self.algorithm = algorithm
        self.ratios = prediction / truth
        self.resolution_function = IQR
        self.sample = sample_name
        self.response_function = np.median
        self.bin_edges = np.array(self.cfg.metrics.regression.ratio_plot.bin_edges[sample_name])
        self.bin_centers = calculate_bin_centers(self.bin_edges)[0]
        self.resolutions, self.responses, self.binned_ratios = self._get_binned_values(self.ratios, self.truth)
        self.resolution, self.response = self._get_overall_resoluton_response()

    def _get_binned_values(self, ratios, gen_tau_vis_pt):
        binned_gen_tau_pt = np.digitize(gen_tau_vis_pt, bins=self.bin_edges)  # Biggest idx is overflow
        binned_ratios = [ratios[binned_gen_tau_pt == bin_idx].to_numpy() for bin_idx in
                         range(1, len(self.bin_edges))]
        resolutions = np.array([self.resolution_function(ratios) for ratios in binned_ratios])
        responses = np.array([self.response_function(ratios) for ratios in binned_ratios])
        return resolutions/responses, responses, binned_ratios

    def _get_overall_resoluton_response(self):
        response = self.response_function(self.ratios)
        resolution = self.resolution_function(self.ratios)
        resolution = resolution / response
        return resolution, response

    def print_results(self):
        print("----------------------------")
        print(f"-------- {self.algorithm}--------")
        print(f"Resolution: {self.resolution} \t Response: {self.response}")
        print("----------------------------")


class RangeContentPlot:
    def __init__(self, cfg, sample_name):
        self.cfg = cfg
        self.sample_name = sample_name
        self.bin_edges = np.array(self.cfg.metrics.regression.ratio_plot.bin_edges[self.sample_name])
        self.fig, self.axes = self.plot()

    def plot(self):
        fig, rows = plt.subplots(nrows=3, ncols=4, sharex='col', figsize=(16, 9))
        axes = rows.flatten()
        for i, ax in enumerate(axes):
            if i == (len(self.bin_edges) - 1):
                break
            ax.set_title(r"$p_{\tau,true} \in$" + f"$[{self.bin_edges[i]}, {self.bin_edges[i + 1]}]\ GeV$", fontsize=12)
            ax.set_xlim(0.5, 1.5)
            ax.set_xlabel("$q$", fontsize=12)
        return fig, axes

    def add_line(self, evaluator):
        bins = np.linspace(0.5, 1.5, 101)
        for ax, data in zip(self.axes, evaluator.binned_ratios):
            hep.histplot(to_bh(data, bins=bins), ax=ax, density=True, label=evaluator.algorithm)
            ax.text(0.05, 0.95, f'IQR = {IQR(data):.3f}', transform=ax.transAxes, fontsize=8, va='top', ha='left')



    def save(self, output_path):
        self.fig.savefig(output_path, bbox_inches='tight', format="pdf")
        plt.close("all")


class LinePlot:
    def __init__(
            self,
            cfg: DictConfig,
            xlabel: str,
            ylabel: str,
            xscale: str = 'linear',
            yscale: str = 'linear',
            ymin: float = 0,
            ymax: float = 1,
            nticks: int = 7
    ):
        self.cfg = cfg
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xscale = xscale
        self.yscale = yscale
        self.nticks = nticks
        self.ymin, self.ymax = ymin, ymax
        self.fig, self.ax = self.plot()

    def add_line(self, x_values, y_values, algorithm, label=""):
        if label == "":
            label = self.cfg.ALGORITHM_PLOT_STYLES[algorithm].name
        self.ax.plot(
            x_values,
            y_values,
            label=label,
            marker=self.cfg.ALGORITHM_PLOT_STYLES[algorithm].marker,
            color=self.cfg.ALGORITHM_PLOT_STYLES[algorithm].color,
            ls=self.cfg.ALGORITHM_PLOT_STYLES[algorithm].ls,
            lw=self.cfg.ALGORITHM_PLOT_STYLES[algorithm].lw,
            ms=10
        )
        self.ax.legend()

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_yscale(self.yscale)
        ax.set_xscale(self.xscale)
        ax.set_ylim((self.ymin, self.ymax))
        ax.grid()
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.linspace(start, end, self.nticks))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
        return fig, ax

    def save(self, output_path: str):
        self.fig.savefig(output_path, bbox_inches='tight', format="pdf")
        plt.close("all")


class Resolution2DPlot:
    def __init__(self, cfg, sample, evaluator):
        self.cfg = cfg
        self.sample = sample
        self.evaluator = evaluator
        self.fig, self.ax = self.plot()

    def plot(self):
        fig, ax = plot_regression_confusion_matrix(
            y_true=self.evaluator.truth.to_numpy(),
            y_pred=self.evaluator.prediction.to_numpy(),
            left_bin_edge=0.0,
            right_bin_edge=self.cfg.metrics.regression.ratio_plot.bin_edges[self.sample][-1],
            n_bins=24,
            figsize=(8, 9),
            cmap="Greys",
            y_label=r"Predicted $p_T$ [GeV]",
            x_label=r"True $p_T$ [GeV]",
            title=None
        )
        return fig, ax

    def save(self, output_path: str):
        self.fig.savefig(output_path, bbox_inches='tight', format="pdf")
        plt.close("all")


class RegressionMultiEvaluator:
    def __init__(self, output_dir: str, cfg: DictConfig, sample: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cfg = cfg
        self.sample = sample
        self.response_lineplot = LinePlot(
            cfg=self.cfg,
            xlabel=cfg.metrics.regression.ratio_plot.response_plot.xlabel,
            ylabel=cfg.metrics.regression.ratio_plot.response_plot.ylabel,
            xscale=cfg.metrics.regression.ratio_plot.response_plot.xscale,
            yscale=cfg.metrics.regression.ratio_plot.response_plot.yscale,
            ymin=cfg.metrics.regression.ratio_plot.response_plot.ylim[0],
            ymax=cfg.metrics.regression.ratio_plot.response_plot.ylim[1],
            nticks=cfg.metrics.regression.ratio_plot.response_plot.nticks,
        )
        self.resolution_lineplot = LinePlot(
            cfg=self.cfg,
            xlabel=cfg.metrics.regression.ratio_plot.resolution_plot.xlabel,
            ylabel=cfg.metrics.regression.ratio_plot.resolution_plot.ylabel,
            xscale=cfg.metrics.regression.ratio_plot.resolution_plot.xscale,
            yscale=cfg.metrics.regression.ratio_plot.resolution_plot.yscale,
            ymin=cfg.metrics.regression.ratio_plot.resolution_plot.ylim[0],
            ymax=cfg.metrics.regression.ratio_plot.resolution_plot.ylim[1],
            nticks=cfg.metrics.regression.ratio_plot.resolution_plot.nticks,
        )
        self.bin_distributions_plots = {}
        self.resolution_2d_plots = {}
        self.resolution_performance_info = {}

    def combine_results(self, evaluators: list):
        for evaluator in evaluators:
            self.response_lineplot.add_line(evaluator.bin_centers, evaluator.responses, evaluator.algorithm, label="")
            self.resolution_lineplot.add_line(evaluator.bin_centers, evaluator.resolutions, evaluator.algorithm,
                                              label="")
            self.resolution_2d_plots[evaluator.algorithm] = Resolution2DPlot(self.cfg, self.sample, evaluator)
            self.bin_distributions_plots[evaluator.algorithm] = RangeContentPlot(self.cfg, self.sample)
            self.bin_distributions_plots[evaluator.algorithm].add_line(evaluator)
            if evaluator.sample not in self.resolution_performance_info.keys():
                self.resolution_performance_info[evaluator.sample] = {}
            self.resolution_performance_info[evaluator.sample][evaluator.algorithm] = {
                "resolution": evaluator.resolution,
                "response": evaluator.response
            }

    def save(self):
        responses_output_path = os.path.join(self.output_dir, "responses.pdf")
        self.response_lineplot.save(responses_output_path)
        resolutions_output_path = os.path.join(self.output_dir, "resolutions.pdf")
        self.resolution_lineplot.save(resolutions_output_path)
        for algorithm, res_2d_plot in self.resolution_2d_plots.items():
            res_2d_plot_output_path = os.path.join(self.output_dir, f"{algorithm}_{self.sample}_2D_resolution.pdf")
            res_2d_plot.save(res_2d_plot_output_path)
            bin_distributions_plot_output_path = os.path.join(self.output_dir, f"{algorithm}_{self.sample}_bin_contents.pdf")
            self.bin_distributions_plots[algorithm].save(bin_distributions_plot_output_path)
        resolution_performance_info_path = os.path.join(self.output_dir, "performance_info.json")
        with open(resolution_performance_info_path, "wt") as out_file:
            json.dump(self.resolution_performance_info, out_file, indent=4, cls=NpEncoder)
