import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import confusion_matrix
from enreg.tools.visualization.histogram import Histogram

hep.style.use(hep.styles.CMS)


def plot_roc(
    efficiencies: dict,
    fakerates: dict,
    colors: dict,
    markers: dict,
    ylim: tuple =(1e-5, 1),
    xlim: tuple =(0, 1),
    title: str = "",
    x_maj_tick_spacing: float = 0.2,
    name_map: dict = {},
    figsize: tuple =(12, 12),
    output_path: str = "ROC.pdf",
) -> None:
    """ Plots the receiver operating characteristic (ROC) curve for the algorithms present in the efficiencies. 

    Args:
        efficiencies : dict
            Dictionary containing the algorithm names as keys and a list of efficiencies as the values. The length of the
            `efficiencies` corresponds to the length of `fakerates`.
        fakerates : dict 
            Dictionary containing the algorithm names as keys and a list of fakerates as the values. The length of the
            `fakerates` corresponds to the length of `efficiencies`.
        colors : dict
            Dictionary assigning a color to each of the algorithm present in the list of `efficiencies` and `fakerates`.
        markers : dict
            Dictionary assigning a marker to each of the algorithm present in the list of `efficiencies` and `fakerates`.
        ylim : tuple
            [default: (1e-5, 1)] Given in the format of (ymin, ymax). The limits on the y-axis to be shown on the figure.
        xlim : tuple
            [default: (0, 1)] Given in the format of (xmin, xmax). The limits on the x-axis to be shown on the figure.
        title : str
            [default: ""] Title of the figure.
        x_maj_tick_spacing : float
            [default: 0.2] Spacing of the major x-axis ticks.
        name_map : dict
            [default: {}] Mapping between the algorithm name and the name printed onto the figure for the given algorithm.
        figsize : tuple
            [default: (12, 12)] The size of the figure that will be created
        output_path : str
            [default: "ROC.pdf"] Destination where the figure will be saved.

    Returns:
        None
    """
    hep.style.use(hep.styles.CMS)
    name_map = {algorithm: algorithm if algorithm not in name_map.keys() else name_map[algorithm] for algorithm in efficiencies.keys()}
    fig, ax = plt.subplots(figsize=figsize)
    for algorithm in efficiencies.keys():
        mask = np.array(fakerates[algorithm]) != 0.0
        x_values = np.array(efficiencies[algorithm])[mask]
        y_values = np.array(fakerates[algorithm])[mask]
        plt.plot(
            x_values,
            y_values,
            color=colors[algorithm],
            marker=markers[algorithm],
            label=name_map[algorithm],
            lw=2,
            ls="",
            markevery=0.02,
            ms=12,
        )
    plt.grid()
    plt.legend(prop={"size": 30})
    plt.title(title, loc="left")
    plt.ylabel(r"$P_{misid}$", fontsize=30)
    plt.xlabel(r"$\varepsilon_{\tau}$", fontsize=30)
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)
    plt.ylim(ylim)
    plt.xlim(xlim)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_maj_tick_spacing))
    plt.yscale("log")
    plt.savefig(output_path, format="pdf")
    plt.close("all")



def plot_regression_confusion_matrix(
    y_true: np.array,
    y_pred: np.array,
    output_path: str,
    left_bin_edge: float = 0.0,
    right_bin_edge: float = 1.0,
    n_bins: int = 24,
    figsize: tuple = (12, 12),
    cmap: str = "Greys",
    y_label: str = "Predicted",
    x_label: str = "Truth",
    title: str = "Confusion matrix",
) -> None:
    """Plots the confusion matrix for the regression task. Although confusion
    matrix is in principle meant for classification task, the problem can be
    solved by binning the predictions and truth values.

    Args:
        y_true : np.array
            The array containing the truth values with shape (n,)
        y_pred : np.array
            The array containing the predicted values with shape (n,)
        output_path : str
            The path where output plot will be saved
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

    Returns:
        None
    """
    bin_edges = np.linspace(left_bin_edge, right_bin_edge, num=n_bins + 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.label_outer()
    bin_counts = np.histogram2d(y_true, y_pred, bins=[bin_edges, bin_edges])[0]
    im = ax.pcolor(bin_edges, bin_edges, bin_counts.transpose(), cmap=cmap, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect("equal")
    ax.set_ylabel(f"{y_label}")
    ax.set_xlabel(f"{x_label}")
    plt.title(title, fontsize=18, loc="center", fontweight="bold", style="italic", family="monospace")
    plt.savefig(output_path, format="pdf")
    plt.close("all")


def plot_classification_confusion_matrix(
    true_cats: np.array,
    pred_cats: np.array,
    categories: list,
    output_path: str,
    cmap: str = "gray",
    bin_text_color: str = "r",
    y_label: str = "Prediction",
    x_label: str = "Truth",
    title: str = "",
    figsize: tuple = (12, 12),
) -> None:
    """Plots the confusion matrix for the classification task. Confusion
    matrix functions has the categories in the other way in order to have the
    truth on the x axis.
    Args:
        true_cats : np.array,
            The true categories
        pred_cats : np.array
            The predicted categories
        categories : list
            Category labels in the correct order
        output_path : str
            The path where the plot will be outputted
        cmap : str
            [default: "gray"] The colormap to be used
        bin_text_color : str
            [default: "r"] The color of the text on bins
        y_label : str
            [default: "Predicted"] The label for the y-axis
        x_label : str
            [default: "Truth"] The label for the x-axis
        title : str
            [default: "Confusion matrix"] The title for the plot
        figsize : tuple
            The size of the figure drawn

    Returns:
        None
    """
    histogram = confusion_matrix(true_cats, pred_cats, normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    hep.style.use(hep.style.ROOT)
    xbins = ybins = np.arange(len(categories) + 1)
    tick_values = np.arange(len(categories)) + 0.5
    hep.hist2dplot(histogram, xbins, ybins, cmap=cmap, cbar=True)
    plt.xticks(tick_values, categories, fontsize=14, rotation=0)
    plt.yticks(tick_values + 0.2, categories, fontsize=14, rotation=90, va="center")
    plt.xlabel(f"{x_label}", fontdict={"size": 14})
    plt.ylabel(f"{y_label}", fontdict={"size": 14})
    ax.tick_params(axis="both", which="both", length=0)
    for i in range(len(ybins) - 1):
        for j in range(len(xbins) - 1):
            bin_value = histogram.T[i, j]
            ax.text(
                xbins[j] + 0.5,
                ybins[i] + 0.5,
                f"{bin_value:.2f}",
                color=bin_text_color,
                ha="center",
                va="center",
                fontweight="bold",
            )
    plt.savefig(output_path, format="pdf")
    plt.close("all")


def plot_histogram(
    entries: np.array,
    output_path: str,
    left_bin_edge: float = 0.0,
    right_bin_edge: float = 1.0,
    n_bins: int = 24,
    figsize: tuple = (12, 12),
    y_label: str = "",
    x_label: str = "",
    title: str = "",
    integer_bins: bool = False,
    hatch="//",
    color="blue",
) -> None:
    """Plots the confusion matrix for the regression task. Although confusion
    matrix is in principle meant for classification task, the problem can be
    solved by binning the predictions and truth values.

    Args:
        entries : np.array
            The array containing the truth values with shape (n,)
        output_path : str
            The path where output plot will be saved
        left_bin_edge : float
            [default: 0.0] The smallest value
        right_bin_edge : float
            [default: 1.0] The largest value
        n_bins : int
            [default: 24] The number of bins the values will be divided into
            linearly. The number of bin edges will be n_bin_edges = n_bins + 1
        figsize : tuple
            [default: (12, 12)] The size of the figure that will be created
        y_label : str
            [default: "Predicted"] The label for the y-axis
        x_label : str
            [default: "Truth"] The label for the x-axis
        title : str
            [default: "Confusion matrix"] The title for the plot

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)
    if integer_bins:
        bin_diff = np.min(np.diff(np.unique(entries)))
        left_of_first_bin = np.min(entries) - float(bin_diff) / 2
        right_of_last_bin = np.max(entries) + float(bin_diff) / 2
        hist, bin_edges = np.histogram(entries, bins=np.arange(left_of_first_bin, right_of_last_bin + bin_diff, bin_diff))
    else:
        hist, bin_edges = np.histogram(entries, bins=np.linspace(left_bin_edge, right_bin_edge, num=n_bins + 1))
    # hep.histplot(hist, bin_edges, yerr=True, label=title, hatch=hatch, color=color)
    hep.histplot(hist, bin_edges, label=title, hatch=hatch, color=color)
    plt.xlabel(x_label, fontdict={"size": 20})
    plt.ylabel(y_label, fontdict={"size": 20})
    plt.grid(True, which="both")
    plt.yscale("log")
    plt.title(title, loc="left")
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))
    # textstr = "\n".join((r"$\mu=%.2f$" % (np.mean(entries),), r"$\sigma=%.2f$" % (np.std(entries),)))
    # props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
    # ax.text(1.07, 0.6, textstr, transform=ax.transAxes, fontsize=16, verticalalignment="top", bbox=props)
    plt.savefig(output_path)#, format="pdf")
    plt.close("all")


def plot_decaymode_correlation_matrix(
    true_cats: np.array,
    pred_cats: np.array,
    categories: list,
    output_path: str,
    cmap: str = "gray",
    bin_text_color: str = "k",
    y_label: str = "Prediction",
    x_label: str = "Truth",
    title: str = "",
    figsize: tuple = (13, 13),
) -> None:
    """Plots the confusion matrix for the classification task. Confusion
    matrix functions has the categories in the other way in order to have the
    truth on the x axis.
    Args:
        true_cats : np.array,
            The true categories
        pred_cats : np.array
            The predicted categories
        categories : list
            Category labels in the correct order
        output_path : str
            The path where the plot will be outputted
        cmap : str
            [default: "gray"] The colormap to be used
        bin_text_color : str
            [default: "r"] The color of the text on bins
        y_label : str
            [default: "Predicted"] The label for the y-axis
        x_label : str
            [default: "Truth"] The label for the x-axis
        title : str
            [default: "Confusion matrix"] The title for the plot
        figsize : tuple
            The size of the figure drawn
    Returns:
        None
    """
    histogram = confusion_matrix(true_cats, pred_cats, normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    hep.style.use(hep.style.ROOT)
    xbins = ybins = np.arange(len(categories) + 1)
    tick_values = np.arange(len(categories)) + 0.5
    hep.hist2dplot(histogram, xbins, ybins, cmap=cmap, alpha=0, cbar=False)
    plt.xticks(tick_values, categories, fontsize=22, rotation=0)
    plt.yticks(tick_values, categories, fontsize=22, rotation=90, va="center")
    plt.xlabel(f"{x_label}", fontdict={"size": 28})
    plt.ylabel(f"{y_label}", fontdict={"size": 28})
    ax.tick_params(axis="both", which="major", length=10)
    ax.tick_params(axis="both", which="minor", length=0)
    for i in range(len(ybins) - 1):
        for j in range(len(xbins) - 1):
            bin_value = histogram.T[i, j]
            ax.text(
                xbins[j] + 0.5,
                ybins[i] + 0.5,
                f"{bin_value:.2f}",
                color=bin_text_color,
                ha="center",
                va="center",
                fontweight="bold",
            )
    plt.savefig(output_path, format="pdf")
    plt.close("all")


def plot_decaymode_correlation_matrix_removed_row(
    true_cats: np.array,
    pred_cats: np.array,
    categories: list,
    output_path: str,
    cmap: str = "gray",
    bin_text_color: str = "k",
    y_label: str = "Prediction",
    x_label: str = "Truth",
    title: str = "",
    figsize: tuple = (13, 13),
) -> None:
    """Plots the confusion matrix for the classification task. Confusion
    matrix functions has the categories in the other way in order to have the
    truth on the x axis.
    Args:
        true_cats : np.array,
            The true categories
        pred_cats : np.array
            The predicted categories
        categories : list
            Category labels in the correct order for x-axis
        output_path : str
            The path where the plot will be outputted
        cmap : str
            [default: "gray"] The colormap to be used
        bin_text_color : str
            [default: "r"] The color of the text on bins
        y_label : str
            [default: "Predicted"] The label for the y-axis
        x_label : str
            [default: "Truth"] The label for the x-axis
        title : str
            [default: "Confusion matrix"] The title for the plot
        figsize : tuple
            The size of the figure drawn
    Returns:
        None
    """
    histogram = confusion_matrix(true_cats, pred_cats, normalize="true")
    histogram = histogram[:, :-1]
    fig, ax = plt.subplots(figsize=figsize)
    # ax.set_aspect("equal", adjustable="box")
    hep.style.use(hep.style.ROOT)
    xbins = np.arange(len(categories) + 1)
    ybins = np.arange(len(categories))
    tick_values_x = np.arange(len(categories)) + 0.5
    tick_values_y = np.arange(len(categories) - 1) + 0.5
    hep.hist2dplot(histogram, xbins, ybins, cmap=cmap, alpha=0, cbar=False)
    plt.xticks(tick_values_x, categories, fontsize=22, rotation=0)
    plt.yticks(tick_values_y, categories[:-1], fontsize=22, rotation=90, va="center")
    plt.xlabel(f"{x_label}", fontdict={"size": 28})
    plt.ylabel(f"{y_label}", fontdict={"size": 28})
    ax.tick_params(axis="both", which="major", length=10)
    ax.tick_params(axis="both", which="minor", length=0)
    for i in range(len(ybins) - 1):
        for j in range(len(xbins) - 1):
            bin_value = histogram.T[i, j]
            ax.text(
                xbins[j] + 0.5,
                ybins[i] + 0.5,
                f"{bin_value:.2f}",
                color=bin_text_color,
                ha="center",
                va="center",
                fontweight="bold",
            )
    plt.savefig(output_path, format="pdf")
    plt.close("all")


def plot_efficiency(
    eff_fake_data,
    key,
    cfg,
    output_dir,
    name_map: dict = {},
):
    name_map = {algorithm: algorithm if algorithm not in name_map.keys() else name_map[algorithm] for algorithm in efficiencies.keys()}
    ylabels = {
        "fakerate": r"P_{misid}",
        "efficiency": r"\varepsilon_{\tau}"
    }
    objects = {
        "fakerate": r"{gen\mathrm{-}jet}",
        "efficiency": r"{gen\mathrm{-}\tau_h}"
    }
    labels = {
        "pt": "p_T",
        "eta": r"\eta",
        "theta": r"\theta"
    }
    units = {
        "pt": "[GeV]",
        "eta": "[GeV]",
        "theta": "[ ^{o} ]"
    }
    yscales = {
        "fakerate": "log",
        "efficiency": "linear"
    }
    ylims = {
        "fakerate": (5e-6, 2e-2),
        "efficiency": None
    }
    x_maj_tick_spacings = {
        "pt": 40,
        "eta": 20,
        "theta": 20
    }

    if key == "fakerates":
        metrics = cfg.metrics.fakerate.variables
    else:
        metrics = cfg.metrics.efficiency.variables
    for metric in metrics:
        output_path = os.path.join(output_dir, f"{metric.name}_{key}.pdf")
        fig, ax = plt.subplots(figsize=(12, 12))
        algorithms = eff_fake_data.keys()
        for i, algorithm in enumerate(algorithms):
            if metric.name == "theta":
                numerator = process_theta_values(eff_fake_data[algorithm][metric.name]["numerator"])
                denominator = process_theta_values(eff_fake_data[algorithm][metric.name]["denominator"])
            else:
                numerator = eff_fake_data[algorithm][metric.name]["numerator"]
                denominator = eff_fake_data[algorithm][metric.name]["denominator"]
            bin_edges = np.linspace(metric.x_range[0], metric.x_range[1], num=metric.n_bins + 1)
            numerator_hist = Histogram(numerator, bin_edges, "numerator")
            denominator_hist = Histogram(denominator, bin_edges, "denominator")
            resulting_hist = numerator_hist / denominator_hist
            plt.errorbar(
                resulting_hist.bin_centers,
                resulting_hist.binned_data,
                xerr=resulting_hist.bin_halfwidths,
                yerr=resulting_hist.uncertainties,
                ms=20,
                color=cfg.colors[algorithm],
                marker=cfg.markers[algorithm],
                linestyle="None",
                label=name_map[algorithm],
            )
        plt.grid()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_maj_tick_spacings[metric]))
        matplotlib.rcParams["axes.unicode_minus"] = False
        plt.xlabel(r"${}^{} \,\, {}$".format(labels[metric], objects[key], units[metric]), fontsize=30)
        plt.ylabel(r"${}$".format(ylabels[key]), fontsize=30)
        plt.ylim(ylims[key])
        plt.yscale(yscales[key])
        ##
        #plt.legend(prop={"size": 30}) -> only if efficiency
        ##
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        plt.savefig(output_path, format="pdf")
        plt.close("all")


def process_theta_values(data: np.array) -> np.array:
    """ Transforms the theta coordinates from radians to degrees

    Args:
        data : np.array
            The data to be transformed

    Returns:
        transformed_data : ak.Array
            The transformed data.
    """
    transformed_data = 90 - np.abs(np.rad2deg(np.array(data)) - 90)
    return transformed_data

