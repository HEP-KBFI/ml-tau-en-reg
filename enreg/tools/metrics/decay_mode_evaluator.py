import os
import json
import numpy as np
import mplhep as hep
from sklearn import metrics
import matplotlib.pyplot as plt
from enreg.tools.general import NpEncoder

hep.style.use(hep.styles.CMS)


def visualize_confusion_matrix(
        histogram: np.array,
        categories: list,
        cmap: str = "Greys",
        bin_text_color: str = "r",
        y_label: str = "Predicted decay modes",
        x_label: str = "True decay modes",
        figsize: tuple = (12, 12),
):
    """Plots the confusion matrix for the classification task. Confusion
    matrix functions has the categories in the other way in order to have the
    truth on the x-axis.
    Args:
        histogram : np.array
            Histogram produced by the sklearn.metrics.confusion_matrix.
        categories : list
            Category labels in the correct order.
        cmap : str
            [default: "gray"] The colormap to be used.
        bin_text_color : str
            [default: "r"] The color of the text on bins.
        y_label : str
            [default: "Predicted"] The label for the y-axis.
        x_label : str
            [default: "Truth"] The label for the x-axis.
        figsize : tuple
            The size of the figure drawn.
    """
    fig, ax = plt.subplots(figsize=figsize)
    xbins = ybins = np.arange(len(categories) + 1)
    tick_values = np.arange(len(categories)) + 0.5
    hep.hist2dplot(histogram, xbins, ybins, cmap=cmap, cbar=True, flow=None)
    plt.xticks(tick_values, categories, fontsize=14, rotation=0)
    plt.yticks(tick_values + 0.2, categories, fontsize=14, rotation=90, va="center")
    plt.xlabel(f"{x_label}", fontdict={"size": 14})
    plt.ylabel(f"{y_label}", fontdict={"size": 14})
    ax.tick_params(axis="both", which="both", length=0)
    for i in range(len(ybins) - 1):
        for j in range(len(xbins) - 1):
            bin_value = histogram.T[i, j]
            ax.text(
                float(xbins[j] + 0.5),
                float(ybins[i] + 0.5),
                f"{bin_value:.2f}",
                color=bin_text_color,
                ha="center",
                va="center",
                fontweight="bold",
            )
    return fig, ax


class DecayModeEvaluator:
    """ Actually we are predicting in the end only 6 (signal) + 1 (bkg) categories, not 16."""

    def __init__(self, predicted: np.array, truth: np.array, output_dir: str, sample: str, algorithm: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.sample = sample
        self.algorithm = algorithm
        self.predicted = predicted
        self.truth = truth
        self.confusion_matrix = metrics.confusion_matrix(self.truth, self.predicted)
        self.normalized_confusion_matrix = metrics.confusion_matrix(self.truth, self.predicted, normalize="true")
        self._decay_mode_name_mapping = {
            0: r"$h^{\pm}$",
            1: r"$h^{\pm}\pi^0$",
            2: r'$h^\pm+\geq2\pi^0$',
            10: r"$h^{\pm}h^{\mp}h^{\pm}$",
            11: r"$h^{\pm}h^{\mp}h^{\pm}+\geq\pi^0$",
            15: "Rare",
        }
        self.categories = list(self._decay_mode_name_mapping.values())
        self.general_metrics, self.class_metrics = self._calculate_performance_metrics()

    def plot_confusion_matrix(self, output_path: str = ""):
        fig, ax = visualize_confusion_matrix(
            histogram=self.normalized_confusion_matrix,
            categories=self.categories,
        )
        if output_path != "":
            plt.savefig(output_path, format="pdf")
            plt.close("all")
        else:
            return fig, ax

    def _calculate_performance_metrics(self):
        class_FPR = (self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)) / self.confusion_matrix.sum()
        class_FNR = (self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)) / self.confusion_matrix.sum()
        class_TPR = (np.diag(self.confusion_matrix)) / self.confusion_matrix.sum()
        class_TNR = (self.confusion_matrix.sum() - (class_FPR + class_FNR + class_TPR)) / self.confusion_matrix.sum()
        class_precision = class_TPR / (class_TPR + class_FPR)
        class_recall = class_TPR / (class_TPR + class_FNR)
        class_F1 = 2 * class_precision * class_recall / (class_precision + class_recall)
        class_accuracy = (class_TPR + class_TNR) / (class_TPR + class_TNR + class_FPR + class_FNR)

        FPR = np.sum(class_FPR) / len(class_FPR)
        FNR = np.sum(class_FNR) / len(class_FNR)
        TPR = np.sum(class_TPR) / len(class_TPR)
        TNR = np.sum(class_TNR) / len(class_TNR)

        precision = TPR / (TPR + FPR)
        recall = TPR / (TPR + FNR)
        F1 = 2 * precision * recall / (precision + recall)
        accuracy = (TPR + TNR) / (TPR + TNR + FPR + FNR)
        class_metrics = {
            "class_FPR": class_FPR,
            "class_FNR": class_FNR,
            "class_TPR": class_TPR,
            "class_TNR": class_TNR,
            "class_precision": class_precision,
            "class_accuracy": class_accuracy,
            "class_recall": class_recall,
            "class_F1": class_F1,
        }
        general_metrics = {
            "FPR": FPR,
            "FNR": FNR,
            "TPR": TPR,
            "TNR": TNR,
            "precision": precision,
            "accuracy": accuracy,
            "recall": recall,
            "F1": F1,
        }
        return general_metrics, class_metrics

    def print_performance(self):
        print("----------------------------------------")
        print("------------ Class metrics -------------")
        print("----------------------------------------")

        print(json.dumps(self.class_metrics, indent=4, cls=NpEncoder))

        print("----------------------------------------")
        print("------------ General metrics -----------")
        print("----------------------------------------")
        print(json.dumps(self.general_metrics, indent=4, cls=NpEncoder))

    def save_performance(self):
        class_metrics_output_path = os.path.join(self.output_dir, f"{self.sample}_{self.algorithm}_class_metrics.json")
        with open(class_metrics_output_path, "wt") as out_file:
            json.dump(self.class_metrics, out_file, indent=4, cls=NpEncoder)

        class_metrics_output_path = os.path.join(self.output_dir, f"{self.sample}_{self.algorithm}_class_metrics.json")
        with open(class_metrics_output_path, "wt") as out_file:
            json.dump(self.class_metrics, out_file, indent=4, cls=NpEncoder)

        confusion_matrix_output_path = os.path.join(self.output_dir,
                                                    f"{self.sample}_{self.algorithm}_confusion_matrix.pdf")
        self.plot_confusion_matrix(output_path=confusion_matrix_output_path)

# Example use:
#     dm_evaluator = DecayModeEvaluator(true_classes, pred_classes, '/path/to/output')
#     dm_evaluator.print_performance()
#     dm_evaluator.plot_confusion_matrix()
#     dm_evaluator.save_performance()
