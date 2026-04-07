import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import math
import os


def _ensure_binary_probabilities(pred_probas):
    pred_probas = np.asarray(pred_probas, dtype=np.float64)
    if pred_probas.ndim != 2 or pred_probas.shape[1] != 2:
        raise ValueError(f"Expected class scores/probabilities with shape (N, 2), got {pred_probas.shape}")
    row_sums = pred_probas.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        pred_shifted = pred_probas - np.max(pred_probas, axis=1, keepdims=True)
        exp_scores = np.exp(pred_shifted)
        pred_probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return pred_probas


def logTrainingProgress(
        tensorboard,
        idx_epoch,
        mode,
        loss,
        accuracy,
        class_true,
        pred_probas,
        weights
):
    pred_probas = _ensure_binary_probabilities(pred_probas)
    class_pred = np.argmax(pred_probas, axis=-1)
    signal_proba = pred_probas[:, 1]
    assert len(class_true) == len(class_pred) and len(class_true) == len(weights)
    weights_sum = weights.sum()
    true_positive_rate = (np.logical_and(class_true == 1, class_pred == 1) * weights).sum() / weights_sum
    false_negative_rate = (np.logical_and(class_true == 1, class_pred == 0) * weights).sum() / weights_sum
    true_negative_rate = (np.logical_and(class_true == 0, class_pred == 0) * weights).sum() / weights_sum
    false_positive_rate = (np.logical_and(class_true == 0, class_pred == 1) * weights).sum() / weights_sum

    precision = true_positive_rate / (true_positive_rate + false_positive_rate)
    recall = true_positive_rate / (true_positive_rate + false_negative_rate)
    F1_score = 2 * precision * recall / (precision + recall)
    true_pos_fraction = np.mean(class_true == 1)
    true_neg_fraction = np.mean(class_true == 0)
    pred_pos_fraction = np.mean(class_pred == 1)
    pred_neg_fraction = np.mean(class_pred == 0)
    mean_score_true_pos = np.mean(signal_proba[class_true == 1]) if np.any(class_true == 1) else np.nan
    mean_score_true_neg = np.mean(signal_proba[class_true == 0]) if np.any(class_true == 0) else np.nan

    print("%s: Avg loss = %1.6f, accuracy = %1.2f%%" % (mode.capitalize(), loss, 100 * accuracy))
    print(
        " rates: TP = %1.2f%%, FP = %1.2f%%, TN = %1.2f%%, FN = %1.2f%%"
        " (precision = %1.2f%%, recall = %1.2f%%, F1 score = %1.6f)"
        % (
            100 * true_positive_rate,
            100 * false_positive_rate,
            100 * true_negative_rate,
            100 * false_negative_rate,
            100 * precision,
            100 * recall,
            F1_score,
        )
    )
    print(
        " class balance: true(+1) = %1.2f%%, true(-1) = %1.2f%%, pred(+1) = %1.2f%%, pred(-1) = %1.2f%%"
        " (mean score | true +1 = %1.4f, mean score | true -1 = %1.4f)"
        % (
            100 * true_pos_fraction,
            100 * true_neg_fraction,
            100 * pred_pos_fraction,
            100 * pred_neg_fraction,
            mean_score_true_pos,
            mean_score_true_neg,
        )
    )

    tensorboard.add_scalar("Loss/%s" % mode, loss, global_step=idx_epoch)
    tensorboard.add_scalar("Accuracy/%s" % mode, 100 * accuracy, global_step=idx_epoch)
    tensorboard.add_pr_curve("ROC_curve/%s" % mode, np.array(class_true), signal_proba, global_step=idx_epoch)
    tensorboard.add_scalar("false_positives/%s" % mode, false_positive_rate, global_step=idx_epoch)
    tensorboard.add_scalar("false_negatives/%s" % mode, false_negative_rate, global_step=idx_epoch)
    tensorboard.add_scalar("precision/%s" % mode, precision, global_step=idx_epoch)
    tensorboard.add_scalar("recall/%s" % mode, recall, global_step=idx_epoch)
    tensorboard.add_scalar("F1_score/%s" % mode, F1_score, global_step=idx_epoch)
    if len(signal_proba[class_true == 1]) > 0:
        tensorboard.add_histogram("tauClassifier_sig/%s" % mode, signal_proba[class_true == 1], global_step=idx_epoch)
    else:
        raise ValueError("No signal samples. Signal_proba[class_true == 1] is empty.")
    if len(signal_proba[class_true == 0]) > 0:
        tensorboard.add_histogram("tauClassifier_bgr/%s" % mode, signal_proba[class_true == 0], global_step=idx_epoch)
    else:
        raise ValueError("No backround samples. Signal_proba[class_true == 0] is empty.")

    fpr, tpr, _ = roc_curve(class_true, signal_proba)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(tpr, fpr)
    plt.xlim(0, 1)
    plt.ylim(1e-5, 1)
    plt.yscale("log")
    plt.xlabel("TPR")
    plt.ylabel("FPR")
    tensorboard.add_figure("roc/{}".format(mode), fig, global_step=idx_epoch)
    logging_data = {
        "F1": float(F1_score),
        "recall": float(recall),
        "precision": float(precision),
        "false_positives": float(false_positive_rate),
        "false_negatives": float(false_negative_rate),
        "true_positives": float(true_positive_rate),
        "true_negatives": float(true_negative_rate),
        "accuracy": float(accuracy),
        "loss": float(loss),
        # "AUC":
    }

    return logging_data


def _safe_ratio(numerator, denominator):
    if denominator <= 0:
        return np.nan
    return numerator / denominator


def _compute_charge_curves(class_true, signal_proba, thresholds=None):
    class_true = np.asarray(class_true, dtype=np.int64)
    signal_proba = np.asarray(signal_proba, dtype=np.float64)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)

    pos_eff = []
    neg_eff = []
    pos_misid = []
    neg_misid = []

    true_pos = class_true == 1
    true_neg = class_true == 0
    n_pos = np.count_nonzero(true_pos)
    n_neg = np.count_nonzero(true_neg)

    for threshold in thresholds:
        pred_pos = signal_proba >= threshold
        pred_neg = ~pred_pos

        pos_eff.append(_safe_ratio(np.count_nonzero(pred_pos & true_pos), n_pos))
        neg_eff.append(_safe_ratio(np.count_nonzero(pred_neg & true_neg), n_neg))
        pos_misid.append(_safe_ratio(np.count_nonzero(pred_pos & true_neg), n_neg))
        neg_misid.append(_safe_ratio(np.count_nonzero(pred_neg & true_pos), n_pos))

    return {
        "thresholds": np.asarray(thresholds, dtype=np.float64),
        "pos_eff": np.asarray(pos_eff, dtype=np.float64),
        "neg_eff": np.asarray(neg_eff, dtype=np.float64),
        "pos_misid": np.asarray(pos_misid, dtype=np.float64),
        "neg_misid": np.asarray(neg_misid, dtype=np.float64),
    }


def _make_charge_fakerate_vs_efficiency_figure(class_true, signal_proba):
    curves = _compute_charge_curves(class_true, signal_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(curves["pos_eff"], curves["pos_misid"], linestyle="None", marker="s", markersize=4, color="red", label=r"$\tau^{+}$")
    ax.plot(curves["neg_eff"], curves["neg_misid"], linestyle="None", marker="^", markersize=4, color="blue", label=r"$\tau^{-}$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-5, 1.0)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\epsilon_{\tau}$")
    ax.set_ylabel(r"$P_{\mathrm{misid}}$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, curves


def _make_charge_efficiency_vs_kinematics_figure(gen_values, class_true, signal_proba, bins, xlabel):
    gen_values = np.asarray(gen_values, dtype=np.float64)
    class_true = np.asarray(class_true, dtype=np.int64)
    signal_proba = np.asarray(signal_proba, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)

    pred_charge_signed = np.where(signal_proba >= 0.5, 1, -1)
    true_charge_signed = np.where(class_true == 1, 1, -1)

    centers = 0.5 * (bins[:-1] + bins[1:])
    pos_eff = []
    neg_eff = []

    for low, high in zip(bins[:-1], bins[1:]):
        in_bin = (gen_values >= low) & (gen_values < high)
        pos_mask = in_bin & (true_charge_signed == 1)
        neg_mask = in_bin & (true_charge_signed == -1)
        pos_eff.append(_safe_ratio(np.count_nonzero(pos_mask & (pred_charge_signed == 1)), np.count_nonzero(pos_mask)))
        neg_eff.append(_safe_ratio(np.count_nonzero(neg_mask & (pred_charge_signed == -1)), np.count_nonzero(neg_mask)))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(centers, pos_eff, marker="s", color="red", label=r"$\tau^{+}$")
    ax.plot(centers, neg_eff, marker="^", color="blue", label=r"$\tau^{-}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Charge-tagging efficiency")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, np.asarray(pos_eff, dtype=np.float64), np.asarray(neg_eff, dtype=np.float64), centers


def logTrainingProgress_charge(
        tensorboard,
        idx_epoch,
        mode,
        loss,
        accuracy,
        class_true,
        pred_probas,
        weights,
        gen_tau_pt,
        gen_tau_eta,
        pt_binning,
        eta_binning,
        output_dir=None,
):
    pred_probas = _ensure_binary_probabilities(pred_probas)
    logging_data = logTrainingProgress(
        tensorboard=tensorboard,
        idx_epoch=idx_epoch,
        mode=mode,
        loss=loss,
        accuracy=accuracy,
        class_true=class_true,
        pred_probas=pred_probas,
        weights=weights,
    )

    signal_proba = pred_probas[:, 1]
    gen_tau_pt = np.asarray(gen_tau_pt, dtype=np.float64)
    gen_tau_eta = np.asarray(gen_tau_eta, dtype=np.float64)

    charge_fakerate_fig, curves = _make_charge_fakerate_vs_efficiency_figure(class_true, signal_proba)
    eff_vs_pt_fig, pos_eff_vs_pt, neg_eff_vs_pt, pt_centers = _make_charge_efficiency_vs_kinematics_figure(
        gen_values=gen_tau_pt,
        class_true=class_true,
        signal_proba=signal_proba,
        bins=pt_binning,
        xlabel=r"$p_{T}^{gen\ \tau}$ [GeV]",
    )
    eff_vs_eta_fig, pos_eff_vs_eta, neg_eff_vs_eta, eta_centers = _make_charge_efficiency_vs_kinematics_figure(
        gen_values=gen_tau_eta,
        class_true=class_true,
        signal_proba=signal_proba,
        bins=eta_binning,
        xlabel=r"$\eta^{gen\ \tau}$",
    )

    figures = {
        "charge_fakerate_vs_efficiency": charge_fakerate_fig,
        "charge_efficiency_vs_gen_tau_pt": eff_vs_pt_fig,
        "charge_efficiency_vs_gen_tau_eta": eff_vs_eta_fig,
    }
    for figure_name, fig in figures.items():
        tensorboard.add_figure(f"{figure_name}/{mode}", fig, idx_epoch)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, f"epoch_{idx_epoch:04d}_{figure_name}_{mode}.png"), bbox_inches="tight")
        plt.close(fig)

    return logging_data


def logTrainingProgress_regression(
        tensorboard,
        idx_epoch,
        mode,
        loss,
        mean_reco_gen_ratio,
        median_reco_gen_ratio,
        stdev_reco_gen_ratio,
        iqr_reco_gen_ratio,
        weights,
        ratios
):
    tensorboard.add_scalar("Loss/%s" % mode, loss, global_step=idx_epoch)
    tensorboard.add_scalar("Mean ratio/%s" % mode, mean_reco_gen_ratio, global_step=idx_epoch)
    tensorboard.add_scalar("Median ratio/%s" % mode, median_reco_gen_ratio, global_step=idx_epoch)
    tensorboard.add_scalar("Stdev ratio/%s" % mode, stdev_reco_gen_ratio, global_step=idx_epoch)
    tensorboard.add_scalar("IQR ratio/%s" % mode, iqr_reco_gen_ratio, global_step=idx_epoch)

    fig = plt.figure(figsize=(5, 5))
    plt.hist(ratios, bins=np.linspace(0.5, 1.5, 100), histtype="step", lw=2)
    plt.xlabel("reco pt / gen tau pt")
    plt.ylabel("number of jets / bin")
    tensorboard.add_figure("ratio/{}".format(mode), fig, global_step=idx_epoch)

    logging_data = {
        "IQR": float(iqr_reco_gen_ratio),
        "median": float(median_reco_gen_ratio),
        "stdev": float(stdev_reco_gen_ratio),
        "mean": float(mean_reco_gen_ratio),
        "loss": float(loss)
    }

    return logging_data


def _compute_ratio_response_resolution(values):
    if len(values) == 0:
        return np.nan, np.nan
    q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    if np.isclose(q50, 0.0):
        resolution = np.nan
    else:
        resolution = (q75 - q25) / q50
    return q50, resolution


def _compute_residual_response_resolution(values):
    if len(values) == 0:
        return np.nan, np.nan
    q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    return q50, q75 - q25


def _sanitize(values, finite_range=None):
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    if finite_range is not None:
        low, high = finite_range
        mask &= values >= low
        mask &= values <= high
    return values[mask]


def _sanitize_pair(x_values, y_values, x_range=None, y_range=None):
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if x_range is not None:
        low, high = x_range
        mask &= x_values >= low
        mask &= x_values <= high
    if y_range is not None:
        low, high = y_range
        mask &= y_values >= low
        mask &= y_values <= high
    return x_values[mask], y_values[mask]


def _make_distribution_figure(values, bins, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(values, bins=bins, histtype="step", lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return fig


def _make_binned_pt_response_figure(gen_tau_pt, pt_response, pt_bins):
    n_bins = len(pt_bins) - 1
    ncols = min(4, n_bins)
    nrows = int(math.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    response_hist_bins = np.linspace(0.5, 1.5, 101)

    for idx in range(n_bins):
        ax = axes[idx]
        low = pt_bins[idx]
        high = pt_bins[idx + 1]
        in_bin = (gen_tau_pt >= low) & (gen_tau_pt < high)
        bin_values = pt_response[in_bin]
        ax.set_title(rf"$p_{{T,true}} \in [{low}, {high}]$ GeV", fontsize=11)
        ax.set_xlim(0.5, 1.5)
        ax.set_xlabel("q")
        if len(bin_values) > 0:
            ax.hist(bin_values, bins=response_hist_bins, histtype="step", lw=1.5, color="C0", density=True)
            _, resolution = _compute_ratio_response_resolution(bin_values)
            ax.text(0.04, 0.94, f"IQR/q50 = {resolution:.3f}", transform=ax.transAxes, va="top", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No entries", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.grid(alpha=0.2)

    for idx in range(n_bins, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    return fig


def _make_binned_metric_figure(
    gen_tau_pt,
    metric_values,
    pt_bins,
    hist_bins,
    xlabel,
    title_label,
    annotation_label,
    metric_summary_fn,
):
    n_bins = len(pt_bins) - 1
    ncols = min(4, n_bins)
    nrows = int(math.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    hist_bins = np.asarray(hist_bins, dtype=np.float64)
    hist_low = hist_bins[0]
    hist_high = hist_bins[-1]

    for idx in range(n_bins):
        ax = axes[idx]
        low = pt_bins[idx]
        high = pt_bins[idx + 1]
        in_bin = (gen_tau_pt >= low) & (gen_tau_pt < high)
        bin_values = metric_values[in_bin]
        in_hist_range = (bin_values >= hist_low) & (bin_values <= hist_high)
        plot_values = bin_values[in_hist_range]

        ax.set_title(rf"$p_{{T,true}} \in [{low}, {high}]$ GeV", fontsize=11)
        ax.set_xlim(hist_low, hist_high)
        ax.set_xlabel(xlabel)
        if len(plot_values) > 0:
            ax.hist(plot_values, bins=hist_bins, histtype="step", lw=1.5, color="C0", density=True)
            ax.text(
                0.04,
                0.94,
                f"{annotation_label} = {metric_summary_fn(bin_values):.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        else:
            ax.text(0.5, 0.5, "No entries", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.grid(alpha=0.2)

    for idx in range(n_bins, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title_label, fontsize=13)
    fig.tight_layout()
    return fig


def _make_response_resolution_vs_pt_figure(gen_tau_pt, pt_response, pt_bins, ylabel, metric_kind, ylim=None):
    centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    values = []
    for low, high in zip(pt_bins[:-1], pt_bins[1:]):
        in_bin = (gen_tau_pt >= low) & (gen_tau_pt < high)
        bin_values = pt_response[in_bin]
        if len(bin_values) == 0:
            values.append(np.nan)
        elif metric_kind == "response":
            response, _ = _compute_ratio_response_resolution(bin_values)
            values.append(response)
        else:
            _, resolution = _compute_ratio_response_resolution(bin_values)
            values.append(resolution)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(centers, values, marker="X", lw=2)
    ax.set_xlabel(r"$p_T^{gen}$ [GeV]")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, np.asarray(values, dtype=np.float64), centers


def _make_stat_vs_pt_figure(gen_tau_pt, metric_values, pt_bins, ylabel, stat_fn, ylim=None):
    centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    values = []
    for low, high in zip(pt_bins[:-1], pt_bins[1:]):
        in_bin = (gen_tau_pt >= low) & (gen_tau_pt < high)
        bin_values = metric_values[in_bin]
        values.append(np.nan if len(bin_values) == 0 else stat_fn(bin_values))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(centers, values, marker="X", lw=2)
    ax.set_xlabel(r"$p_T^{gen}$ [GeV]")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, np.asarray(values, dtype=np.float64), centers


def logTrainingProgress_p4(
        tensorboard,
        idx_epoch,
        mode,
        loss,
        component_losses,
        pt_responses,
        mass_responses,
        eta_residuals,
        phi_residuals,
        gen_tau_pt,
        pt_binning,
        output_dir=None,
):
    tensorboard.add_scalar("Loss/%s" % mode, loss, global_step=idx_epoch)
    for component_name, component_loss in component_losses.items():
        tensorboard.add_scalar("Loss_%s/%s" % (component_name, mode), component_loss, global_step=idx_epoch)
    gen_tau_pt_all = np.asarray(gen_tau_pt, dtype=np.float64)

    regression_metrics = {}
    ratio_metric_inputs = {
        "pt": _sanitize(pt_responses, finite_range=(0.0, 5.0)),
        "mass": _sanitize(mass_responses, finite_range=(0.0, 5.0)),
    }
    residual_metric_inputs = {
        "eta": _sanitize(eta_residuals),
        "phi": _sanitize(phi_residuals, finite_range=(-np.pi, np.pi)),
    }

    for component_name, values in ratio_metric_inputs.items():
        response, resolution = _compute_ratio_response_resolution(values)
        regression_metrics[f"{component_name}_response"] = float(response)
        regression_metrics[f"{component_name}_resolution"] = float(resolution)
    for component_name, values in residual_metric_inputs.items():
        response, resolution = _compute_residual_response_resolution(values)
        regression_metrics[f"{component_name}_response"] = float(response)
        regression_metrics[f"{component_name}_resolution"] = float(resolution)

    for metric_name, metric_value in regression_metrics.items():
        tensorboard.add_scalar("%s/%s" % (metric_name, mode), metric_value, global_step=idx_epoch)

    figures = {
        "pt_response": _make_distribution_figure(
            ratio_metric_inputs["pt"],
            bins=np.linspace(0.5, 1.5, 100),
            xlabel=r"$p_{T}^{pred} / p_{T}^{gen}$",
            ylabel="Jets / bin",
            title="pT response",
        ),
        "mass_response": _make_distribution_figure(
            ratio_metric_inputs["mass"],
            bins=np.linspace(0.0, 2.0, 100),
            xlabel=r"$m^{pred} / m^{gen}$",
            ylabel="Jets / bin",
            title="Mass response",
        ),
        "eta_residual": _make_distribution_figure(
            residual_metric_inputs["eta"],
            bins=100,
            xlabel=r"$\eta^{pred} - \eta^{gen}$",
            ylabel="Jets / bin",
            title="Eta residual",
        ),
        "phi_residual": _make_distribution_figure(
            residual_metric_inputs["phi"],
            bins=np.linspace(-np.pi, np.pi, 100),
            xlabel=r"$\phi^{pred} - \phi^{gen}$",
            ylabel="Jets / bin",
            title="Wrapped phi residual",
        ),
    }

    pt_bins = np.asarray(pt_binning, dtype=np.float64)
    gen_tau_pt, pt_for_binning = _sanitize_pair(
        gen_tau_pt_all,
        pt_responses,
        x_range=(pt_bins[0], pt_bins[-1]),
        y_range=(0.0, 5.0),
    )

    figures["pt_resolution_bins"] = _make_binned_pt_response_figure(gen_tau_pt, pt_for_binning, pt_bins)
    response_fig, binned_responses, pt_centers = _make_response_resolution_vs_pt_figure(
        gen_tau_pt=gen_tau_pt,
        pt_response=pt_for_binning,
        pt_bins=pt_bins,
        ylabel=r"$p_T$ scale $(q_{50})$",
        metric_kind="response",
        ylim=(0.9, 1.1),
    )
    figures["pt_response_vs_gen_pt"] = response_fig
    resolution_fig, binned_resolutions, _ = _make_response_resolution_vs_pt_figure(
        gen_tau_pt=gen_tau_pt,
        pt_response=pt_for_binning,
        pt_bins=pt_bins,
        ylabel=r"$p_T$ resol. $(q_{75}-q_{25})/q_{50}$",
        metric_kind="resolution",
        ylim=(0.0, 0.2),
    )
    figures["pt_resolution_vs_gen_pt"] = resolution_fig

    gen_tau_pt_mass, mass_for_binning = _sanitize_pair(
        gen_tau_pt_all,
        mass_responses,
        x_range=(pt_bins[0], pt_bins[-1]),
        y_range=(0.0, 5.0),
    )
    figures["mass_resolution_bins"] = _make_binned_metric_figure(
        gen_tau_pt=gen_tau_pt_mass,
        metric_values=mass_for_binning,
        pt_bins=pt_bins,
        hist_bins=np.linspace(0.0, 2.0, 100),
        xlabel="q",
        title_label="Mass response by gen pT bin",
        annotation_label="IQR/q50",
        metric_summary_fn=lambda values: _compute_ratio_response_resolution(values)[1],
    )
    mass_resolution_fig, binned_mass_resolutions, _ = _make_stat_vs_pt_figure(
        gen_tau_pt=gen_tau_pt_mass,
        metric_values=mass_for_binning,
        pt_bins=pt_bins,
        ylabel=r"Mass resol. $(q_{75}-q_{25})/q_{50}$",
        stat_fn=lambda values: _compute_ratio_response_resolution(values)[1],
        ylim=(0.0, 2.0),
    )
    figures["mass_resolution_vs_gen_pt"] = mass_resolution_fig

    gen_tau_pt_eta, eta_for_binning = _sanitize_pair(
        gen_tau_pt_all,
        eta_residuals,
        x_range=(pt_bins[0], pt_bins[-1]),
    )
    figures["eta_resolution_bins"] = _make_binned_metric_figure(
        gen_tau_pt=gen_tau_pt_eta,
        metric_values=eta_for_binning,
        pt_bins=pt_bins,
        hist_bins=np.linspace(-0.08, 0.08, 100),
        xlabel=r"$\eta^{pred} - \eta^{gen}$",
        title_label="Eta residual by gen pT bin",
        annotation_label="IQR",
        metric_summary_fn=lambda values: _compute_residual_response_resolution(values)[1],
    )
    eta_resolution_fig, binned_eta_resolutions, _ = _make_stat_vs_pt_figure(
        gen_tau_pt=gen_tau_pt_eta,
        metric_values=eta_for_binning,
        pt_bins=pt_bins,
        ylabel=r"$\eta$ residual IQR",
        stat_fn=lambda values: _compute_residual_response_resolution(values)[1],
    )
    figures["eta_resolution_vs_gen_pt"] = eta_resolution_fig

    gen_tau_pt_phi, phi_for_binning = _sanitize_pair(
        gen_tau_pt_all,
        phi_residuals,
        x_range=(pt_bins[0], pt_bins[-1]),
        y_range=(-np.pi, np.pi),
    )
    figures["phi_resolution_bins"] = _make_binned_metric_figure(
        gen_tau_pt=gen_tau_pt_phi,
        metric_values=phi_for_binning,
        pt_bins=pt_bins,
        hist_bins=np.linspace(-0.08, 0.08, 100),
        xlabel=r"$\phi^{pred} - \phi^{gen}$",
        title_label="Wrapped phi residual by gen pT bin",
        annotation_label="IQR",
        metric_summary_fn=lambda values: _compute_residual_response_resolution(values)[1],
    )
    phi_resolution_fig, binned_phi_resolutions, _ = _make_stat_vs_pt_figure(
        gen_tau_pt=gen_tau_pt_phi,
        metric_values=phi_for_binning,
        pt_bins=pt_bins,
        ylabel=r"$\phi$ residual IQR",
        stat_fn=lambda values: _compute_residual_response_resolution(values)[1],
    )
    figures["phi_resolution_vs_gen_pt"] = phi_resolution_fig

    for figure_name, fig in figures.items():
        tensorboard.add_figure(f"{figure_name}/{mode}", fig, idx_epoch)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, f"epoch_{idx_epoch:04d}_{figure_name}_{mode}.png"), bbox_inches="tight")
        plt.close(fig)

    logging_data = {
        "loss": float(loss),
    }
    logging_data.update({f"loss_{component_name}": float(component_loss) for component_name, component_loss in component_losses.items()})
    logging_data.update({metric_name: float(metric_value) for metric_name, metric_value in regression_metrics.items()})
    logging_data["pt_response_binned"] = binned_responses
    logging_data["pt_resolution_binned"] = binned_resolutions
    logging_data["mass_resolution_binned"] = binned_mass_resolutions
    logging_data["eta_resolution_binned"] = binned_eta_resolutions
    logging_data["phi_resolution_binned"] = binned_phi_resolutions
    logging_data["pt_bin_centers"] = pt_centers

    return logging_data



def logTrainingProgress_decaymode(
        tensorboard,
        idx_epoch,
        mode,
        loss,
        weights,
        confusion_matrix
):
    tensorboard.add_scalar("Loss/%s" % mode, loss, global_step=idx_epoch)

    confusion_matrix_norm = confusion_matrix / np.sum(confusion_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_norm,
                                  display_labels=range(confusion_matrix.shape[0]))
    disp.plot(values_format=".2f", cmap="Blues", text_kw={"fontsize": 6})
    tensorboard.add_figure("confusion_matrix/{}".format(mode), disp.figure_, global_step=idx_epoch)

    class_FPR = (confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)) / confusion_matrix.sum()
    class_FNR = (confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)) / confusion_matrix.sum()
    class_TPR = (np.diag(confusion_matrix)) / confusion_matrix.sum()
    class_TNR = (confusion_matrix.sum() - (class_FPR + class_FNR + class_TPR)) / confusion_matrix.sum()
    class_precision = class_TPR / (class_TPR + class_FPR)
    class_recall = class_TPR / (class_TPR + class_FNR)
    class_F1 = 2 * class_precision * class_recall / (class_precision + class_recall)
    class_accuracy = (class_TPR + class_TNR) / (class_TPR + class_TNR, class_FPR, class_FNR)

    FPR = np.sum(class_FPR) / len(class_FPR)
    FNR = np.sum(class_FNR) / len(class_FNR)
    TPR = np.sum(class_TPR) / len(class_TPR)
    TNR = np.sum(class_TNR) / len(class_TNR)

    # This here is
    # TODO: If reporting macro-average then in cases of 3 or more classes, std should also be reported.
    precision = TPR / (TPR + FPR)
    recall = TPR / (TPR + FNR)
    F1 = 2 * precision * recall / (precision + recall)
    accuracy = (TPR + TNR) / (TPR + TNR + FPR + FNR)

    logging_data = {
        "confusion_matrix": confusion_matrix,
        "loss": loss,

        # "class_AUC":
        "class_FPR": class_FPR,
        "class_FNR": class_FNR,
        "class_TPR": class_TPR,
        "class_TNR": class_TNR,
        "class_precision": class_precision,
        "class_recall": class_recall,
        "class_F1": class_F1,

        # "AUC":
        "FPR": FPR,
        "FNR": FNR,
        "TPR": TPR,
        "TNR": TNR,
        "precision": precision,
        "accuracy": accuracy,
        "recall": recall,
        "F1": F1,
    }
    return logging_data
