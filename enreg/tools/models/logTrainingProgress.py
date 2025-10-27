import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)


def logTrainingProgress(
    tensorboard, idx_epoch, mode, loss, accuracy, class_true, pred_probas, weights
):
    class_pred = np.argmax(pred_probas, axis=-1)
    signal_proba = pred_probas[:, 1]
    assert len(class_true) == len(class_pred) and len(class_true) == len(weights)
    weights_sum = weights.sum()
    true_positive_rate = (
        np.logical_and(class_true == 1, class_pred == 1) * weights
    ).sum() / weights_sum
    false_negative_rate = (
        np.logical_and(class_true == 1, class_pred == 0) * weights
    ).sum() / weights_sum
    true_negative_rate = (
        np.logical_and(class_true == 0, class_pred == 0) * weights
    ).sum() / weights_sum
    false_positive_rate = (
        np.logical_and(class_true == 0, class_pred == 1) * weights
    ).sum() / weights_sum

    precision = true_positive_rate / (true_positive_rate + false_positive_rate)
    recall = true_positive_rate / (true_positive_rate + false_negative_rate)
    F1_score = 2 * precision * recall / (precision + recall)

    print(
        "%s: Avg loss = %1.6f, accuracy = %1.2f%%"
        % (mode.capitalize(), loss, 100 * accuracy)
    )
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

    tensorboard.add_scalar("Loss/%s" % mode, loss, global_step=idx_epoch)
    tensorboard.add_scalar("Accuracy/%s" % mode, 100 * accuracy, global_step=idx_epoch)
    tensorboard.add_pr_curve(
        "ROC_curve/%s" % mode, np.array(class_true), signal_proba, global_step=idx_epoch
    )
    tensorboard.add_scalar(
        "false_positives/%s" % mode, false_positive_rate, global_step=idx_epoch
    )
    tensorboard.add_scalar(
        "false_negatives/%s" % mode, false_negative_rate, global_step=idx_epoch
    )
    tensorboard.add_scalar("precision/%s" % mode, precision, global_step=idx_epoch)
    tensorboard.add_scalar("recall/%s" % mode, recall, global_step=idx_epoch)
    tensorboard.add_scalar("F1_score/%s" % mode, F1_score, global_step=idx_epoch)
    if len(signal_proba[class_true == 1]) > 0:
        tensorboard.add_histogram(
            "tauClassifier_sig/%s" % mode,
            signal_proba[class_true == 1],
            global_step=idx_epoch,
        )
    else:
        raise ValueError("No signal samples. Signal_proba[class_true == 1] is empty.")
    if len(signal_proba[class_true == 0]) > 0:
        tensorboard.add_histogram(
            "tauClassifier_bgr/%s" % mode,
            signal_proba[class_true == 0],
            global_step=idx_epoch,
        )
    else:
        raise ValueError(
            "No backround samples. Signal_proba[class_true == 0] is empty."
        )

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
    ratios,
):
    tensorboard.add_scalar("Loss/%s" % mode, loss, global_step=idx_epoch)
    tensorboard.add_scalar(
        "Mean ratio/%s" % mode, mean_reco_gen_ratio, global_step=idx_epoch
    )
    tensorboard.add_scalar(
        "Median ratio/%s" % mode, median_reco_gen_ratio, global_step=idx_epoch
    )
    tensorboard.add_scalar(
        "Stdev ratio/%s" % mode, stdev_reco_gen_ratio, global_step=idx_epoch
    )
    tensorboard.add_scalar(
        "IQR ratio/%s" % mode, iqr_reco_gen_ratio, global_step=idx_epoch
    )

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
        "loss": float(loss),
    }

    return logging_data


def logTrainingProgress_decaymode(
    tensorboard, idx_epoch, mode, loss, weights, confusion_matrix
):
    tensorboard.add_scalar("Loss/%s" % mode, loss, global_step=idx_epoch)

    confusion_matrix_norm = confusion_matrix / np.sum(confusion_matrix)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_norm,
        display_labels=range(confusion_matrix.shape[0]),
    )
    disp.plot(values_format=".2f", cmap="Blues", text_kw={"fontsize": 6})
    tensorboard.add_figure(
        "confusion_matrix/{}".format(mode), disp.figure_, global_step=idx_epoch
    )

    class_FPR = (
        confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    ) / confusion_matrix.sum()
    class_FNR = (
        confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    ) / confusion_matrix.sum()
    class_TPR = (np.diag(confusion_matrix)) / confusion_matrix.sum()
    class_TNR = (
        confusion_matrix.sum() - (class_FPR + class_FNR + class_TPR)
    ) / confusion_matrix.sum()
    class_precision = class_TPR / (class_TPR + class_FPR)
    class_recall = class_TPR / (class_TPR + class_FNR)
    class_F1 = 2 * class_precision * class_recall / (class_precision + class_recall)
    class_accuracy = (class_TPR + class_TNR) / (
        class_TPR + class_TNR,
        class_FPR,
        class_FNR,
    )

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
