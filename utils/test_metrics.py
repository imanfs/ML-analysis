import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import matplotlib.pyplot as plt


def cumulative_tp_fp(sorted_image_scores_ious, score_threshold, IOU_threshold=0.75):

    tp = [
        1 if iou > IOU_threshold else 0
        for score, iou in sorted_image_scores_ious
        if score > score_threshold
    ]
    cumulative_tp = [0]
    cumulative_fp = [0]
    for result in tp:
        if result == 1:
            cumulative_tp.append(cumulative_tp[-1] + 1)
            cumulative_fp.append(cumulative_fp[-1])
        else:
            cumulative_fp.append(cumulative_fp[-1] + 1)
            cumulative_tp.append(cumulative_tp[-1])
    return cumulative_tp, cumulative_fp


def calculate_AP(recall, precision):
    pr_boxes = [(0, 1)]  # (precision, recall) at the end of the curve (reverse order)
    for i, recall_val in enumerate(reversed(recall)):
        i = len(recall) - 1 - i
        max_precision = max(precision[i:])
        if max_precision != pr_boxes[-1][0] and recall_val <= pr_boxes[-1][1]:
            pr_boxes.append((max_precision, recall_val))

    pr_boxes.append((1, 0))
    AP_area = 0
    for pr1, pr2 in zip(reversed(pr_boxes), reversed(pr_boxes[:-1])):
        AP_area += pr2[0] * (pr2[1] - pr1[1])
    return AP_area, pr_boxes


def AP_at_IOU(sorted_image_scores_ious, gt_labels, IOU_threshold=0.75):
    tp_acc, fp_acc = cumulative_tp_fp(
        sorted_image_scores_ious, 0, IOU_threshold=IOU_threshold
    )
    recall = [i / gt_labels for i in tp_acc[1:]]
    precision = [tp / (tp + fp) for tp, fp in zip(tp_acc[1:], fp_acc[1:])]
    AP_area, _ = calculate_AP(recall, precision)
    return AP_area


def make_PR_curve(
    sorted_image_scores_ious,
    gt_labels,
    IOU_thresholds,
    save_dir,
    show_pr_curve=False,
    show_ap=True,
    return_AP_values=False,
):
    AP_values = []
    for IOU_threshold in IOU_thresholds:
        tp_acc, fp_acc = cumulative_tp_fp(
            sorted_image_scores_ious, 0, IOU_threshold=IOU_threshold
        )
        recall = [i / gt_labels for i in tp_acc[1:]]
        precision = [tp / (tp + fp) for tp, fp in zip(tp_acc[1:], fp_acc[1:])]

        f1 = torch.Tensor(
            [2 * r * p / (r + p + 1e-16) for r, p in zip(recall, precision)]
        )
        best_idx = torch.argmax(f1)
        print(
            f"IOU: {IOU_threshold:.2f}\n"
            f"\tMax F1: {torch.max(f1):.3f}"
            f"\n\t\t@ Conf value: {sorted_image_scores_ious[best_idx][0]:.3f}"
            f"\n\t\t@ Precision:  {precision[best_idx]:.3f}"
            f"\n\t\t@ Recall:     {recall[best_idx]:.3f}\n"
        )
        AP_area, pr_boxes = calculate_AP(recall, precision)
        AP_values.append(AP_area)
        ax = plt.gca()
        if show_pr_curve:
            plt.plot(recall, precision)
        if show_ap:
            plt.step(
                [x for y, x in pr_boxes],
                [y for y, x in pr_boxes],
                where="post",
                label=f"IOU:{IOU_threshold:.2f}  AP: {AP_area:.3f}",
            )

    plt.title("PR curve and AP @ range of IOUs")
    f_scores = torch.linspace(0.2, 0.8, steps=4)
    for f_score in f_scores:
        x = torch.linspace(0.01, 1.0, steps=100)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate(
            f"f1={f_score:0.1f}", xy=(y[45] + 0.01, 0.9), color="gray", alpha=0.5
        )
    plt.gcf().set_dpi(500)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True)
    plt.xticks(torch.arange(0, 1.1, 0.1))
    plt.yticks(torch.arange(0, 1.1, 0.1))

    plt.savefig(f"{save_dir}/evaluation graphs/PR_curve.png")
    plt.clf()

    if return_AP_values:
        return AP_values


def make_recall_iou_curve(
    sorted_image_scores_ious, total_gt_bboxes, save_dir, conf_threshold=0.5
):

    recall_values = []
    for IOU_threshold in torch.linspace(0.5, 1, steps=50):
        tp = [
            1 if iou > IOU_threshold else 0
            for score, iou in sorted_image_scores_ious
            if score > conf_threshold
        ]

        recall_values.append(sum(tp) / total_gt_bboxes)
    AR = 2 * torch.trapezoid(torch.Tensor(recall_values))
    plt.title(f"Recall - IOU curve @ confidence threshold {conf_threshold}")
    plt.plot(torch.linspace(0.5, 1, steps=50), recall_values, label=f"AR: {AR}")
    ax = plt.gca()
    plt.gcf().set_dpi(500)
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0, 1.05])
    plt.xlabel("IOU")
    plt.ylabel("recall")
    plt.legend()
    plt.grid(True)
    plt.xticks(torch.arange(0.5, 1.05, 0.1))
    plt.yticks(torch.arange(0, 1.05, 0.1))

    plt.savefig(f"{save_dir}/evaluation graphs/recall_iou.png")
    plt.clf()


def f1_graph(sorted_image_scores_ious, total_gt_bboxes, save_dir, IOU_threshold=0.75):
    precision_values = []
    recall_values = []
    f1_values = []

    for conf_threshold in torch.linspace(0, 1, steps=500):
        tp_acc, fp_acc = cumulative_tp_fp(
            sorted_image_scores_ious, conf_threshold, IOU_threshold=IOU_threshold
        )
        # precision = [tp/(tp+fp) for tp, fp in zip(tp_acc[1:], fp_acc[1:])]
        recall = tp_acc[-1] / total_gt_bboxes
        precision = tp_acc[-1] / max((tp_acc[-1] + fp_acc[-1]), 1)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append((2 * recall * precision) / (recall + precision + 1e-16))
        # plt.step([x for y, x in pr_boxes], [y for y, x in pr_boxes], where="post", label=f"IOU:{threshold}  AP: {AP_area:.3f}")
    plt.plot(torch.linspace(0, 1, steps=500), recall_values, label="recall")
    plt.plot(torch.linspace(0, 1, steps=500), f1_values, label="F1")
    plt.plot(torch.linspace(0, 1, steps=500), precision_values, label="precision")
    ax = plt.gca()
    plt.gcf().set_dpi(500)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    plt.xlabel("Conf threshold")
    # plt.xlabel("IOU")
    plt.ylabel("Score")
    plt.title(
        f"f1, precision and recall for different confidence threshold @ IOU={IOU_threshold}"
    )
    plt.legend(loc="best")
    plt.grid(True)
    plt.xticks(torch.arange(0, 1.1, 0.1))
    plt.yticks(torch.arange(0, 1.1, 0.1))

    plt.savefig(f"{save_dir}/evaluation graphs/f1_graph.png")
    plt.clf()
