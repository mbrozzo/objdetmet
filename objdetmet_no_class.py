import math
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import json
import matplotlib
from matplotlib import pyplot as plt


def load_label_no_class(path, warnings=False):
    path = Path(path)
    empty = (
        np.zeros((0, 4), np.float32),
        np.zeros((0), np.float32),
    )
    if not path.exists():
        if warnings:
            print(
                f"WARNING: label file {path} missing, empty labels will be returned for the corresponding image."
            )
        return empty
    try:
        lines = path.read_text().splitlines()
        if not lines:
            return empty
        # else
        boxes = []
        confidences = []
        for line in lines:
            conf = 1
            split_line = line.split()
            if len(split_line) == 5:
                cl, x, y, w, h = split_line
            elif len(split_line) == 6:
                cl, x, y, w, h, conf = split_line
            else:
                raise Exception("Too many or too few values in one line.")
            x = float(x)
            y = float(y)
            w = abs(float(w))
            h = abs(float(h))
            l = x - w / 2
            t = y - h / 2
            r = x + w / 2
            b = y + h / 2
            box = [
                float(l),
                float(t),
                float(r),
                float(b),
            ]
            conf = float(conf)
            boxes.append(box)
            confidences.append(conf)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
    except Exception as e:
        if warnings:
            print(
                f"WARNING: Could not load labels from {path}, empty labels will be returned for the corresponding image."
            )
            print(e)
        return empty
    return boxes, confidences


LOW = np.s_[..., :2]
HIGH = np.s_[..., 2:]


def calculate_iou_matrix(bxs1, bxs2):
    if len(bxs1) == 0 or len(bxs2) == 0:
        return np.zeros((len(bxs1), len(bxs2)), dtype=np.float32)
    intersections = np.maximum(
        0.0,
        np.minimum(bxs1[:, None, 2:], bxs2[None, :, 2:])
        - np.maximum(bxs1[:, None, :2], bxs2[None, :, :2]),
    ).prod(-1)
    areas1 = (bxs1[:, None, 2:] - bxs1[:, None, :2]).prod(-1)
    areas2 = (bxs2[None, :, 2:] - bxs2[None, :, :2]).prod(-1)
    unions = areas1 + areas2 - intersections
    ious = np.where(unions == 0.0, 0.0, intersections / unions)
    return ious


def calculate_confusion_matrix(
    iou_matrix,
    det_confidences,
    conf_thresh,
    iou_thresh,
):
    # Filter det by confidence threshold
    conf_idxs = np.where(det_confidences >= conf_thresh)[0]
    iou_matrix = iou_matrix[:, conf_idxs]
    # Initialize confusion matrix
    cm = np.zeros((2, 2), dtype=np.int32)
    if iou_matrix.shape[0] == 0 and iou_matrix.shape[1] == 0:
        return cm
    if iou_matrix.shape[0] == 0:
        # Only false positives
        cm[1, 0] = iou_matrix.shape[1]
    elif iou_matrix.shape[1] == 0:
        # Only false negatives
        cm[0, 1] = iou_matrix.shape[0]
    else:
        # Create matches matrix
        # Matrix where: element i,j True <=> gt i and detection j boxes match
        # Multiple detections can match one gt, but not vice versa
        box_matches_matrix = np.logical_and(
            iou_matrix > iou_thresh, iou_matrix == iou_matrix.max(0)
        )
        # Update cm with matches (true positives)
        # Only one match per gt is counted when there are multiple
        tp = box_matches_matrix.any(1).astype(np.int32).sum()
        cm[0, 0] = tp
        # Update cm with false positives
        fp = np.logical_not(box_matches_matrix.any(0)).astype(np.int32).sum()
        cm[1, 0] += fp
        # Update cm with false negatives
        # fn = np.logical_not(box_matches_matrix.any(1)).astype(np.int32).sum()
        fn = box_matches_matrix.shape[0] - tp
        cm[0, 1] += fn
    return cm


def calculate_tp_fp_fn(confusion_matrix):
    tp = confusion_matrix[0, 0]
    fp = confusion_matrix[1, 0]
    fn = confusion_matrix[0, 1]
    return tp, fp, fn


FORBIDDEN_CHARS = [
    "\\",
    "/",
    ":",
    "@",
    "$",
    "#",
    "<",
    ">",
    "+",
    "%",
    "!",
    "`",
    "&",
    "*",
    "'",
    '"',
    "|",
    "{",
    "}",
    "?",
    "=",
]


def make_filename_safe(str_, replacer="_"):
    for c in FORBIDDEN_CHARS:
        str_ = str_.replace(c, replacer)
    return str_


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_generate = subparsers.add_parser("generate")
    parser_generate.add_argument(
        "ground_truths_dir",
        type=Path,
    )
    parser_generate.add_argument(
        "detections_dir",
        type=Path,
    )
    parser_generate.add_argument(
        "out_dir",
        type=Path,
    )
    parser_generate.add_argument(
        "--conf-thresh",
        "-c",
        type=float,
        default=0.8,
    )
    parser_generate.add_argument(
        "--iou-thresh",
        "-i",
        type=float,
        default=0.5,
    )
    parser_generate.add_argument(
        "--conf-step",
        type=float,
        default=0.01,
    )
    generate_exclusion = parser_generate.add_mutually_exclusive_group()
    generate_exclusion.add_argument(
        "--no-json",
        action="store_true",
        default=False,
    )
    generate_exclusion.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
    )
    parser_generate.set_defaults(func=generate)

    parser_json2plots = subparsers.add_parser("json2plots")
    parser_json2plots.add_argument(
        "json_file_path",
        type=Path,
    )
    parser_json2plots.add_argument(
        "out_dir",
        type=Path,
    )
    parser_json2plots.set_defaults(func=json2plots)

    args = parser.parse_args()
    args.func(args)


def generate(args):
    ground_truths_dir: Path = args.ground_truths_dir
    detections_dir: Path = args.detections_dir
    out_dir: Path = args.out_dir
    conf_step: float = args.conf_step
    conf_th_default: float = args.conf_thresh
    iou_th_default: float = args.iou_thresh
    gen_json: bool = not args.no_json
    gen_plots: bool = not args.no_plots

    iou_th_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    if iou_th_default not in iou_th_list:
        iou_th_default = min(iou_th_list, key=lambda x: abs(x - iou_th_default))
        print(
            f"WARNING: iou_th_default rounded to closest available one: {iou_th_default}."
        )
    iou_th_default_idx = iou_th_list.index(iou_th_default)

    conf_th_list = []
    conf_th = 0.0
    i = 0
    while conf_th <= 1.0:
        conf_th_list.append(conf_th)
        i += 1
        conf_th = i / (1 / conf_step)  # More precise results
    if conf_th_default not in conf_th_list:
        conf_th_default = min(
            conf_th_list, key=lambda x: abs(x - conf_th_default)
        )
        print("WARNING: conf_th_default rounded to closest available one.")
    conf_th_default_idx = conf_th_list.index(conf_th_default)

    # Define stats arrays
    CM_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list), 2, 2),
        dtype=np.int32,
    )
    TP_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list)), dtype=np.int32
    )
    FP_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list)), dtype=np.int32
    )
    FN_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list)), dtype=np.int32
    )
    P_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list)), dtype=np.float32
    )
    R_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list)), dtype=np.float32
    )
    F1_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list)), dtype=np.float32
    )
    AP_by_iou = np.zeros((len(iou_th_list)), dtype=np.float32)
    mAP_by_iou = np.zeros((len(iou_th_list)), dtype=np.float32)
    mAP_05_095 = np.array(0)

    # Calculate confusion matrix
    print("Calculating confusion matrices")
    for gt_path in tqdm(list(ground_truths_dir.glob("*.txt"))):
        gt_boxes, _ = load_label_no_class(gt_path)
        detection_path = detections_dir / gt_path.name
        det_boxes, det_confidences = load_label_no_class(detection_path)

        # rows: gt, cols: det
        iou_matrix = calculate_iou_matrix(gt_boxes, det_boxes)

        for i, iou_th in enumerate(iou_th_list):
            for j, conf_th in enumerate(conf_th_list):
                CM_by_iou_and_conf[i, j] += calculate_confusion_matrix(
                    iou_matrix,
                    det_confidences,
                    conf_th,
                    iou_th,
                )

    # Calculate other stats
    print("Calculating other metrics")
    for i, iou_th in enumerate(iou_th_list):
        for j, conf_th in enumerate(conf_th_list):
            cm = CM_by_iou_and_conf[i, j]
            tp, fp, fn = calculate_tp_fp_fn(cm)
            # Use float to avoid overflow
            tp = tp.astype(np.float32)
            fp = fp.astype(np.float32)
            fn = fn.astype(np.float32)
            TP_by_iou_and_conf[i, j] = tp
            FP_by_iou_and_conf[i, j] = fp
            FN_by_iou_and_conf[i, j] = fn
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            P_by_iou_and_conf[i, j] = p
            R_by_iou_and_conf[i, j] = r
            f1 = np.nan_to_num(2 * p * r / (p + r))
            F1_by_iou_and_conf[i, j] = f1

    F1_max_idx = F1_by_iou_and_conf[iou_th_default_idx].argmax()
    weighted_F1_max = F1_by_iou_and_conf[iou_th_default_idx][F1_max_idx]
    conf_th_best = conf_th_list[F1_max_idx]

    # Calculate AP
    for i, iou_th in enumerate(iou_th_list):
        p_by_conf = P_by_iou_and_conf[i]
        r_by_conf = R_by_iou_and_conf[i]
        ap = (np.maximum.accumulate(p_by_conf, 0) * r_by_conf).sum(0)
        AP_by_iou[i] = ap
        mAP_by_iou[i] = ap.mean()
    mAP_05_095 = mAP_by_iou.mean()

    metrics = {
        "Confidence Thresholds": conf_th_list,
        "Default IoU Threshold": iou_th_default,
        f"Confusion Matrix @iou={iou_th_default},conf={conf_th_default}": CM_by_iou_and_conf[
            iou_th_default_idx
        ][
            conf_th_default_idx
        ].tolist(),
        f"Confusion Matrix @iou={iou_th_default},conf={conf_th_best}": CM_by_iou_and_conf[
            iou_th_default_idx
        ][
            F1_max_idx
        ].tolist(),
        f"Precision curve @iou={iou_th_default}": P_by_iou_and_conf[
            iou_th_default_idx
        ].tolist(),
        f"Recall curve @iou={iou_th_default}": R_by_iou_and_conf[
            iou_th_default_idx
        ].tolist(),
        f"F1 curve @iou={iou_th_default}": F1_by_iou_and_conf[
            iou_th_default_idx
        ].tolist(),
        f"Maximum F1 @iou={iou_th_default}": float(weighted_F1_max),
        f"Confidence Threshold for Maximum F1 @iou={iou_th_default}": float(
            conf_th_best
        ),
        "mAP@0.5": float(mAP_by_iou[iou_th_list.index(0.5)]),
        "mAP@0.75": float(mAP_by_iou[iou_th_list.index(0.75)]),
        "mAP@[0.5:0.05:0.95]": float(mAP_05_095),
    }
    if gen_json:
        print("Saving JSON")
        out_dir.mkdir(exist_ok=True, parents=True)
        with (out_dir / "metrics.json").open("w") as out_json:
            json.dump(metrics, out_json, indent=4)
    if gen_plots:
        metrics2plots(metrics, out_dir)


def json2plots(args):
    json_file_path = args.json_file_path
    out_dir = args.out_dir
    with json_file_path.open() as in_json:
        metrics = json.load(in_json)
    metrics2plots(metrics, out_dir)


def metrics2plots(metrics, out_dir):
    print("Generating plots")
    conf_th_list = metrics["Confidence Thresholds"]
    iou_th_default = metrics["Default IoU Threshold"]

    for title, data in metrics.items():
        data = np.array(data)
        if title.startswith("Confusion Matrix"):
            # Confusion matrix
            fig = plt.figure(dpi=300)
            ax = plt.gca()
            data = data.astype(np.float32)
            data[-1, -1] = float("nan")
            cmap = matplotlib.cm.get_cmap("Wistia")
            cmap.set_bad(color="lightgrey")
            im = ax.matshow(data, cmap=cmap)
            fig.colorbar(im)
            for (i, j), z in np.ndenumerate(data):
                if math.isnan(z):
                    z = "N/A"
                else:
                    z = str(int(z))
                ax.text(j, i, z, ha="center", va="center")
            plt.title(title)
            plt.xlabel("Predicted class")
            plt.ylabel("True class")
            plt.xticks([0, 1], ["P", "N"])
            plt.yticks([0, 1], ["P", "N"])
            ax.xaxis.set_ticks_position("bottom")
            plt.savefig(
                out_dir / f"{make_filename_safe(title)}.png",
                bbox_inches="tight",
            )
        elif title.startswith("Precision curve") or title.startswith(
            "Recall curve"
        ):
            # Precision and recall to confidence curves
            plt.figure(dpi=300)
            ax = plt.gca()
            plt.plot(conf_th_list, data)
            plt.title(title)
            plt.xlabel("Confidence")
            plt.ylabel(title.split(maxsplit=1)[0])
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xticks(np.linspace(0.0, 1.0, 21), rotation=90)
            plt.savefig(
                out_dir / f"{make_filename_safe(title)}.png",
                bbox_inches="tight",
            )

    plt.figure(dpi=300)
    title = f"F1 score @iou={iou_th_default}"
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("F1 score")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    ax = plt.gca()
    for label, data in metrics.items():
        data = np.array(data)
        if "F1 curve" in label:
            plt.plot(conf_th_list, data)
    plt.xticks(np.linspace(0.0, 1.0, 21), rotation=90)
    plt.savefig(
        out_dir / f"{make_filename_safe(title)}.png", bbox_inches="tight"
    )

    plt.figure(dpi=300)
    title = f"Precision-Recall curve @iou={iou_th_default}"
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    ax = plt.gca()
    p = np.array(metrics[f"Precision curve @iou={iou_th_default}"])
    r = np.array(metrics[f"Recall curve @iou={iou_th_default}"])
    p_acc = np.maximum.accumulate(np.nan_to_num(p))
    # r_acc = np.flip(np.maximum.accumulate(np.flip(r)))
    plt.plot(r, p_acc, color="tab:blue")
    plt.plot(r, p, linestyle=":", color="tab:blue")
    plt.savefig(
        out_dir / f"{make_filename_safe(title)}.png", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
