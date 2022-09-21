from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import json
from matplotlib import pyplot as plt


classes_to_integers = {
    "1": 0,
    "2.1/3": 1,
    "2.2": 2,
    "2.3/6.1": 3,
    "4.1": 4,
    "4.2": 5,
    "4.3": 6,
    "5.1": 7,
    "5.2": 8,
    "6.2": 9,
    "7": 10,
    "7E": 11,
    "8": 12,
    "9": 13,
    "LQ": 14,
    "MP": 15,
}

class_names = list(classes_to_integers.keys())
class_names.sort(key=lambda x: classes_to_integers[x])
class_names_with_bg = class_names + ["bg"]


def load_label(path, warnings=False):
    path = Path(path)
    empty = (
        np.zeros((0), np.int32),
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
        else:
            classes = []
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
                cl = int(float(cl))
                x = float(w)
                y = float(h)
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
                classes.append(cl)
                boxes.append(box)
                confidences.append(conf)
            classes = np.array(classes)
            boxes = np.array(boxes)
            confidences = np.array(confidences)
    except Exception as e:
        if warnings:
            print(
                f"WARNING: Could not load labels from {path}, empty labels will be returned for the corresponding image."
            )
            print(e)
        return empty
    return classes, boxes, confidences


LOW = np.s_[..., :2]
HIGH = np.s_[..., 2:]


def calculate_iou_matrix(bxs1, bxs2):
    if len(bxs1) == 0 or len(bxs2) == 0:
        return np.zeros((len(bxs1), len(bxs2)), dtype="float")
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
    gt_classes,
    det_classes,
    det_confidences,
    conf_thresh,
    iou_thresh,
    n_classes,
):
    # Filter det by confidence threshold
    conf_idxs = np.where(det_confidences >= conf_thresh)[0]
    det_classes = det_classes[conf_idxs]
    iou_matrix = iou_matrix[:, conf_idxs]
    # Initialize confusion matrix
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype="int")
    if iou_matrix.shape[0] == 0 and iou_matrix.shape[1] == 0:
        pass
    if iou_matrix.shape[0] == 0:
        # Only false positives
        for cl in det_classes:
            cm[n_classes, cl] += 1
    elif iou_matrix.shape[1] == 0:
        # Only false negatives
        for cl in gt_classes:
            cm[cl, n_classes] += 1
    else:
        # Create matches matrix
        # Matrix where: element i,j True <=> gt i and detection j boxes match
        box_matches_matrix = np.logical_and(
            np.logical_and(
                iou_matrix > iou_thresh, iou_matrix == iou_matrix.max(0)
            ),
            iou_matrix == iou_matrix.max(1)[:, None],
        )
        box_matches_idxs = np.where(box_matches_matrix)
        # Get classes of each couple of matched gt and box
        gt_classes_matches = gt_classes[box_matches_idxs[0]]
        det_classes_matches = det_classes[box_matches_idxs[1]]
        classes_of_matches = np.stack(
            [gt_classes_matches, det_classes_matches], 1
        )
        # Update cm with matches
        for match in classes_of_matches:
            cm[match[0], match[1]] += 1
        # Update cm with false positives
        fp_idxs = np.where(np.logical_not(box_matches_matrix.any(0)))
        fp_classes = det_classes[fp_idxs]
        for cl in fp_classes:
            cm[n_classes, cl] += 1
        # Update cm with false negatives
        fn_idxs = np.where(np.logical_not(box_matches_matrix.any(1)))
        fn_classes = gt_classes[fn_idxs]
        for cl in fn_classes:
            cm[cl, n_classes] += 1
    return cm


def calculate_tp_fp_fn(confusion_matrix):
    # Arrays of length n_classes with metrics for each class and sum for all classes
    tp = confusion_matrix.diagonal()
    fp = confusion_matrix.sum(0) - tp
    fn = confusion_matrix.sum(1) - tp
    return tp[:-1], fp[:-1], fn[:-1]


def round_to_base(x, base):
    return base * round(float(x) / base)


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
        "--n-classes",
        "-n",
        type=int,
        default=16,
    )
    parser_generate.add_argument(
        "--conf-thresh",
        "-c",
        type=float,
        default=0.5,
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
        default=0.005,
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
    n_classes: int = args.n_classes
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
        conf_th = conf_step * i
    if conf_th_default not in conf_th_list:
        conf_th_default = min(
            conf_th_list, key=lambda x: abs(x - conf_th_default)
        )
        print("WARNING: iou_th_default rounded to closest available one.")
    conf_th_default_idx = conf_th_list.index(conf_th_default)

    # Define stats arrays
    CM_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes + 1, n_classes + 1),
        dtype="int",
    )
    TP_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes), dtype="int"
    )
    FP_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes), dtype="int"
    )
    FN_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes), dtype="int"
    )
    P_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes), dtype="float"
    )
    R_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes), dtype="float"
    )
    F1_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes), dtype="float"
    )
    macro_F1_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list)), dtype="float"
    )
    micro_F1_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list)), dtype="float"
    )
    weighted_F1_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list)), dtype="float"
    )
    AP_by_iou_and_class = np.zeros((len(iou_th_list), n_classes), dtype="float")
    mAP_by_iou = np.zeros((len(iou_th_list)), dtype="float")
    mAP_05_095 = np.array(0)

    # Calculate confusion matrix
    print("Calculating confusion matrices")
    for gt_path in tqdm(list(ground_truths_dir.glob("*.txt"))):
        gt_classes, gt_boxes, _ = load_label(gt_path)
        detection_path = detections_dir / gt_path.name
        det_classes, det_boxes, det_confidences = load_label(detection_path)

        # rows: gt, cols: det
        iou_matrix = calculate_iou_matrix(gt_boxes, det_boxes)

        for i, iou_th in enumerate(iou_th_list):
            for j, conf_th in enumerate(conf_th_list):
                CM_by_iou_and_conf[i, j] += calculate_confusion_matrix(
                    iou_matrix,
                    gt_classes,
                    det_classes,
                    det_confidences,
                    conf_th,
                    iou_th,
                    n_classes,
                )

    print("Calculating other metrics")
    # Calculate other stats
    for i, iou_th in enumerate(iou_th_list):
        for j, conf_th in enumerate(conf_th_list):
            cm = CM_by_iou_and_conf[i, j]
            tp, fp, fn = calculate_tp_fp_fn(cm)
            TP_by_iou_conf_and_class[i, j] = tp
            FP_by_iou_conf_and_class[i, j] = fp
            FN_by_iou_conf_and_class[i, j] = fn
            p = np.nan_to_num(tp / (tp + fp))
            r = np.nan_to_num(tp / (tp + fn))
            P_by_iou_conf_and_class[i, j] = p
            R_by_iou_conf_and_class[i, j] = r
            f1 = np.nan_to_num(2 * p * r / (p + r))
            F1_by_iou_conf_and_class[i, j] = f1
            macro_F1_by_iou_and_conf[i, j] = f1.mean()
            tp_total = tp.sum()
            micro_F1_by_iou_and_conf[i, j] = np.nan_to_num(
                tp_total / (tp_total + 0.5 * (fp.sum() + fn.sum()))
            )
            weights = cm.sum(1)[:-1]
            weighted_F1_by_iou_and_conf[i, j] = np.nan_to_num(
                (f1 * weights).sum() / weights.sum()
            )

    weighted_F1_max_idx = weighted_F1_by_iou_and_conf[
        iou_th_default_idx
    ].argmax()
    weighted_F1_max = weighted_F1_by_iou_and_conf[iou_th_default_idx][
        weighted_F1_max_idx
    ]
    conf_th_best = conf_th_list[weighted_F1_max_idx]

    # Calculate AP
    for i, iou_th in enumerate(iou_th_list):
        p_by_conf_and_class = P_by_iou_conf_and_class[i]
        r_by_conf_and_class = R_by_iou_conf_and_class[i]
        ap = (
            np.maximum.accumulate(p_by_conf_and_class, 0) * r_by_conf_and_class
        ).sum(0)
        AP_by_iou_and_class[i] = ap
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
            weighted_F1_max_idx
        ].tolist(),
        f"Precision curves by class @iou={iou_th_default}": P_by_iou_conf_and_class[
            iou_th_default_idx
        ].tolist(),
        f"Recall curves by class @iou={iou_th_default}": R_by_iou_conf_and_class[
            iou_th_default_idx
        ].tolist(),
        f"F1 curves by class @iou={iou_th_default}": F1_by_iou_conf_and_class[
            iou_th_default_idx
        ].tolist(),
        f"Macro-F1 curve @iou={iou_th_default}": macro_F1_by_iou_and_conf[
            iou_th_default_idx
        ].tolist(),
        f"Micro-F1 curve @iou={iou_th_default}": micro_F1_by_iou_and_conf[
            iou_th_default_idx
        ].tolist(),
        f"Weighted-F1 curve @iou={iou_th_default}": weighted_F1_by_iou_and_conf[
            iou_th_default_idx
        ].tolist(),
        f"Maximum Weighted F1 @iou={iou_th_default}": float(weighted_F1_max),
        f"Confidence Threshold for Maximum Weighted F1 @iou={iou_th_default}": float(
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

    cm = plt.get_cmap("tab20")
    colors = [cm(2.0 * i / 20) for i in range(10)] + [
        cm((2.0 * i + 1) / 20) for i in range(10)
    ]

    for title, data in metrics.items():
        data = np.array(data)
        if title.startswith("Confusion Matrix"):
            # Confusion matrix
            fig = plt.figure(dpi=300)
            ax = plt.gca()
            im = ax.matshow(data)
            fig.colorbar(im)
            for (i, j), z in np.ndenumerate(data):
                ax.text(j, i, str(z), ha="center", va="center")
            plt.title(title)
            plt.xlabel("Predicted class")
            plt.ylabel("True class")
            plt.xticks(
                list(range(len(class_names_with_bg))),
                class_names_with_bg,
                rotation=90,
            )
            plt.yticks(
                list(range(len(class_names_with_bg))), class_names_with_bg
            )
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
            ax.set_prop_cycle(color=colors)
            for i in range(len(class_names)):
                plt.plot(conf_th_list, data[:, i], label=class_names[i])
            plt.title(title)
            plt.xlabel("Confidence")
            plt.ylabel(title.split(maxsplit=1)[0])
            plt.xticks(np.linspace(0.0, 1.0, 21), rotation=90)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.savefig(
                out_dir / f"{make_filename_safe(title)}.png",
                bbox_inches="tight",
            )

    plt.figure(dpi=300)
    title = f"F1 scores by class and averaged @iou={iou_th_default}"
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("F1 score")
    ax = plt.gca()
    ax.set_prop_cycle(color=colors)
    for label, data in metrics.items():
        data = np.array(data)
        label = label.split(" @", maxsplit=1)[0]
        if "F1 curves" in label:
            for i in range(len(data[0])):
                plt.plot(conf_th_list, data[:, i], label=class_names[i])
        elif "F1 curve" in label:
            plt.plot(conf_th_list, data, label=label[:-6], lw=3)
    plt.xticks(np.linspace(0.0, 1.0, 21), rotation=90)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(
        out_dir / f"{make_filename_safe(title)}.png", bbox_inches="tight"
    )

    plt.figure(dpi=300)
    title = f"Precision-Recall curves by class @iou={iou_th_default}"
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    ax = plt.gca()
    ax.set_prop_cycle(
        color=[val for pair in zip(colors, colors) for val in pair]
    )
    for i in range(len(class_names)):
        p = np.array(
            metrics[f"Precision curves by class @iou={iou_th_default}"]
        )[:, i]
        r = np.array(metrics[f"Recall curves by class @iou={iou_th_default}"])[
            :, i
        ]
        p_acc = np.maximum.accumulate(p)
        r_acc = np.flip(np.maximum.accumulate(np.flip(r)))
        plt.plot(r_acc, p_acc, label=class_names[i])
        plt.plot(r, p, linestyle=":")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(
        out_dir / f"{make_filename_safe(title)}.png", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
