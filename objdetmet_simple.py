from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import json
import matplotlib
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


def load_and_simplify_label(path, n_classes, warnings=False):
    path = Path(path)
    empty = np.zeros((n_classes), np.int32)
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
        classes = np.array([int(float(line.split()[0])) for line in lines])
        classes = np.unique(classes)
        gt = np.zeros((n_classes), np.int32)
        gt[classes] = 1
    except Exception as e:
        if warnings:
            print(
                f"WARNING: Could not load labels from {path}, empty labels will be returned for the corresponding image."
            )
            print(e)
        return empty
    return gt


def load_simple_label(path, n_classes, warnings=False):
    path = Path(path)
    empty = np.zeros((n_classes), np.int32)
    if not path.exists():
        if warnings:
            print(
                f"WARNING: label file {path} missing, empty labels will be returned for the corresponding image."
            )
        return empty
    try:
        lines = path.read_text().splitlines()
        if not lines:
            print(
                f"WARNING: label file {path} contains no labels, empty labels will be returned for the corresponding image."
            )
            return empty
        # else
        if len(lines) > 1:
            print(
                f"WARNING: label file {path} contains too many lines, empty labels will be returned for the corresponding image."
            )
            return empty
        # else
        values = lines[0].split()
        values = np.array([float(v) for v in values])
    except Exception as e:
        if warnings:
            print(
                f"WARNING: Could not load labels from {path}, empty labels will be returned for the corresponding image."
            )
            print(e)
        return empty
    return values


def calculate_confusion_matrices(gt, det, conf_thresh):
    n_classes = len(gt)
    # Filter det by confidence threshold
    det = np.where(det >= conf_thresh, 1, 0)
    # Initialize confusion matrices
    cm = np.zeros((n_classes, 2, 2), dtype=np.int32)
    # First dimension: actual; second dimension: predicted
    # [[TP, FN],
    #  [FP, TN]]
    for cl in range(n_classes):
        cm[cl, 1 - gt[cl], 1 - det[cl]] = 1
    return cm


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
        default=0.8,
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
    n_classes: int = args.n_classes
    gen_json: bool = not args.no_json
    gen_plots: bool = not args.no_plots

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
    CM_by_conf_and_class = np.zeros(
        (len(conf_th_list), n_classes, 2, 2), dtype=np.int32
    )
    TP_by_conf_and_class = np.zeros(
        (len(conf_th_list), n_classes), dtype=np.int32
    )
    TN_by_conf_and_class = np.zeros(
        (len(conf_th_list), n_classes), dtype=np.int32
    )
    FP_by_conf_and_class = np.zeros(
        (len(conf_th_list), n_classes), dtype=np.int32
    )
    FN_by_conf_and_class = np.zeros(
        (len(conf_th_list), n_classes), dtype=np.int32
    )
    P_by_conf_and_class = np.zeros(
        (len(conf_th_list), n_classes), dtype=np.float32
    )
    R_by_conf_and_class = np.zeros(
        (len(conf_th_list), n_classes), dtype=np.float32
    )
    F1_by_conf_and_class = np.zeros(
        (len(conf_th_list), n_classes), dtype=np.float32
    )
    macro_F1_by_conf = np.zeros((len(conf_th_list)), dtype=np.float32)
    micro_F1_by_conf = np.zeros((len(conf_th_list)), dtype=np.float32)
    weighted_F1_by_conf = np.zeros((len(conf_th_list)), dtype=np.float32)
    MCC_by_conf_and_class = np.zeros(
        (len(conf_th_list), n_classes), dtype=np.float32
    )
    macro_MCC_by_conf = np.zeros((len(conf_th_list)), dtype=np.float32)
    micro_MCC_by_conf = np.zeros((len(conf_th_list)), dtype=np.float32)
    weighted_MCC_by_conf = np.zeros((len(conf_th_list)), dtype=np.float32)
    AP_by_class = np.zeros(n_classes, dtype=np.float32)
    mAP = np.array(0)

    # Calculate confusion matrix
    print("Calculating confusion matrices")
    for gt_path in tqdm(list(ground_truths_dir.glob("*.txt"))):
        gt = load_and_simplify_label(gt_path, n_classes, warnings=True)
        detection_path = detections_dir / gt_path.name
        det = load_simple_label(detection_path, n_classes, warnings=True)

        for i, conf_th in enumerate(conf_th_list):
            CM_by_conf_and_class[i] += calculate_confusion_matrices(
                gt, det, conf_th
            )

    # Calculate other stats
    print("Calculating other metrics")
    for i, conf_th in enumerate(conf_th_list):
        cm = CM_by_conf_and_class[i]
        # Use float to avoid overflow
        tp = cm[:, 0, 0].astype(np.float32)
        tn = cm[:, 1, 1].astype(np.float32)
        fp = cm[:, 1, 0].astype(np.float32)
        fn = cm[:, 0, 1].astype(np.float32)
        TP_by_conf_and_class[i] = tp
        TN_by_conf_and_class[i] = tn
        FP_by_conf_and_class[i] = fp
        FN_by_conf_and_class[i] = fn
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        P_by_conf_and_class[i] = p
        R_by_conf_and_class[i] = r
        f1 = np.nan_to_num(2 * p * r / (p + r))
        F1_by_conf_and_class[i] = f1
        macro_F1_by_conf[i] = f1.mean()
        tp_total = tp.sum()
        tn_total = tn.sum()
        fp_total = fp.sum()
        fn_total = fn.sum()
        micro_F1_by_conf[i] = np.nan_to_num(
            tp_total / (tp_total + 0.5 * (fp_total + fn_total))
        )
        weights = cm[:, 0, :].sum(-1)
        weighted_F1_by_conf[i] = np.nan_to_num(
            (f1 * weights).sum() / weights.sum()
        )
        mcc = np.nan_to_num(
            (tp * tn - fp * fn)
            / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
            nan=0,
        )
        MCC_by_conf_and_class[i] = mcc
        macro_MCC_by_conf[i] = mcc.mean()
        micro_MCC_by_conf[i] = np.nan_to_num(
            tp_total * tn_total - fp_total * fn_total
        ) / np.sqrt(
            (tp_total + fp_total)
            * (tp_total + fn_total)
            * (tn_total + fp_total)
            * (tn_total + fn_total)
        )
        weighted_MCC_by_conf[i] = np.nan_to_num(
            (mcc * weights).sum() / weights.sum()
        )

    weighted_F1_max_idx = weighted_F1_by_conf.argmax()
    weighted_F1_max = weighted_F1_by_conf[weighted_F1_max_idx]
    conf_th_best = conf_th_list[weighted_F1_max_idx]

    # Calculate AP
    print("Calculating AP")
    r_values = []
    r_step = 0.01
    r = 1.0
    i = int(1 / r_step)
    while r >= 0.0:
        r_values.append(r)
        i -= 1
        r = i / (1 / r_step)  # More precise results

    # Interpolate p
    P_interp_by_conf_and_class = np.maximum.accumulate(
        np.nan_to_num(P_by_conf_and_class), axis=1
    )

    with tqdm(len(class_names) * len(r_values)) as pbar:
        # 101-point interpolation (COCO)
        for cl in range(len(class_names)):
            p_interp_by_r = np.zeros_like(r_values)
            for j, r in enumerate(r_values):
                try:
                    p_interp_by_r[j] = P_interp_by_conf_and_class[
                        np.where(R_by_conf_and_class[:, cl] >= r)[0][-1],
                        cl,
                    ]
                except:
                    pass
                pbar.update()
            AP_by_class[cl] = p_interp_by_r.mean()
        mAP = AP_by_class.mean()

    metrics = {
        "Confidence Thresholds": conf_th_list,
        f"Confusion Matrices by class @conf={conf_th_default}": CM_by_conf_and_class[
            conf_th_default_idx
        ].tolist(),
        f"Confusion Matrices by class @conf={conf_th_best}": CM_by_conf_and_class[
            weighted_F1_max_idx
        ].tolist(),
        "Precision curves by class": P_by_conf_and_class.tolist(),
        "Recall curves by class": R_by_conf_and_class.tolist(),
        "F1 curves by class": F1_by_conf_and_class.tolist(),
        "Macro-F1 curve": macro_F1_by_conf.tolist(),
        "Micro-F1 curve": micro_F1_by_conf.tolist(),
        "Weighted-F1 curve": weighted_F1_by_conf.tolist(),
        "Maximum Weighted F1": float(weighted_F1_max),
        "Confidence Threshold for Maximum Weighted F1": float(conf_th_best),
        "MCC curves by class": MCC_by_conf_and_class.tolist(),
        "Macro-MCC curve": macro_MCC_by_conf.tolist(),
        "Micro-MCC curve": micro_MCC_by_conf.tolist(),
        "Weighted-MCC curve": weighted_MCC_by_conf.tolist(),
        "AP by class": AP_by_class.tolist(),
        "mAP": float(mAP),
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

    # cm = plt.get_cmap("tab20")
    # colors = [cm(2.0 * i / 20) for i in range(10)] + [
    #     cm((2.0 * i + 1) / 20) for i in range(10)
    # ]
    colors = [
        "tab:orange",  # 1 orange, black
        "tab:red",  # 2.1/3 red, black
        "tab:green",  # 2.2 green, white
        "teal",  # 2.3/6.1 white, black
        "tab:pink",  # 4.1 rose, white
        "tab:brown",  # 4.2 red, white
        "tab:blue",  # 4.3 blue, white
        "tab:olive",  # 5.1 yellow, black
        "lightsalmon",  # 5.2 yellow, red
        "lightsteelblue",  # 6.2 light grey, grey
        "greenyellow",  # 7 light yellow, black
        "darkseagreen",  # 7E white
        "darkblue",  # 8 grey, white
        "tab:gray",  # 9 white, grey
        "tab:purple",  # LQ grey, light grey
        "tab:cyan",  # MP light grey, black
        "magenta",  # Macro
        "purple",  # Micro
        "blue",  # Weighted
    ]

    for title, data in metrics.items():
        data = np.array(data)
        if title.startswith("Confusion Matrices"):
            n = np.ceil(np.sqrt(len(data))).astype(np.int32)
            # Confusion matrices
            fig, axes = plt.subplots(n, n, dpi=300)
            fig.suptitle(title)
            for i, cm in enumerate(data):
                cm = cm.astype(np.float32)
                tn = cm[1, 1]
                cm[1, 1] = float("nan")
                x = i % n
                y = i // n
                ax = axes[y, x]
                ax.set_title(class_names[i], {"fontsize": "small"})
                cmap = matplotlib.cm.get_cmap("Wistia")
                cmap.set_bad(color="lightgrey")
                im = ax.matshow(cm, cmap=cmap)
                # fig.colorbar(im, ax=ax)
                cm[1, 1] = tn
                cm = cm.astype(np.int32)
                for (i, j), z in np.ndenumerate(cm):
                    ax.text(
                        j,
                        i,
                        str(z),
                        ha="center",
                        va="center",
                        fontdict={"fontsize": "small"},
                    )
                ax.set_xticks([0, 1], ["P", "N"], fontsize="small")
                ax.set_yticks([0, 1], ["P", "N"], fontsize="small")
                ax.xaxis.set_ticks_position("bottom")
            fig.supxlabel("Predicted")
            fig.supylabel("Actual")
            plt.tight_layout(pad=0.5)
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
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xticks(np.linspace(0.0, 1.0, 21), rotation=90)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.savefig(
                out_dir / f"{make_filename_safe(title)}.png",
                bbox_inches="tight",
            )

    plt.figure(dpi=300)
    title = f"F1 scores by class and averaged"
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("F1 score")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    ax = plt.gca()
    ax.set_prop_cycle(color=colors)
    for label, data in metrics.items():
        data = np.array(data)
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
    title = f"MCC scores by class and averaged"
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("MCC score")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-1.01, 1.01])
    ax = plt.gca()
    ax.set_prop_cycle(color=colors)
    for label, data in metrics.items():
        data = np.array(data)
        if "MCC curves" in label:
            for i in range(len(data[0])):
                plt.plot(conf_th_list, data[:, i], label=class_names[i])
        elif "MCC curve" in label:
            plt.plot(conf_th_list, data, label=label[:-6], lw=3)
    plt.xticks(np.linspace(0.0, 1.0, 21), rotation=90)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(
        out_dir / f"{make_filename_safe(title)}.png", bbox_inches="tight"
    )

    plt.figure(dpi=300)
    title = f"Precision-Recall curves by class"
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    
    f1_values = np.linspace(0.1, 0.9, 9)
    half_samples = 10
    p_by_f1 = np.zeros((len(f1_values), 2 * half_samples))
    r_by_f1 = np.zeros((len(f1_values), 2 * half_samples))
    for j, f1 in enumerate(f1_values):
        for i, p in enumerate(np.geomspace(f1, 1, half_samples)):
            r = p * f1 / (2 * p - f1)
            p_by_f1[j][half_samples + i] = p
            r_by_f1[j][half_samples + i] = r
            p_by_f1[j][half_samples - i - 1] = r
            r_by_f1[j][half_samples - i - 1] = p
    for f1, p, r in zip(f1_values, p_by_f1, r_by_f1):
        plt.plot(r, p, linestyle="dotted", color="lightgray")
        plt.text(
            r[half_samples],
            p[half_samples],
            f"F1={f1:.1f}",
            color="lightgray",
            zorder=-1,
        )
    
    ax = plt.gca()
    ax.set_prop_cycle(
        color=[val for pair in zip(colors, colors) for val in pair]
    )
    for i in range(len(class_names)):
        p = np.array(metrics[f"Precision curves by class"])[:, i]
        r = np.array(metrics[f"Recall curves by class"])[:, i]
        p_acc = np.maximum.accumulate(np.nan_to_num(p))
        # r_acc = np.flip(np.maximum.accumulate(np.flip(r)))
        plt.step(r, p_acc, where="post", label=class_names[i])
        plt.plot(r, p, linestyle=":")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(
        out_dir / f"{make_filename_safe(title)}.png", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
