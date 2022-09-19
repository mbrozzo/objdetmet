from pathlib import Path
import argparse
from types import new_class
import numpy as np
from tqdm import tqdm

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


def load_label(path):
    path = Path(path)
    empty = (
        np.zeros((0, 1, np.int32)),
        np.zeros((0, 4, np.float32)),
        np.zeros((0, 1, np.float32)),
    )
    if not path.exists():
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
                cl = int(cl)
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
        print(
            f"ERROR: Could not load labels from {path}, empty labels will be returned for the corresponding image."
        )
        print(e)
        return empty
    return classes, boxes, confidences


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
    conf_idxs = np.where(det_confidences >= conf_thresh)
    det_classes = det_classes[conf_idxs]
    iou_matrix = iou_matrix[:, conf_idxs]
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
    classes_of_matches = np.stack([gt_classes_matches, det_classes_matches], 1)
    # Construct confusion matrix
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype="int")
    # Update cm with matches
    for match in classes_of_matches:
        cm[match] += 1
    # Update cm with false positives
    fp_idxs = np.where(np.logical_not(box_matches_matrix.any(0)))
    fp_classes = gt_classes[fp_idxs]
    for cl in fp_classes:
        cm[n_classes, cl] += 1
    # Update cm with false negatives
    fn_idxs = np.where(np.logical_not(box_matches_matrix.any(1)))
    fn_classes = det_classes[fn_idxs]
    for cl in fn_classes:
        cm[cl, n_classes] += 1
    return cm


def calculate_tp_fp_fn(confusion_matrix):
    # Arrays of length n_classes with metrics for each class and sum for all classes
    tp = confusion_matrix.diagonal()
    fp = confusion_matrix.sum(0) - tp
    fn = confusion_matrix.sum(1) - tp
    tp[-1] = tp[:-1].sum()
    fp[-1] = fp[:-1].sum()
    fn[-1] = fn[:-1].sum()
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ground_truths_dir",
        type=Path,
    )
    parser.add_argument(
        "detections_dir",
        type=Path,
    )
    parser.add_argument(
        "--n-classes",
        "-n",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--iou-thresh",
        "-i",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--conf-thresh",
        "-c",
        type=float,
        default=0.1,
    )
    args = parser.parse_args()
    ground_truths_dir: Path = args.ground_truths_dir
    detections_dir: Path = args.detections_dir
    conf_thresh_default: float = args.conf_thresh
    iou_thresh_default: float = args.iou_thresh
    n_classes: int = args.n_classes
    # images_dir: Path = args.images_dir

    iou_step = 0.05
    iou_th_list = []
    iou_th = 0.5
    while iou_th < 1.0:
        iou_th_list.append(iou_th)
        iou_th += iou_step

    conf_step = 0.01
    conf_th_list = []
    conf_th = 0
    while iou_th <= 1.0:
        conf_th_list.append(conf_th)
        conf_th += conf_step

    # Define stats arrays
    CM_by_iou_and_conf = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes + 1, n_classes + 1),
        dtype="int",
    )
    TP_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes + 1), dtype="int"
    )
    FP_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes + 1), dtype="int"
    )
    FN_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes + 1), dtype="int"
    )
    F1_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes + 1), dtype="float"
    )
    P_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes + 1), dtype="float"
    )
    R_by_iou_conf_and_class = np.zeros(
        (len(iou_th_list), len(conf_th_list), n_classes + 1), dtype="float"
    )
    AP_by_iou_and_class = np.zeros(
        (len(iou_th_list), n_classes + 1), dtype="float"
    )
    mAP_by_iou = np.zeros((len(iou_th_list)), dtype="float")
    mAP_05_095 = np.array(0)

    # Calculate confusion matrix
    for gt_path in tqdm(list(ground_truths_dir.glob("*.txt"))):
        detection_path = detections_dir / gt_path.stem
        if not detection_path.exists():
            print("WARNING: Detection label file not found, skipping.")
            continue
        gt_classes, gt_boxes, _ = load_label(gt_path)
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

    # Calculate other stats
    for i, iou_th in enumerate(iou_th_list):
        for j, conf_th in enumerate(conf_th_list):
            tp, fp, fn = calculate_tp_fp_fn(CM_by_iou_and_conf[i, j])
            TP_by_iou_conf_and_class[i, j] = tp
            FP_by_iou_conf_and_class[i, j] = fp
            FN_by_iou_conf_and_class[i, j] = fn
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            P_by_iou_conf_and_class[i, j] = p
            R_by_iou_conf_and_class[i, j] = r
            F1_by_iou_conf_and_class[i, j] = 2 * p * r / (p + r)

    # Calculate AP
    for i, iou_th in enumerate(iou_th_list):
        p_by_conf_and_class = P_by_iou_conf_and_class[i]
        r_by_conf_and_class = R_by_iou_conf_and_class[i]
        ap = (
            np.maximum.accumulate(p_by_conf_and_class, 0) * r_by_conf_and_class
        ).sum(0)
        AP_by_iou_and_class[i] = ap
        mAP_by_iou[i] = ap.mean(1)
    mAP_05_095 = mAP_by_iou.mean()


if __name__ == "__main__":
    main()
