import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def stabilize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def evaluate_map(model, dataloader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)

            preds = []
            gts = []
            for output, target in zip(outputs, targets):
                preds.append(
                    {
                        "boxes": output["boxes"].detach().cpu(),
                        "scores": output["scores"].detach().cpu(),
                        "labels": output["labels"].detach().cpu(),
                    }
                )
                gts.append(
                    {
                        "boxes": target["boxes"].detach().cpu(),
                        "labels": target["labels"].detach().cpu(),
                    }
                )
            metric.update(preds, gts)

    return metric.compute()


def compute_prf1(model, dataloader, device, score_threshold=0.5, iou_threshold=0.5):
    model.eval()

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
                gt_boxes = target["boxes"].detach().cpu()

                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]

                if len(pred_boxes) == 0:
                    false_negatives += len(gt_boxes)
                    continue

                if len(gt_boxes) == 0:
                    false_positives += len(pred_boxes)
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                matched_gt = set()
                for pred_idx in range(ious.shape[0]):
                    gt_idx = torch.argmax(ious[pred_idx]).item()
                    max_iou = ious[pred_idx, gt_idx].item()
                    if max_iou >= iou_threshold and gt_idx not in matched_gt:
                        matched_gt.add(gt_idx)
                        true_positives += 1
                    else:
                        false_positives += 1

                false_negatives += len(gt_boxes) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
