import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set before importing ultralytics
warnings.filterwarnings("ignore")

from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info


def ap75_from_result(r):
    # all_ap is usually [nc, 10] for IoU=0.50:0.05:0.95; index 5 corresponds to 0.75
    try:
        return float(r.box.all_ap[:, 5].mean())
    except Exception:
        return None


if __name__ == "__main__":
    model = RTDETR("runs/detect/tt_FSR/weights/best.pt")

    r = model.val(
        data="datasets/tt100k.yaml",
        split="test",
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project="runs/val",
        name="tt_FSR",
    )

    p = r.results_dict.get("metrics/precision(B)")
    rc = r.results_dict.get("metrics/recall(B)")
    map50 = r.results_dict.get("metrics/mAP50(B)")
    map5095 = r.results_dict.get("metrics/mAP50-95(B)")
    map75 = ap75_from_result(r)

    pre_ms = float(r.speed.get("preprocess", 0.0))
    inf_ms = float(r.speed.get("inference", 0.0))
    post_ms = float(r.speed.get("postprocess", 0.0))
    total_ms = pre_ms + inf_ms + post_ms

    fps_all = (1000.0 / total_ms) if total_ms > 0 else None
    fps_inf = (1000.0 / inf_ms) if inf_ms > 0 else None

    _, n_p, _, flops = model_info(model.model)  # flops in GFLOPs

    out = {
        "P": p,
        "R": rc,
        "mAP50": map50,
        "mAP75": map75,
        "mAP50-95": map5095,
        "FPS_all": fps_all,      # preprocess + inference + postprocess
        "FPS_infer": fps_inf,    # inference only
        "Params": int(n_p) if n_p is not None else None,
        "GFLOPs": float(flops) if flops is not None else None,
    }

    save_path = r.save_dir / "metrics.txt"
    with open(str(save_path), "w", encoding="utf-8") as f:
        for k, v in out.items():
            f.write(f"{k}: {v}\n")
