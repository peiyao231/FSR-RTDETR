import warnings
warnings.filterwarnings("ignore")

from ultralytics import RTDETR

# -------------------------
# Config (recommended)
# -------------------------
WEIGHTS = "runs/train/exp/weights/best.pt"         
SOURCE  = "datasets/TT100K/images/test"           
PROJECT = "runs/detect"
NAME    = "tt_FSAS+LGF+RMBConv_pred"             
if __name__ == "__main__":
    model = RTDETR(WEIGHTS)

    model.predict(
        source=SOURCE,
        conf=0.25,
        save=True,

        project=PROJECT,
        name=NAME,

    )
