import warnings
warnings.filterwarnings("ignore")

from ultralytics import RTDETR

# =========================
# Public-safe / reproducible paths (recommended)
# =========================
MODEL_YAML = "ultralytics/cfg/models/rt-detr/FSR.yaml"
DATA_YAML  = "tt100k.yaml"  
PROJECT_DIR = "runs/detect"
RUN_NAME = "tt_FSR"



if __name__ == "__main__":
    model = RTDETR(MODEL_YAML)

    # model.load("path/to/pretrained.pt")  

    model.train(
        data=DATA_YAML,
        cache=False,
        imgsz=640,
        epochs=200,
        batch=4,
        workers=4,
        device=0,              
        optimizer="AdamW",
        amp=False,            
        lr0=1e-4,
        weight_decay=5e-4,

        fliplr=0.0,           
        flipud=0.0,           
        degrees=3.0,           
        mosaic=1.0,           

        patience=50,
        project=PROJECT_DIR,
        name=RUN_NAME,
    )
