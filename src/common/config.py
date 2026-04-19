from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DATASET_ROOT = SRC_ROOT / "Datasets" / "Gold"
WEIGHTS_ROOT = PROJECT_ROOT / "weights"
RESULTS_ROOT = PROJECT_ROOT / "results"


def ensure_runtime_dirs():
    for path in [
        WEIGHTS_ROOT / "faster_rcnn",
        WEIGHTS_ROOT / "retinanet",
        WEIGHTS_ROOT / "yolo_student1",
        WEIGHTS_ROOT / "yolo_student2",
        RESULTS_ROOT / "tables",
        RESULTS_ROOT / "figures",
        RESULTS_ROOT / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)
