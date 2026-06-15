import sys
from pathlib import Path
from src.base_model.model_ResNet50 import ResNet50V2_manual

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
sys.path.insert(0, str(ROOT / "src" / "training"))

from config import MODELS_DIR, create_model
from train_utils import treinar_geral

NOME_MODELO = "resnet50v2 manual"

if __name__ == "__main__":
    treinar_geral(NOME_MODELO, MODELS_DIR, ResNet50V2_manual)
