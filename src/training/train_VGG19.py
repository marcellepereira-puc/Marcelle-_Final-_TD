import sys
from pathlib import Path
from src.base_model.model_VGG19 import modelo_vgg19

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
sys.path.insert(0, str(ROOT / "src" / "training"))

from config import MODELS_DIR, create_vgg19_model
from train_utils import treinar_geral

# Define o nome do modelo.
NOME_MODELO = "vgg19 manual"

# Treina o modelo.
if __name__ == "__main__":
    treinar_geral(NOME_MODELO, MODELS_DIR, modelo_vgg19)
