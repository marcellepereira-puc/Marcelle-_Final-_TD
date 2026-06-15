"""Configuração de GPU para treino (CUDA ou DirectML no Windows)."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

_GPU_CONFIGURADA = False


def configurar_gpu() -> bool:
    global _GPU_CONFIGURADA
    if _GPU_CONFIGURADA:
        return True

    gpus = tf.config.list_physical_devices("GPU")
    dml = tf.config.list_physical_devices("DML")

    print("Dispositivos TensorFlow:", tf.config.list_physical_devices())

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        print(f"GPU ativa: {len(gpus)} dispositivo(s)")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        _GPU_CONFIGURADA = True
        return True

    if dml:
        print(f"GPU DirectML ativa: {len(dml)} dispositivo(s)")
        for device in dml:
            print(f"  - {device.name}")
        _GPU_CONFIGURADA = True
        return True

    print("[aviso] Nenhuma GPU detectada — treino exige GPU.")
    print("Use Python 3.10 com: pip install -r requirements.txt")
    return False
