import shutil
from pathlib import Path

_BASE = Path(__file__).resolve().parent
_DATASETS = _BASE.parents[1] / "datasets"
SPLIT_DIR = _DATASETS / "split"
PASTA_SAIDA = SPLIT_DIR / "all"

# datasets que serao agrupados (nao inclui "all")
DATASETS = [
    "covid-chestxray-dataset-master",
    "CXR8",
    "Imbalanced-Tuberculosis",
    "makedataset",
]

# mantem somente as 4 labels do projeto
LABELS = [
    "No Finding",
    "Pneumonia",
    "Tuberculosis",
    "Convid",
]

SPLITS = ("train", "val", "test")
EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")

# Função auxiliar para verificar se um arquivo é uma imagem com base em sua extensão.
def _eh_imagem(arquivo: Path) -> bool:
    return arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_IMAGEM

# Função para gerar um nome de destino único para uma imagem, evitando sobrescrever arquivos com o mesmo nome, adicionando um prefixo baseado no nome do dataset.
def _nome_destino(destino: Path, dataset: str, nome_arquivo: str) -> Path:

    caminho = destino / nome_arquivo
    if not caminho.exists():
        return caminho

    prefixo = dataset.replace(" ", "_")
    return destino / f"{prefixo}_{nome_arquivo}"

# Função para copiar imagens de uma pasta de origem para uma pasta de destino, criando a pasta de destino se necessário, e contando quantas imagens foram copiadas.
def copiar_imagens(origem: Path, destino: Path, dataset: str) -> int:
    destino.mkdir(parents=True, exist_ok=True)
    copiadas = 0

    for arquivo in origem.iterdir():
        if not _eh_imagem(arquivo):
            continue

        caminho_destino = _nome_destino(destino, dataset, arquivo.name)
        if caminho_destino.exists():
            continue

        shutil.copy2(arquivo, caminho_destino)
        copiadas += 1

    return copiadas

# Função principal para agrupar os datasets em um único diretório, iterando sobre os datasets, splits e labels, copiando as imagens para o destino e mantendo um resumo do número de imagens copiadas por split e por label.
def agrupar_split_all():
    """
    Cria datasets/split/all/ juntando todos os datasets.

    Estrutura final:
    datasets/split/all/train/No Finding/
    datasets/split/all/val/Pneumonia/
    datasets/split/all/test/Tuberculosis/
    datasets/split/all/test/Convid/
    """
    if not SPLIT_DIR.exists():
        print(f"[aviso] Split nao encontrado: {SPLIT_DIR}")
        print("Execute: python src/data/ger_split.py")
        return

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    resumo = {split: 0 for split in SPLITS}
    contagem_por_label = {label: 0 for label in LABELS}

    print(f"Agrupando datasets em: {PASTA_SAIDA}")
    print(f"Labels mantidas: {', '.join(LABELS)}\n")

    for dataset in DATASETS:
        pasta_dataset = SPLIT_DIR / dataset
        if not pasta_dataset.is_dir():
            print(f"[aviso] Dataset nao encontrado, pulando: {pasta_dataset}")
            continue

        print(f"--- {dataset} ---")

        for split in SPLITS:
            for label in LABELS:
                pasta_origem = pasta_dataset / split / label
                pasta_destino = PASTA_SAIDA / split / label

                if not pasta_origem.is_dir():
                    continue

                copiadas = copiar_imagens(pasta_origem, pasta_destino, dataset)
                resumo[split] += copiadas
                contagem_por_label[label] += copiadas

                if copiadas > 0:
                    print(f"  {split}/{label}: +{copiadas} imagens")

        print()

    total = sum(resumo.values())
    if total == 0:
        print("Nenhuma imagem copiada.")
        return

    print("Resumo por split:")
    for split in SPLITS:
        print(f"  {split}: {resumo[split]}")

    print("\nResumo por label:")
    for label in LABELS:
        print(f"  {label}: {contagem_por_label[label]}")

    print(f"\nConcluido — total={total}")
    print(f"Pasta criada: {PASTA_SAIDA}")

# Função principal para executar o script, verificando se a pasta de split existe e chamando a função de agrupamento.
if __name__ == "__main__":
    agrupar_split_all()
