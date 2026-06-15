import random
import shutil
from pathlib import Path

_BASE = Path(__file__).resolve().parent
_DATASETS = _BASE.parents[1] / "datasets"

PASTA_ENTRADA = _DATASETS / "processed" / "processed_v1"
PASTA_SAIDA = _DATASETS / "split"

ROTULOS_PERMITIDOS = ("No Finding", "Pneumonia", "Tuberculosis", "Convid")

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")

# Proporções para divisão dos dados
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Semente para reprodutibilidade da divisão aleatória dos dados.
RANDOM_SEED = 42

# Datasets a serem agrupados (não inclui "all", que é o destino final).
SPLITS = ("train", "val", "test")

# Função auxiliar para verificar label e normalizar os nomes das labels, mapeando os nomes das pastas para as labels permitidas, e tratando casos específicos de cada dataset.
def mapear_label(nome_pasta: str, nome_dataset: str) -> str | None:
    nome = nome_pasta.strip().lower()

    if nome_dataset == "covid-chestxray-dataset-master" and nome == "pneumonia":
        return "Convid"
    if "covid" in nome or "convid" in nome:
        return "Convid"
    if nome in {"no finding", "normal"}:
        return "No Finding"
    if "tuberculosis" in nome:
        return "Tuberculosis"
    if "pneumonia" in nome:
        return "Pneumonia"

    return None

# Função para normalizar o nome da label, usando a função de mapeamento e tratando casos em que o nome da pasta não corresponde a nenhuma label permitida, retornando um nome genérico baseado no nome da pasta.
def normalizar_label(nome_pasta: str, nome_dataset: str) -> str:
    label = mapear_label(nome_pasta, nome_dataset)
    if label is not None:
        return label

    nome = nome_pasta.strip().replace("/", "_").replace("\\", "_")
    return nome or "desconhecido"

# Função para listar os datasets 
def listar_datasets(pasta_entrada: Path) -> list[Path]:
    if not pasta_entrada.exists():
        return []
    return sorted(pasta for pasta in pasta_entrada.iterdir() if pasta.is_dir())

# Função para coletar as imagens organizadas por dataset e label, verificando se cada pasta existe e se cada arquivo é uma imagem válida, e retornando um dicionário com as chaves sendo tuplas de (dataset, label) e os valores sendo listas de tuplas de (caminho da imagem, nome do arquivo).
def coletar_imagens_por_dataset_label() -> dict[tuple[str, str], list[tuple[Path, str]]]:
    imagens: dict[tuple[str, str], list[tuple[Path, str]]] = {}
    pasta_entrada = Path(PASTA_ENTRADA)

    if not pasta_entrada.exists():
        print(f"[aviso] Pasta de entrada não encontrada, pulando: {pasta_entrada}")
        return imagens

    for pasta_dataset in listar_datasets(pasta_entrada):
        if not pasta_dataset.exists():
            print(f"[aviso] Dataset não encontrado, pulando: {pasta_dataset}")
            continue

        for pasta_label in pasta_dataset.iterdir():
            if not pasta_label.is_dir():
                continue

            if not pasta_label.exists():
                print(f"[aviso] Pasta de label não encontrada, pulando: {pasta_label}")
                continue

            dataset = pasta_dataset.name
            label = normalizar_label(pasta_label.name, dataset)
            chave = (dataset, label)
            imagens.setdefault(chave, [])

            for arquivo in pasta_label.rglob("*"):
                if not arquivo.is_file() or arquivo.suffix.lower() not in EXTENSOES_IMAGEM:
                    continue

                imagens[chave].append((arquivo, arquivo.name))

    return imagens

# Função para copiar as imgens de distino para destino, criando a pasta de destino se necessário, e contando quantas imagens foram copiadas, evitando sobrescrever arquivos com o mesmo nome.
def copiar_imagens(lista_imagens, destino: Path) -> int:
    destino.mkdir(parents=True, exist_ok=True)
    copiadas = 0

    for origem, nome_arquivo in lista_imagens:
        if not origem.exists():
            print(f"[aviso] Imagem não encontrada, pulando: {origem}")
            continue

        caminho_destino = destino / nome_arquivo
        if caminho_destino.exists():
            continue

        shutil.copy2(origem, caminho_destino)
        copiadas += 1

    return copiadas


def dividir_lista(lista_imagens: list, train_ratio: float, val_ratio: float):
    total = len(lista_imagens)
    train_fim = int(total * train_ratio)
    val_fim = train_fim + int(total * val_ratio)
    return lista_imagens[:train_fim], lista_imagens[train_fim:val_fim], lista_imagens[val_fim:]

# dividie as imagens em train, val e test, copiando as imagens para o destino e mantendo um resumo do número de imagens copiadas por split e por label, e imprimindo o resultado no console.
def dividir_imagens(pasta_entrada=PASTA_ENTRADA, pasta_saida=PASTA_SAIDA):
    pasta_entrada = Path(pasta_entrada)
    pasta_saida = Path(pasta_saida)

    if not pasta_entrada.exists():
        print(f"[aviso] Pasta de entrada não encontrada, pulando split: {pasta_entrada}")
        return {}

    random.seed(RANDOM_SEED)

    imagens_por_grupo = coletar_imagens_por_dataset_label()
    if not imagens_por_grupo:
        print("\nNenhuma imagem encontrada.")
        return {}

    resumo = {split: 0 for split in SPLITS}
    datasets = sorted({dataset for dataset, _ in imagens_por_grupo.keys()})
    labels = sorted({label for _, label in imagens_por_grupo.keys()})

    print(f"Gerando split em: {pasta_saida}")
    print(f"Estrutura: split/{{dataset}}/{{train|val|test}}/{{label}}/")
    print(f"Datasets encontrados: {len(datasets)}")
    print(f"Labels encontradas: {len(labels)}\n")

    for (dataset, label), imagens in sorted(imagens_por_grupo.items()):
        if not imagens:
            print(f"[aviso] Nenhuma imagem para '{dataset}/{label}', pulando.")
            continue

        random.shuffle(imagens)
        train, val, test = dividir_lista(imagens, TRAIN_RATIO, VAL_RATIO)
        particoes = {"train": train, "val": val, "test": test}

        for split, lista in particoes.items():
            destino = pasta_saida / dataset / split / label
            copiadas = copiar_imagens(lista, destino)
            resumo[split] += copiadas

        print(
            f"{dataset}/{label}: total={len(imagens)} | "
            f"train={len(train)} | val={len(val)} | test={len(test)}"
        )

    total_geral = sum(resumo.values())
    if total_geral == 0:
        print("\nNenhuma imagem copiada.")
    else:
        print(
            f"\nConcluído — train={resumo['train']} | "
            f"val={resumo['val']} | test={resumo['test']} | total={total_geral}"
        )

    return resumo

# Função principal para executar o script, verificando se a pasta de split existe e chamando a função de divisão.
if __name__ == "__main__":
    dividir_imagens()
