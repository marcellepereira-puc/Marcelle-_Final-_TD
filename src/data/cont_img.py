import csv
from pathlib import Path
# Script para contar o número de imagens em cada pasta de cada dataset e gerar um relatório em CSV.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED_PATH = _PROJECT_ROOT / "datasets" / "processed" / "processed_v1"
_RELATORIO_CSV = _PROJECT_ROOT / "src" / "data" / "contagem_imagens.csv"

# Extensões de imagem a serem consideradas
EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")
COLUNAS = ("No Finding", "Pneumonia", "Tuberculosis", "Convid", "Total")

# Lista de datasets a serem processados, com o nome da pasta e o nome de exibição.
DATASETS = (
    ("covid-chestxray-dataset-master", "Covid-chestxray-dataset"),
    ("CXR8", "CXR8"),
    ("Imbalanced-Tuberculosis", "Imbalanced-Tuberculosis"),
    ("makedataset","MadeDataset"),
    ("padchest", "Padchest"),
)

# Função auxiliar para verificar se um arquivo é uma imagem com base em sua extensão.
def _eh_imagem(caminho: Path) -> bool:
    return caminho.is_file() and caminho.suffix.lower() in EXTENSOES_IMAGEM

# Função para contar o número de imagens em uma pasta, incluindo subpastas.
def contar_imagens_pasta(pasta: Path) -> int:
    if not pasta.exists():
        return 0
    return sum(1 for arquivo in pasta.rglob("*") if _eh_imagem(arquivo))

# Função para mapear o nome da pasta para a coluna correspondente no relatório, com base no nome do dataset.
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

# Função para contar o número de imagens em cada pasta de um dataset e retornar um dicionário com as contagens.
def contar_por_dataset(pasta_dataset: Path, nome_dataset: str) -> dict[str, int]:
    contagens = {coluna: 0 for coluna in COLUNAS[:-1]}

    if not pasta_dataset.exists():
        return {**contagens, "Total": 0}

    for pasta_label in pasta_dataset.iterdir():
        if not pasta_label.is_dir():
            continue

        coluna = mapear_label(pasta_label.name, nome_dataset)
        if coluna is None:
            continue

        contagens[coluna] += contar_imagens_pasta(pasta_label)

    contagens["Total"] = sum(contagens[coluna] for coluna in COLUNAS[:-1])
    return contagens

# Função para montar a tabela de contagem de imagens para todos os datasets, retornando as colunas e as linhas da tabela.
def montar_tabela() -> tuple[list[str], list[dict[str, int]]]:
    linhas = []
    totais = {coluna: 0 for coluna in COLUNAS}

    for pasta, nome_exibicao in DATASETS:
        contagens = contar_por_dataset(_PROCESSED_PATH / pasta, pasta)
        linhas.append({"dataset": nome_exibicao, **contagens})
        for coluna in COLUNAS:
            totais[coluna] += contagens[coluna]

    linhas.append({"dataset": "Total Geral", **totais})
    return list(COLUNAS), linhas

# Função para imprimir a tabela de contagem de imagens no console, formatada com colunas alinhadas.
def imprimir_tabela(colunas: list[str], linhas: list[dict[str, int]]) -> None:
    cabecalho = ["Dataset", *colunas]
    larguras = [len(col) for col in cabecalho]

    for linha in linhas:
        larguras[0] = max(larguras[0], len(linha["dataset"]))
        for indice, coluna in enumerate(colunas, start=1):
            larguras[indice] = max(larguras[indice], len(str(linha[coluna])))

    separador = " | "
    print(cabecalho[0].ljust(larguras[0]), end=separador)
    print(separador.join(col.rjust(larguras[i + 1]) for i, col in enumerate(colunas)))
    print("-+-".join("-" * largura for largura in larguras))

    for linha in linhas:
        print(linha["dataset"].ljust(larguras[0]), end=separador)
        print(
            separador.join(
                str(linha[coluna]).rjust(larguras[i + 1])
                for i, coluna in enumerate(colunas)
            )
        )

# Função para salvar a tabela de contagem de imagens em um arquivo CSV, criando a pasta se necessário.
def salvar_csv(caminho: Path, colunas: list[str], linhas: list[dict[str, int]]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=["Dataset", *colunas])
        writer.writeheader()
        for linha in linhas:
            writer.writerow({"Dataset": linha["dataset"], **{col: linha[col] for col in colunas}})

# Função principal para executar o script, verificando se a pasta de dados processados existe, montando a tabela, imprimindo-a e salvando o relatório em CSV.
if __name__ == "__main__":
    if not _PROCESSED_PATH.exists():
        print(f"Pasta não encontrada: {_PROCESSED_PATH}")
    else:
        colunas, linhas = montar_tabela()
        imprimir_tabela(colunas, linhas)
        salvar_csv(_RELATORIO_CSV, colunas, linhas)
        print(f"\nRelatório salvo em: {_RELATORIO_CSV}")
