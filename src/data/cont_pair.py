import csv
from pathlib import Path

# Script para contar o número de pares de imagens (âncora, positiva, negativa) em cada pasta de cada experimento e gerar um relatório em CSV.
ROOT = Path(__file__).resolve().parents[2]
PAIR_DIR = ROOT / "datasets" / "pair"
CSV_SAIDA = Path(__file__).resolve().parent / "contagem_pairs.csv"

# Extensões de imagem a serem consideradas
EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")
SHOTS = [1, 3, 5]

# Classes de interesse para o relatório
CLASSES = [
    "No Finding",
    "Pneumonia",
    "Tuberculosis",
    "Convid",
]

# Mapeamento dos nomes das pastas para os nomes dos datasets a serem exibidos no relatório.
DATASETS = {
    "01": "covid-chestxray-dataset-master",
    "02": "CXR8",
    "03": "Imbalanced-Tuberculosis",
    "04": "makedataset",
    "05": "all",
}

# Colunas do relatório CSV.
COLUNAS = (
    "Experimento",
    "Dataset",
    "Shot",
    "Classe",
    "Ancoras",
    "Positivas",
    "Negativas",
    "Total",
)

# Função auxiliar para verificar se um arquivo é uma imagem com base em sua extensão.
def _eh_imagem(caminho: Path) -> bool:
    return caminho.is_file() and caminho.suffix.lower() in EXTENSOES_IMAGEM

# Função para contar o número de imagens em uma pasta, incluindo subpastas.
def contar_imagens_pasta(pasta: Path) -> int:
    if not pasta.exists():
        return 0
    return sum(1 for arquivo in pasta.iterdir() if _eh_imagem(arquivo))

# Função para contar o número de pares de imagens (âncora, positiva, negativa) em uma pasta específica, dada a estrutura de pastas do dataset.
def contar_tipo(pasta_pair: Path, tipo: str, shot: str, classe: str) -> int:
    pasta = pasta_pair / tipo / shot / classe
    return contar_imagens_pasta(pasta)

# Função para montar as linhas do relatório, iterando sobre os experimentos, shots e classes, e contando o número de pares de cada tipo, além de calcular subtotais e totais.
def montar_linhas() -> list[dict]:
    linhas = []
    totais = {"Ancoras": 0, "Positivas": 0, "Negativas": 0, "Total": 0}

    if not PAIR_DIR.exists():
        return linhas

    for experimento in sorted(p for p in PAIR_DIR.iterdir() if p.is_dir()):
        nome_exp = experimento.name
        nome_dataset = DATASETS.get(nome_exp, nome_exp)

        for shot in SHOTS:
            pasta_shot = f"{shot}-shot"
            subtotais = {"Ancoras": 0, "Positivas": 0, "Negativas": 0, "Total": 0}

            for classe in CLASSES:
                qtd_ancora = contar_tipo(experimento, "ancora", pasta_shot, classe)
                qtd_positiva = contar_tipo(experimento, "positiva", pasta_shot, classe)
                qtd_negativa = contar_tipo(experimento, "negativa", pasta_shot, classe)
                total = qtd_ancora + qtd_positiva + qtd_negativa

                if total == 0:
                    continue

                linha = {
                    "Experimento": nome_exp,
                    "Dataset": nome_dataset,
                    "Shot": pasta_shot,
                    "Classe": classe,
                    "Ancoras": qtd_ancora,
                    "Positivas": qtd_positiva,
                    "Negativas": qtd_negativa,
                    "Total": total,
                }
                linhas.append(linha)

                for chave in subtotais:
                    subtotais[chave] += linha[chave]

            if subtotais["Total"] > 0:
                linhas.append({
                    "Experimento": nome_exp,
                    "Dataset": nome_dataset,
                    "Shot": pasta_shot,
                    "Classe": "Subtotal",
                    "Ancoras": subtotais["Ancoras"],
                    "Positivas": subtotais["Positivas"],
                    "Negativas": subtotais["Negativas"],
                    "Total": subtotais["Total"],
                })

                for chave in totais:
                    totais[chave] += subtotais[chave]

    if totais["Total"] > 0:
        linhas.append({
            "Experimento": "Total",
            "Dataset": "Total Geral",
            "Shot": "-",
            "Classe": "-",
            "Ancoras": totais["Ancoras"],
            "Positivas": totais["Positivas"],
            "Negativas": totais["Negativas"],
            "Total": totais["Total"],
        })

    return linhas

#  Função para imprimir a tabela de contagem de pares de imagens no console, formatada com colunas alinhadas.
def imprimir_tabela(linhas: list[dict]) -> None:
    if not linhas:
        print("Nenhuma imagem encontrada em datasets/pair/")
        return

    cabecalho = list(COLUNAS)
    larguras = [len(col) for col in cabecalho]

    for linha in linhas:
        for indice, coluna in enumerate(cabecalho):
            larguras[indice] = max(larguras[indice], len(str(linha[coluna])))

    separador = " | "
    print(separador.join(col.ljust(larguras[i]) for i, col in enumerate(cabecalho)))
    print("-+-".join("-" * largura for largura in larguras))

    for linha in linhas:
        print(
            separador.join(
                str(linha[coluna]).ljust(larguras[i])
                for i, coluna in enumerate(cabecalho)
            )
        )

# Função para salvar a tabela de contagem de pares de imagens em um arquivo CSV, criando a pasta se necessário.
def salvar_csv(caminho: Path, linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=COLUNAS)
        writer.writeheader()
        writer.writerows(linhas)

# Função principal para executar o script, verificando se a pasta de dados de pares existe, montando a tabela, imprimindo-a e salvando o relatório em CSV.
if __name__ == "__main__":
    if not PAIR_DIR.exists():
        print(f"Pasta nao encontrada: {PAIR_DIR}")
        print("Execute: python src/data/ger_pair.py")
    else:
        linhas = montar_linhas()
        imprimir_tabela(linhas)
        salvar_csv(CSV_SAIDA, linhas)
        print(f"\nRelatorio salvo em: {CSV_SAIDA}")
