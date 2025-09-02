#!/usr/bin/env python
# build_panel.py
import argparse
import os
import json
import pandas as pd
from pandas.tseries.offsets import MonthEnd

def load_data(input_path: str | None = None, base_dir: str | None = None) -> pd.DataFrame:
    """
    Carrega o JSON com imóveis. Use --input OU --base-dir.
    - --input: caminho direto para o arquivo .json
    - --base-dir: base que contém data/mock_data/imoveis_mock.json
    """
    if input_path:
        json_path = input_path
    elif base_dir:
        json_path = os.path.join(base_dir, "data", "mock_data", "imoveis_mock.json")
    else:
        raise ValueError("Informe --input ou --base-dir.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    return df

def filter_offer(df: pd.DataFrame, offer: str, col: str = "finalidade") -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Coluna '{col}' não encontrada no DataFrame.")
    mask = (
        df[col].astype(str).str.strip().str.lower()
        == str(offer).strip().lower()
    )
    return df.loc[mask].copy()



def get_price_m2(df: pd.DataFrame, align_month_end: bool = False) -> pd.DataFrame:
    """
    Cria preco_m2, date (a partir de ano/mes) e id (a partir de bairro).
    Faz coerção numérica básica nas colunas relevantes.
    """
    out = df.copy()

    for col in ["preco", "area_m2", "quartos", "suites", "banheiros", "vagas", "ano", "mes"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "preco" not in out.columns or "area_m2" not in out.columns:
        raise KeyError("As colunas 'preco' e 'area_m2' são necessárias para calcular 'preco_m2'.")
    out["preco_m2"] = out["preco"] / out["area_m2"]

    if "ano" in out.columns and "mes" in out.columns:
        out["date"] = pd.to_datetime(
            dict(
                year=out["ano"].fillna(1970).astype(int),
                month=out["mes"].fillna(1).astype(int),
                day=1,
            ),
            errors="coerce",
        )
        if align_month_end:
            out["date"] = out["date"] + MonthEnd(0)
    elif "date" not in out.columns:
        raise KeyError("Forneça 'ano' e 'mes' (ou já traga 'date') para construir a coluna de data.")

    if "bairro" not in out.columns:
        raise KeyError("A coluna 'bairro' é necessária para gerar 'id'.")
    out["id"] = out["bairro"].astype("category").cat.codes

    return out

def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega por (date, id) tirando a média de métricas selecionadas.
    """
    group_cols = ["date", "id"]
    value_cols = ["preco_m2", "quartos", "suites", "banheiros", "vagas"]
    for c in value_cols:
        if c not in df.columns:
            df[c] = pd.NA
    panel = df.groupby(group_cols, as_index=False)[value_cols].mean(numeric_only=True)
    return panel

def export_panel(panel: pd.DataFrame, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    panel.to_csv(output_path, index=False, encoding="utf-8")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Gera painel mensal (date x id) com médias por bairro.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="Caminho direto para o JSON de imóveis (ex.: .../imoveis_mock.json)")
    g.add_argument("--base-dir", help="Diretório base que contém data/mock_data/imoveis_mock.json")
    parser.add_argument("--offer",default = 'Venda', help="Seleciona o tipo de oferta")
    parser.add_argument("--output", default="panel.csv", help="Caminho do CSV de saída (default: panel.csv)")
    parser.add_argument("--month-end", action="store_true", help="Alinhar 'date' para o fim do mês")
    args = parser.parse_args()

    df = load_data(input_path=args.input, base_dir=args.base_dir)
    df = get_price_m2(df, align_month_end=args.month_end)
    df = filter_offer(df,offer = args.offer)
    panel = build_panel(df)
    out = export_panel(panel, args.output)
    print(f"✔ Painel salvo em: {out}  (linhas={len(panel)}, colunas={panel.shape[1]})")

if __name__ == "__main__":
    main()