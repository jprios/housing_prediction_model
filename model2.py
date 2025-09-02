
"""
Gera um Excel com duas abas:
- previsoes: previsões de teste (id, data, y_true, y_pred e APE%)
- estatisticas: métricas globais + por id + por mês

E salva o melhor modelo e suas especificações em:
- artifacts/models/panel_best_model_<NOME>.joblib
- artifacts/specs/model_spec_panel.json

Uso:
  python model.py --input panel.csv --output resultados.xlsx
  (opcionais)
  --frac-train 0.8 --artifacts-dir artifacts
"""

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn

# ===================== Helpers =====================

def add_lags_roll(df, id_col='id', time_col='date', base_cols=None):
    """Cria lags e média móvel por id."""
    df = df.sort_values([id_col, time_col]).copy()
    if base_cols is None:
        base_cols = [c for c in ['preco_m2','quartos','suites','banheiros','vagas','area_m2'] if c in df.columns]
    for col in base_cols:
        df[f'{col}_L1']  = df.groupby(id_col)[col].shift(1)
        df[f'{col}_L2']  = df.groupby(id_col)[col].shift(2)
        df[f'{col}_MA3'] = df.groupby(id_col)[col].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    return df

def temporal_split(df, time_col='date', frac_train=0.8):
    """Divide por tempo (treino passado / teste futuro)."""
    cutoff = df[time_col].quantile(frac_train)
    train = df[df[time_col] <= cutoff].copy()
    test  = df[df[time_col]  > cutoff].copy()
    return train, test, pd.to_datetime(cutoff)

def evaluate_all(y_true, preds: dict):
    """Calcula métricas para cada chave de preds (assume já no espaço original do alvo)."""
    rows = []
    for name, y_pred in preds.items():
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2   = float(r2_score(y_true, y_pred))
        rows.append({"modelo": name, "MAPE(%)": mape*100, "MAE": mae, "RMSE": rmse, "R2": r2})
    return pd.DataFrame(rows)

def percent_err(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return 100.0 * np.abs(y_true - y_pred) / denom

def ensure_dirs(artifacts_dir: Path):
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "specs").mkdir(parents=True, exist_ok=True)

def save_best_artifacts(best_name: str,
                        best_estimator,
                        metrics_global: pd.DataFrame,
                        X_columns: list,
                        artifacts_dir: Path,
                        extra: dict | None = None):
    """Salva .joblib + spec json."""
    ensure_dirs(artifacts_dir)
    models_dir = artifacts_dir / "models"
    specs_dir  = artifacts_dir / "specs"

    model_filename = f"panel_best_model_{best_name}.joblib"
    model_path = models_dir / model_filename
    joblib.dump(best_estimator, model_path.as_posix())

    # acha a linha do campeão nas métricas
    row = metrics_global.loc[metrics_global["modelo"] == best_name].iloc[0].to_dict()

    spec = {
        "model_file": model_path.as_posix(),
        "model_name": best_name,
        "source": "panel_training_script",
        "features": list(map(str, X_columns)),
        "target": "preco_m2",
        "metrics": {k: float(v) for k, v in row.items() if k != "modelo"},
        "candidates": metrics_global.to_dict(orient="records"),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "sklearn_version": sklearn.__version__,
            "created_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    if extra:
        spec.update(extra)

    spec_path = specs_dir / "model_spec_panel.json"
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    return model_path, spec_path

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="panel.csv", help="Caminho do CSV de entrada")
    ap.add_argument("--output", type=str, default="resultados.xlsx", help="Caminho do Excel de saída")
    ap.add_argument("--frac-train", type=float, default=0.8, help="Fraçao temporal para treino (0-1)")
    ap.add_argument("--artifacts-dir", type=str, default="artifacts", help="Diretório-base de artefatos (models/specs)")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)

    # 1) Carregar dados
    path = Path(args.input)
    df = pd.read_csv(path)

    # 2) Datas e ID
    if 'date' not in df.columns:
        raise ValueError("Esperava coluna 'date' no CSV.")
    df['date'] = pd.to_datetime(df['date'])

    if 'id' not in df.columns:
        # tenta derivar de outra coluna categórica comum
        for c in ['bairro','ID','unidade']:
            if c in df.columns:
                df['id'] = pd.factorize(df[c])[0]
                break
        if 'id' not in df.columns:
            raise ValueError("Sem coluna 'id' (ou equivalente).")

    # 3) Alvo
    if 'preco_m2' not in df.columns:
        if {'preco','area_m2'}.issubset(df.columns):
            df['preco_m2'] = df['preco'] / df['area_m2']
        else:
            raise ValueError("Faltando 'preco_m2' ou ('preco' e 'area_m2').")

    # 4) Engenharia de atributos temporais
    df = add_lags_roll(df, id_col='id', time_col='date')
    df = df.dropna().reset_index(drop=True)

    # 5) Features numéricas (sem OHE, para manter robustez/velocidade)
    target = 'preco_m2'
    drop_cols = ['id', 'date']
    X_all = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns]).select_dtypes(include=[np.number])
    y_all = df[target].astype(float)

    # 6) Split temporal
    train, test, cutoff = temporal_split(df, time_col='date', frac_train=args.frac_train)
    msk_tr = df['date'] <= cutoff
    X_tr, X_te = X_all[msk_tr], X_all[~msk_tr]
    y_tr, y_te = y_all[msk_tr], y_all[~msk_tr]

    # 7) Modelos
    # HGB robusto + log transform no target (via FunctionTransformer com inverse_transform)
    hgb = HistGradientBoostingRegressor(
        random_state=42, max_iter=400, learning_rate=0.08,
        max_depth=None, min_samples_leaf=20, early_stopping=True, validation_fraction=0.1
    )
    pipe_hgb = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', RobustScaler()),
        ('hgb', hgb)
    ])
    t_y = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
    reg_hgb = TransformedTargetRegressor(regressor=pipe_hgb, transformer=t_y)

    # Baseline linear: ElasticNet (também em log)
    enet = ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=42, max_iter=8000)
    pipe_enet = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', RobustScaler()),
        ('enet', enet)
    ])
    reg_enet = TransformedTargetRegressor(regressor=pipe_enet, transformer=t_y)

    # 8) Treino
    reg_hgb.fit(X_tr, y_tr)
    reg_enet.fit(X_tr, y_tr)

    # 9) Previsões (teste) — usando explicitamente inverse_transform do target
    #    a) previsões no espaço transformado (log)
    y_pred_hgb_log  = reg_hgb.regressor_.predict(X_te)
    y_pred_enet_log = reg_enet.regressor_.predict(X_te)
    #    b) inverter para o espaço original via inverse_transform (EXIGÊNCIA #2)
    y_pred_hgb  = reg_hgb.transformer_.inverse_transform(np.asarray(y_pred_hgb_log).reshape(-1, 1)).ravel()
    y_pred_enet = reg_enet.transformer_.inverse_transform(np.asarray(y_pred_enet_log).reshape(-1, 1)).ravel()

    # ensemble simples
    mape_hgb  = mean_absolute_percentage_error(y_te, y_pred_hgb)
    mape_enet = mean_absolute_percentage_error(y_te, y_pred_enet)
    w_hgb = 1.0 if mape_hgb <= mape_enet else 0.6
    w_en  = 1.0 - w_hgb
    y_pred_ens = w_hgb * y_pred_hgb + w_en * y_pred_enet

    # 10) Tabelas de saída
    # a) Previsões (com id e data do conjunto de teste)
    test_idx = df.loc[~msk_tr, ['id','date']].reset_index(drop=True)
    preds_df = pd.DataFrame({
        'id':   test_idx['id'].to_numpy(),
        'date': test_idx['date'].dt.strftime('%Y-%m-%d').to_numpy(),
        'y_true': y_te.to_numpy(),
        'y_pred_hgb': y_pred_hgb,
        'y_pred_enet': y_pred_enet,
        'y_pred_ens': y_pred_ens,
    })
    preds_df['APE_hgb(%)']  = percent_err(preds_df['y_true'], preds_df['y_pred_hgb'])
    preds_df['APE_enet(%)'] = percent_err(preds_df['y_true'], preds_df['y_pred_enet'])
    preds_df['APE_ens(%)']  = percent_err(preds_df['y_true'], preds_df['y_pred_ens'])

    # b) Estatísticas (globais + por id + por mês)
    metrics_global = evaluate_all(y_te.to_numpy(), {
        'HGB': y_pred_hgb,
        'ElasticNet': y_pred_enet,
        'Ensemble': y_pred_ens
    }).sort_values('MAPE(%)')

    # por id
    tmp = preds_df.copy()
    metrics_by_id = (tmp
        .groupby('id')
        .apply(lambda g: pd.Series({
            'MAPE_hgb(%)': mean_absolute_percentage_error(g['y_true'], g['y_pred_hgb']) * 100,
            'MAPE_enet(%)': mean_absolute_percentage_error(g['y_true'], g['y_pred_enet']) * 100,
            'MAPE_ens(%)': mean_absolute_percentage_error(g['y_true'], g['y_pred_ens']) * 100,
            'MAE_ens': np.mean(np.abs(g['y_true'] - g['y_pred_ens'])),
            'RMSE_ens': np.sqrt(np.mean((g['y_true'] - g['y_pred_ens'])**2)),
            'n_test': len(g)
        }))
        .reset_index()
        .sort_values('MAPE_ens(%)')
    )

    # por mês (aaaa-mm)
    tmp['mes'] = pd.to_datetime(tmp['date']).dt.to_period('M').astype(str)
    metrics_by_month = (tmp
        .groupby('mes')
        .apply(lambda g: pd.Series({
            'MAPE_hgb(%)': mean_absolute_percentage_error(g['y_true'], g['y_pred_hgb']) * 100,
            'MAPE_enet(%)': mean_absolute_percentage_error(g['y_true'], g['y_pred_enet']) * 100,
            'MAPE_ens(%)': mean_absolute_percentage_error(g['y_true'], g['y_pred_ens']) * 100,
            'MAE_ens': np.mean(np.abs(g['y_true'] - g['y_pred_ens'])),
            'RMSE_ens': np.sqrt(np.mean((g['y_true'] - g['y_pred_ens'])**2)),
            'n_test': len(g)
        }))
        .reset_index()
        .sort_values('mes')
    )

    # 11) Exporta Excel (2 abas)
    out_path = Path(args.output)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        preds_df.to_excel(writer, sheet_name="previsoes", index=False)

        # monta aba de estatísticas com blocos
        start_row = 0
        metrics_global.to_excel(writer, sheet_name="estatisticas", index=False, startrow=start_row)
        ws = writer.sheets["estatisticas"]
        ws.write(start_row, 0, "Métricas globais")  # título
        start_row += len(metrics_global) + 3

        ws.write(start_row, 0, "Métricas por id (Ensemble e comparativos)")
        start_row += 1
        metrics_by_id.to_excel(writer, sheet_name="estatisticas", index=False, startrow=start_row)
        start_row += len(metrics_by_id) + 3

        ws.write(start_row, 0, "Métricas por mês (yyyy-mm)")
        start_row += 1
        metrics_by_month.to_excel(writer, sheet_name="estatisticas", index=False, startrow=start_row)

    # 12) Salvar o MELHOR modelo + spec (EXIGÊNCIA #1)
    # Se o melhor geral for "Ensemble", salvamos o melhor *modelo individual* (impossível salvar o ensemble simples como estimator).
    best_row = metrics_global.iloc[0]
    best_name_global = str(best_row["modelo"])
    best_name_to_save = best_name_global if best_name_global in ("HGB", "ElasticNet") else ("HGB" if mape_hgb <= mape_enet else "ElasticNet")
    best_estimator = reg_hgb if best_name_to_save == "HGB" else reg_enet

    extra = {
        "chosen_global": best_name_global,
        "ensemble_weights": {"HGB": float(w_hgb), "ElasticNet": float(w_en)},
    }
    model_path, spec_path = save_best_artifacts(
        best_name=best_name_to_save,
        best_estimator=best_estimator,
        metrics_global=metrics_global,
        X_columns=list(X_tr.columns),
        artifacts_dir=artifacts_dir,
        extra=extra
    )

    print(f"[OK] Arquivo salvo: {out_path.resolve()}")
    print(f"[OK] Melhor modelo salvo em: {model_path}")
    print(f"[OK] Spec salva em: {spec_path}")
    print(metrics_global)

if __name__ == "__main__":
    main()
