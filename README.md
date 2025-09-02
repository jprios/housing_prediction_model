# Real Estate Price per m² — Panel-Based Modeling

This project builds a monthly **panel** (date × neighborhood id) from raw listing JSON and trains models to predict **price per square meter (R$/m²)**. The basic flow is:

1) Convert JSON → `panel.csv`  
2) Train and evaluate models → `resultados.xlsx`  
3) Save the best model (`.joblib`) + its specification (`model_spec_panel.json`)

---

## Project Structure

- `build_panel.py` — Creates `panel.csv` from a JSON file (e.g., `imoveis_mock_v5.json`). It computes `preco_m2`, builds a `date` column from `ano/mes`, creates a neighborhood `id`, and aggregates to a monthly panel.
- `model2.py` — Trains two regressors (HistGradientBoosting and ElasticNet) with a log-transform on the target via `TransformedTargetRegressor`. **Metrics are computed after inverse-transform**. Exports an Excel file with predictions/stats and saves artifacts.
- `imoveis_mock_v5.json` — Example raw data for quick testing.
- `requirements.txt` — Python dependencies.

> Generated outputs:
> - `panel.csv` (aggregated panel)  
> - `resultados.xlsx` (predictions + statistics)  
> - `artifacts/models/panel_best_model_<NAME>.joblib` (best model)  
> - `artifacts/specs/model_spec_panel.json` (model spec containing `model_file`, features, and metrics)

---

## Requirements

- Python **3.10+** recommended
- Pip/venv (or conda)

### Install

```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 1) Build the Panel (`panel.csv`)

You can point directly to the JSON file **or** to a base directory that contains `data/mock_data/imoveis_mock.json`.

**Examples:**
```bash
# direct JSON path
python build_panel.py --input imoveis_mock_v5.json --offer Venda --output panel.csv

# base directory (will look for data/mock_data/imoveis_mock.json)
python build_panel.py --base-dir . --offer Venda --output panel.csv

# align the 'date' column to the end of the month
python build_panel.py --input imoveis_mock_v5.json --offer Venda --month-end --output panel.csv
```

**Useful parameters**
- `--offer`: filter offer type (e.g., `Venda`, `Aluguel`)  
- `--month-end`: shifts `date` to month-end (optional)

> The script will create the output folder if it does not exist.

---

## 2) Train & Save Artifacts

```bash
python model2.py --input panel.csv --output resultados.xlsx --artifacts-dir artifacts
```

What it does:
- Temporal split (train on the past, test on the future)  
- Trains **HistGradientBoosting** and **ElasticNet** using `TransformedTargetRegressor` (log1p/expM1)  
- **Computes metrics after inverse-transform** (original R$/m² scale)  
- Creates `resultados.xlsx` with:
  - **previsoes** sheet: `id`, `date`, `y_true`, `y_pred_*`, `APE_%`
  - **estatisticas** sheet: global metrics, by `id`, by month
- Picks the best global model and saves:
  - `artifacts/models/panel_best_model_<NAME>.joblib`
  - `artifacts/specs/model_spec_panel.json`

---

## 3) Quick Start (2 commands)

```bash
python build_panel.py --input imoveis_mock_v5.json --offer Venda --output panel.csv
python model2.py --input panel.csv --output resultados.xlsx --artifacts-dir artifacts
```

---

## 4) Troubleshooting

- **Missing `date`** in CSV: build `panel.csv` using `build_panel.py` (it constructs `date` from `ano/mes`).  
- **Missing `id`**: the panel builder creates `id` from `bairro`.  
- **Missing `preco_m2`**: `build_panel.py` computes it from `preco` and `area_m2`.  
- **Folders don’t exist**: both scripts create required directories automatically.

---

## 5) Publish to Git (example)

```bash
git init
git add .
git commit -m "Initial commit: panel builder, training, and artifacts"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

---

## License

MIT (or your preferred license).
