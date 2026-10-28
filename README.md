# Electricity Theft Detection Dashboard

A local Flask application around the existing Gaussian Naive Bayes model in `Electricity_Theft_Bayesian_Complete.ipynb`. The app loads the saved model and its preprocessing objects, displays the notebook's held-out evaluation, and supports individual and CSV predictions from the same 26 daily readings.

The saved evaluation artifacts contain real results from the executed notebook. The application does not retrain the model, select a new threshold, or use uploaded records as evaluation data. A prediction is a flag for review, not proof of electricity theft.

## Project files

```text
electricity-theft-app/
├── app.py
├── requirements.txt
├── pytest.ini
├── README.md
├── model/
│   ├── model.joblib
│   ├── metrics.json
│   ├── metadata.json
│   └── evaluation.json
├── templates/index.html
├── static/css/style.css
├── static/js/app.js
├── samples/demo_readings.csv
└── tests/test_app.py
```

`model.joblib` is copied from the actual saved notebook artifact. It contains GaussianNB, the fitted `StandardScaler`, training medians, raw reading-column order, feature names, the validation-selected threshold, and test metrics. `evaluation.json` contains confusion-matrix counts, class counts, and ROC and precision–recall curve coordinates from the actual held-out test labels and scores. The source dataset is not included.

## Run on Windows with VS Code

Use 64-bit Python 3.11 or 3.12. The saved model was produced with scikit-learn 1.9.1, which is pinned in `requirements.txt` for compatible loading.

Open the `electricity-theft-app` folder in VS Code. In the integrated PowerShell terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`, then continue with the install and run commands. Open <http://127.0.0.1:5000> in a browser. The backend health endpoint is <http://127.0.0.1:5000/api/health>.

## Run on macOS or Linux

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Chart.js and the optional web fonts are loaded from their public CDNs, so the dashboard's charts need an internet connection in the browser. The API and predictions run locally.

## Google Colab

Upload or clone this application folder into the Colab runtime, then install and start Flask:

```python
%cd /content/electricity-theft-app
!pip install -r requirements.txt
!python app.py > /tmp/electricity-theft-app.log 2>&1 &
from google.colab.output import serve_kernel_port_as_iframe
serve_kernel_port_as_iframe(5000)
```

The `model/` folder must be present in the runtime. If exporting directly from the notebook, set `ELECTRICITY_THEFT_APP_MODEL_DIR` to the app's `model` folder before running the notebook's final export cell.

## Regenerate artifacts from the notebook

The final export cell of `Electricity_Theft_Bayesian_Complete.ipynb` now writes the Flask artifacts to `electricity-theft-app/model/` by default. Run the notebook from the project root and execute its cells through the final export cell. It will create:

- `model.joblib` with the fitted classifier and the exact training-fitted preprocessing.
- `metrics.json` with held-out test metrics and class counts.
- `metadata.json` with the threshold, feature schema, dataset, and split sizes.
- `evaluation.json` with the held-out confusion matrix and ROC/precision–recall curve coordinates.

The notebook loads the real SGCC CSV from the working directory or the path in `SGCC_CSV_PATH`. If running the final export from a different directory, set `ELECTRICITY_THEFT_APP_MODEL_DIR` to the destination folder before executing it. The threshold remains the value selected on validation data; test labels are only used for final evaluation.

The app accepts **raw daily readings**, not the 12 engineered feature values. The backend repeats the notebook's log summaries, trend, daily-change features, median imputation, and saved scaling in the same order. CSV headers must match all 26 `daily_reading_columns` in `model/metadata.json`; an optional identifier or label column is not accepted.

## API

- `GET /api/health` reports server, model, and metric availability.
- `GET /api/metrics` returns the saved test metrics and model metadata.
- `GET /api/evaluation` returns the confusion matrix, test class counts, and genuine curve coordinates when exported.
- `POST /api/predict` accepts JSON with a `readings` array in notebook column order or an object keyed by all reading-column names.
- `POST /api/predict-csv` accepts a multipart upload with the field name `file`.

Invalid, missing, negative, and non-finite readings are rejected at the API boundary. The notebook handles missing historical readings during training; the browser API requires complete customer inputs so demonstrations cannot silently substitute missing values.

## Tests

Install the listed dependencies, activate the environment, and run:

```bash
pytest -q
```

The tests exercise the actual saved model bundle, valid and invalid manual inputs, valid and invalid CSV uploads, missing-artifact behavior, and the metric values rendered in the dashboard. There are no mocked model results in the shipped app or these integration tests.

## Manual demonstration

1. Start Flask and open the dashboard. Confirm the model status says **Model ready** and check the seven test metrics.
2. Review the confusion matrix, ROC and precision–recall curves, and test-set class distribution. These charts use the notebook's saved held-out results.
3. In **Manual readings**, choose **Use sample values** and run a prediction. Explain that the output is a model score compared with the validation-selected threshold.
4. Try an empty field or a negative value to see input validation.
5. Upload `samples/demo_readings.csv` to demonstrate batch prediction. It is a reference profile filled with the saved training medians, not a real customer row. For your own file, use the 26 headers listed in `model/metadata.json`, one customer per row, and non-negative readings. Search the results and download the predictions CSV.
6. In the presentation, describe false positives and class imbalance. Make clear that a positive flag means “review,” not confirmed theft.

## Troubleshooting

- **Model unavailable:** confirm `model/model.joblib` exists beside `app.py`, then check that it loads with the installed `scikit-learn` and `joblib` versions. Rerun the notebook's final export cell if the bundle is missing or stale.
- **Metrics or curves unavailable:** confirm `metrics.json` and `evaluation.json` exist and rerun the notebook through its final export cell. The app shows an unavailable message rather than inventing results.
- **Port 5000 is in use:** set `PORT=5001` before starting the app, then open <http://127.0.0.1:5001>.
- **Windows cannot install a package:** use 64-bit Python 3.10–3.12 and recreate the virtual environment from the project folder.
- **Dashboard has no charts:** allow the browser to load Chart.js from its CDN or use a browser session with internet access. Flask predictions do not depend on the CDN.
- **CSV schema error:** use the exact 26 daily-reading headers and do not include labels, customer IDs, or engineered features.
