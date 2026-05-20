# Insurance Dashboard

A small Flask app that predicts insurance cost based on user inputs (age, BMI, number of children). The repository includes a minimal UI and a pre-trained scikit-learn model saved as `insurance_model.pkl`.

**Quick overview**

- `app.py`: Flask application that loads the trained model and serves the web UI.
- `templates/index.html`: Main HTML template (form + result card). I updated this to a modern, responsive, and classy UI.
- `insurance_model.pkl`: Pre-trained model used to make predictions (loaded with `joblib`).
- `insurance_cleaned.csv`, `Project_ML.ipynb`: data and notebook used during model development.

## Requirements

- Python 3.8+ (Windows instructions shown)
- The following Python packages:

```bash
pip install flask numpy joblib
```

(If your model was trained with scikit-learn, also install `scikit-learn` to retrain or inspect the model.)

## Run locally

1. (Optional) Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

If you don't have `requirements.txt`, install directly:

```powershell
pip install flask numpy joblib
```

3. Start the app:

```powershell
python app.py
```

4. Open `http://127.0.0.1:5000/` in your browser.

## How it works

- When the Flask app starts, `app.py` loads the model file `insurance_model.pkl` using `joblib.load`.
- The root route (`/`) renders the `index.html` template which contains a small form for `age`, `bmi`, and `children`.
- Submitting the form posts to `/predict`. The server reads the form values, packages them into a NumPy array, and calls `model.predict(...)`.
- The prediction result is returned to the template as `prediction_text` and displayed in the result card. The UI shows a subtle confetti animation when a prediction is present.

## Notes & Safety

- The prediction is a statistical estimate and for demo purposes only — do not treat it as financial or medical advice.
- If `insurance_model.pkl` is missing, `app.py` will raise an error at startup. Keep the model in the repository root or update the path in `app.py`.

## Suggested improvements

- Move CSS and JS into `static/` and reference them from the template for better caching and maintainability.
- Add `requirements.txt` (e.g., `flask`, `numpy`, `joblib`, `scikit-learn`) and pin versions.
- Add input validation on the server side and nicer client-side validation.
- Add unit tests around the prediction function and a small integration test for the Flask app.
- Add charts (Chart.js) showing historical distributions or how features affect predictions.

---

If you want, I can (pick one):
- Add a `requirements.txt` and pin versions.
- Move styles and scripts to `static/` files.
- Run the app here and verify the UI loads.
