# QuickBite — Complete Food Delivery Time Predictor

This project includes a browser UI, FastAPI backend, and the supplied trained model pipeline.

## Files
- `index.html` — frontend form and prediction results
- `main.py` — FastAPI backend (`/`, `/health`, `/predict`, `/docs`)
- `model_pipeline.joblib` — supplied trained ML pipeline
- `requirements.txt` — pinned Python dependencies
- `vercel.json` — Vercel Python function routing

## Run locally on Windows

1. Install Python 3.11 or 3.12.
2. Open Command Prompt in this folder.
3. Create and activate a virtual environment:

   ```bat
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the backend:

   ```bat
   uvicorn main:app --reload
   ```

6. Open `http://127.0.0.1:8000` in your browser.
7. Test the API status at `http://127.0.0.1:8000/health` and interactive API docs at `http://127.0.0.1:8000/docs`.

## Deploy to Vercel

Run these commands from this folder after installing the Vercel CLI and signing in:

```bat
vercel
```

Follow the prompts, then test the preview URL. If it works:

```bat
vercel --prod
```

Then open the deployed site's `/health` endpoint. It should return `"model_loaded": true`.

## Important model note

The supplied model appears to have been serialized with scikit-learn 1.6.1, so this project pins `scikit-learn==1.6.1`. Loading serialized scikit-learn objects with a different version can fail or behave unexpectedly. The model is the one supplied in the original ZIP; this project does not fabricate a new trained model.

The categorical options in the form are common delivery dataset values. If your original training dataset uses different category labels, choose the exact labels used by that dataset/model.
