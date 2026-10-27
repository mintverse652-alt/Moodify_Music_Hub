# MoodTune — Mood-Based Music Recommender

A beginner-friendly Streamlit web app that recommends songs using mood, genre, language, energy, TF-IDF and cosine similarity.

## Run on Windows

1. Extract the ZIP.
2. Open the extracted `Mood_Based_Music_Recommender` folder in VS Code.
3. Open Terminal in VS Code.
4. Run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

5. Streamlit will show a local URL, usually `http://localhost:8501`. Open it in Chrome.

If PowerShell blocks activation, run this in the same terminal:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
Then activate the environment again.

## Files
- `app.py` — app UI and recommendation algorithm
- `songs.csv` — demo song catalogue
- `requirements.txt` — Python libraries

## Note
This is a demo catalogue for a student project. It does not play full songs directly; the YouTube search links help find each track. Similarity scores are relative ranking scores, not verified music-quality ratings.
