import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MoodTune | Music Recommender", page_icon="🎧", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: radial-gradient(circle at 12% 5%, #34205e 0, #17142b 35%, #0c0b16 100%); color: #f8f7ff; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
.hero { padding: 30px 34px; border: 1px solid rgba(255,255,255,.12); border-radius: 26px;
background: linear-gradient(120deg, rgba(117,76,255,.35), rgba(255,86,164,.13)); margin-bottom: 24px; }
.kicker { color:#d6caff; letter-spacing:3px; font-size:12px; font-weight:700; }
.hero h1 { font-size:44px; line-height:1.15; margin:10px 0; color:white; }
.hero p { color:#c7c2db; font-size:15px; max-width:700px; }
.pill { display:inline-block; padding:7px 12px; margin:4px; border-radius:20px; background:#292442; color:#ddd4ff; font-size:12px; }
.song { background:rgba(255,255,255,.055); border:1px solid rgba(255,255,255,.1); border-radius:18px; padding:17px; margin:8px 0; }
.song-title { color:#fff; font-size:18px; font-weight:700; }
.artist { color:#c5bddf; font-size:13px; margin-top:3px; }
.meta { color:#b7acd8; font-size:12px; margin-top:10px; }
.small-label { color:#bcb2da; font-size:12px; letter-spacing:1.5px; font-weight:600; }
div.stButton > button { border-radius:12px; border:0; background:linear-gradient(90deg,#a78bfa,#f472b6); color:#130e24; font-weight:700; padding:12px 18px; }
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label { color:#eee8ff; }
section[data-testid="stSidebar"] { background:#151226; }
hr { border-color:rgba(255,255,255,.12); }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_songs():
    return pd.read_csv("songs.csv")

try:
    songs = load_songs()
except Exception:
    st.error("songs.csv file nahi mil rahi. app.py aur songs.csv ko same folder mein rakhein.")
    st.stop()

st.markdown("""
<div class="hero">
  <div class="kicker">YOUR PERSONAL SOUNDTRACK</div>
  <h1>Find your mood.<br><span style="color:#d8b4fe">Feel the music.</span> 🎧</h1>
  <p>MoodTune aapke mood, language aur music taste ke mutabiq songs suggest karta hai — powered by content similarity matching.</p>
  <span class="pill">✦ Mood-aware</span><span class="pill">♫ Similarity matching</span><span class="pill">♡ Made for your vibe</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.6], gap="large")
with col1:
    st.markdown("### ✨ Set your vibe")
    mood = st.selectbox("Aaj aapka mood kaisa hai?", ["Happy", "Sad", "Relaxed", "Energetic", "Romantic"])
    language = st.selectbox("Language", ["Any", "English", "Hindi", "Urdu", "Punjabi", "Instrumental", "Balochi/Urdu", "Urdu/Punjabi", "Telugu"])
    genres = sorted(songs["genre"].dropna().unique().tolist())
    genre = st.selectbox("Favourite genre", ["Any"] + genres)
    energy = st.select_slider("Energy level", options=["Chill", "Balanced", "Upbeat"], value="Balanced")
    count = st.slider("Recommendations", min_value=3, max_value=10, value=5)
    recommend = st.button("✨ Find my songs", use_container_width=True)

with col2:
    st.markdown("### 🎶 Your music mix")
    st.markdown("<p style='color:#bdb5d5'>Choose your vibe and discover tracks that fit the moment.</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class="song"><div class="small-label">HOW IT WORKS</div><div class="song-title" style="margin-top:8px">01 &nbsp; Pick a mood</div><div class="artist">Tell us how you're feeling right now.</div></div>
    <div class="song"><div class="song-title">02 &nbsp; Tune your taste</div><div class="artist">Choose a language, genre and energy level.</div></div>
    <div class="song"><div class="song-title">03 &nbsp; Discover your soundtrack</div><div class="artist">The recommender ranks songs using TF-IDF features and cosine similarity.</div></div>
    """, unsafe_allow_html=True)

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if recommend:
    filtered = songs.copy()
    # Use similarity features so results reflect mood, genre, language, energy and description.
    filtered["feature_text"] = (
        filtered["mood"].astype(str) + " " + filtered["genre"].astype(str) + " " +
        filtered["language"].astype(str) + " " + filtered["energy"].astype(str) + " " +
        filtered["description"].astype(str)
    )
    query_text = f"{mood} {genre if genre != 'Any' else ''} {language if language != 'Any' else ''} {energy} music songs"
    corpus = filtered["feature_text"].tolist() + [query_text]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    filtered["similarity"] = scores

    # Prioritize selected mood and explicitly selected filters while retaining similarity ranking.
    filtered.loc[filtered["mood"].str.lower() == mood.lower(), "similarity"] += 0.30
    if language != "Any":
        filtered.loc[filtered["language"].str.lower() == language.lower(), "similarity"] += 0.20
    if genre != "Any":
        filtered.loc[filtered["genre"].str.lower() == genre.lower(), "similarity"] += 0.20
    if energy != "Balanced":
        filtered.loc[filtered["energy"].str.lower() == energy.lower(), "similarity"] += 0.12

    results = filtered.sort_values("similarity", ascending=False).head(count).copy()
    st.session_state.recommendations = (results, mood, language, genre, energy)

if st.session_state.recommendations is not None:
    results, selected_mood, selected_language, selected_genre, selected_energy = st.session_state.recommendations
    st.markdown("---")
    st.markdown(f"### 💜 Recommended for your **{selected_mood.lower()}** mood")
    st.caption("Recommendations are ranked by content similarity. Songs are demo catalogue entries; playback is not embedded.")
    cols = st.columns(2)
    for idx, (_, row) in enumerate(results.iterrows()):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="song">
              <div class="small-label">TRACK {idx+1:02d} &nbsp; · &nbsp; {row['mood'].upper()}</div>
              <div class="song-title" style="margin-top:8px">♫ &nbsp; {row['song']}</div>
              <div class="artist">{row['artist']}</div>
              <div class="meta">{row['genre']} &nbsp; • &nbsp; {row['language']} &nbsp; • &nbsp; {row['energy']}</div>
              <div style="color:#d2c8ec;font-size:12px;margin-top:10px">{row['description']}</div>
              <div style="margin-top:12px;color:#f0abfc;font-size:12px">Match score: {min(100, int(row['similarity']*100))}%</div>
            </div>
            """, unsafe_allow_html=True)
            search_query = str(row["song"]) + " " + str(row["artist"])
            st.link_button("▶ Search song on YouTube", "https://www.youtube.com/results?search_query=" + __import__("urllib.parse").parse.quote(search_query), use_container_width=True)


st.markdown("""
<hr>
<div style="text-align:center; padding:15px 0;">
    <h4>🎵 MoodTune — Music for Every Mood</h4>
    <p>Your mood, your music, your moment.</p>
    <small>Made with ❤️ for music lovers</small>
    <br>
    <small>© 2026 MoodTune. All rights reserved.</small>
</div>
""", unsafe_allow_html=True)
