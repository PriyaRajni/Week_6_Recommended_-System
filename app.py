import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Mini Movie Recommender", page_icon="🎬")

# --- Tiny, embedded dataset (feel free to edit/expand) ---
MOVIES = [
    {"title": "The Matrix", "genres": "Action|Sci-Fi", "year": 1999,
     "description": "A hacker discovers the true nature of reality and leads a rebellion."},
    {"title": "Inception", "genres": "Action|Sci-Fi|Thriller", "year": 2010,
     "description": "A thief enters dreams to plant ideas; layered heist across dreamscapes."},
    {"title": "Interstellar", "genres": "Adventure|Drama|Sci-Fi", "year": 2014,
     "description": "Explorers travel through a wormhole seeking a new home for humanity."},
    {"title": "The Dark Knight", "genres": "Action|Crime|Drama", "year": 2008,
     "description": "Batman faces the Joker in a gritty battle for Gotham's soul."},
    {"title": "Pulp Fiction", "genres": "Crime|Drama", "year": 1994,
     "description": "Interwoven tales in LA’s underworld; nonlinear storytelling classic."},
    {"title": "The Shawshank Redemption", "genres": "Drama", "year": 1994,
     "description": "Hope and friendship in a prison; a story of perseverance."},
    {"title": "Spirited Away", "genres": "Animation|Fantasy|Adventure", "year": 2001,
     "description": "A girl enters a spirit world bathhouse; coming-of-age fantasy."},
    {"title": "The Godfather", "genres": "Crime|Drama", "year": 1972,
     "description": "Epic saga of the Corleone family and the cost of power."},
    {"title": "Toy Story", "genres": "Animation|Comedy|Family", "year": 1995,
     "description": "Toys come alive when humans aren’t around; unlikely friendship."},
    {"title": "Mad Max: Fury Road", "genres": "Action|Adventure|Sci-Fi", "year": 2015,
     "description": "High-octane chase across a post-apocalyptic wasteland."},
]
df = pd.DataFrame(MOVIES)

# --- Build content features ---
def build_model(dataframe: pd.DataFrame):
    # Create a single text field per movie
    text = (
        dataframe["title"].fillna("") + " " +
        dataframe["genres"].str.replace("|", " ", regex=False).fillna("") + " " +
        dataframe["description"].fillna("")
    )
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))
    X = vec.fit_transform(text)
    sim = cosine_similarity(X)   # movie-to-movie similarity
    np.fill_diagonal(sim, 0.0)   # avoid recommending itself
    return vec, X, sim

VEC, X, SIM = build_model(df)

def recommend_by_title(title: str, k: int = 5) -> pd.DataFrame:
    # Find index for the title
    matches = df.index[df["title"] == title].tolist()
    if not matches:
        return pd.DataFrame(columns=["title", "genres", "year", "score"])
    idx = matches[0]
    scores = SIM[idx]
    order = scores.argsort()[::-1]  # descending
    order = [i for i in order if i != idx][:k]
    out = df.iloc[order].copy()
    out["score"] = scores[order]
    return out.sort_values("score", ascending=False)

def recommend_from_multiple(titles, k: int = 5) -> pd.DataFrame:
    if not titles:
        return pd.DataFrame(columns=["title", "genres", "year", "score"])

    # find indices for the selected titles
    valid_titles = set(df["title"])
    idxs = [df.index[df["title"] == t][0] for t in titles if t in valid_titles]
    if not idxs:
        return pd.DataFrame(columns=["title", "genres", "year", "score"])

    # Sparse mean -> np.matrix; convert to ndarray
    centroid = X[idxs].mean(axis=0)                    # 1 x n_features (np.matrix)
    centroid = np.asarray(centroid).reshape(1, -1)     # make it a 2D ndarray

    sims = cosine_similarity(centroid, X).ravel()
    # exclude the selected titles themselves
    exclude = set(idxs)
    order = [i for i in sims.argsort()[::-1] if i not in exclude][:k]
    out = df.iloc[order].copy()
    out["score"] = sims[order]
    return out.sort_values("score", ascending=False)

# --- UI ---
st.title("🎬 Mini Movie Recommender (Content-Based)")
st.caption("TF-IDF over title + genres + description, ranked by cosine similarity.")

tab1, tab2 = st.tabs(["Single Favorite → Similar Movies", "Multiple Favorites (Blended)"])
#tab1  = st.tabs(["Single Favorite → Similar Movies"])
with tab1:
    st.subheader("Pick one favorite movie")
    sel = st.selectbox("Favorite title:", df["title"].tolist(), index=0)
    k = st.slider("How many recommendations?", 1, 10, 5)
    if st.button("Recommend (Single)"):
        recs = recommend_by_title(sel, k=k)
        st.dataframe(recs[["title", "genres", "year", "score"]])

with tab2:
    st.subheader("Pick multiple favorites to blend their 'taste'")
    multi = st.multiselect("Favorite titles:", df["title"].tolist(), default=["The Matrix","Inception"])
    k2 = st.slider("How many recommendations?", 1, 10, 5, key="k2")
    if st.button("Recommend (Blended)"):
        recs = recommend_from_multiple(multi, k=k2)
        st.dataframe(recs[["title", "genres", "year", "score"]])


st.markdown("---")
st.markdown("**How it works:** We vectorize each movie’s text (title + genres + description) with TF-IDF, then rank by cosine similarity to your selection(s).")
