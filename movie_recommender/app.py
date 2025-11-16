# app.py -- Streamlit Movie & Series Recommender (no API key required)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import difflib
from collections import defaultdict
from functools import lru_cache

st.set_page_config(page_title="Movie Recommender", layout="wide")

# ------------------ CONFIG / PATHS ------------------
MOVIES_CSV = "movies_filtered_enriched.csv"   # from Colab Cell 12
USER_FACTORS_NPY = "user_factors.npy"
ITEM_FACTORS_NPY = "item_factors.npy"
USER_MAP_PKL = "user_map.pkl"
ITEM_MAP_PKL = "item_map.pkl"
REV_ITEM_MAP_PKL = "rev_item_map.pkl"

# ------------------ LOAD ARTIFACTS (with graceful fallback) ------------------
@st.cache_data
def load_movies(path=MOVIES_CSV):
    if os.path.exists(path):
        df = pd.read_csv(path)
        # ensure expected cols exist
        for c in ["movieId","title","genres"]:
            if c not in df.columns:
                df[c] = ""
        # optional columns
        if 'language' not in df.columns:
            df['language'] = None
        if 'is_series' not in df.columns:
            df['is_series'] = False
        return df
    else:
        # minimal fallback (so app still runs)
        st.warning(f"{path} not found. App will run in demo/content-only mode.")
        df = pd.DataFrame(columns=["movieId","title","genres","language","is_series"])
        return df

@st.cache_data
def load_factors():
    """Try to load user/item factors and mappings. Return None for missing pieces."""
    missing = []
    for p in [USER_FACTORS_NPY, ITEM_FACTORS_NPY, USER_MAP_PKL, ITEM_MAP_PKL, REV_ITEM_MAP_PKL]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        return None, None, None, None, None
    try:
        user_f = np.load(USER_FACTORS_NPY, allow_pickle=True)
        item_f = np.load(ITEM_FACTORS_NPY, allow_pickle=True)
        with open(USER_MAP_PKL,"rb") as f: user_map = pickle.load(f)
        with open(ITEM_MAP_PKL,"rb") as f: item_map = pickle.load(f)
        with open(REV_ITEM_MAP_PKL,"rb") as f: rev_item_map = pickle.load(f)
        return user_f, item_f, user_map, item_map, rev_item_map
    except Exception as e:
        st.error("Failed to load factor artifacts: " + str(e))
        return None, None, None, None, None

movies = load_movies()
user_factors, item_factors, user_map, item_map, rev_item_map = load_factors()
has_cf = item_factors is not None and item_map is not None and rev_item_map is not None

# ------------------ UTILITIES ------------------
@st.cache_data
def build_search_lists():
    titles = movies['title'].astype(str).tolist()
    lower = [t.lower() for t in titles]
    return titles, lower

TITLES, TITLES_LOWER = build_search_lists()

_series_re = re.compile(r"\b(tv|series|season|episode|miniseries|mini-series|s\d{2})\b", re.I)

def detect_series(title):
    return bool(_series_re.search(title)) if isinstance(title, str) else False

def find_matches(query, topn=6, cutoff=0.4):
    q = str(query).strip().lower()
    if not q:
        return []
    # substring matches first
    subs = []
    for orig, low in zip(TITLES, TITLES_LOWER):
        if q in low:
            subs.append(orig)
            if len(subs) >= topn:
                return subs
    # fallback to difflib
    unique_titles = list(dict.fromkeys(TITLES))  # preserve order, unique
    close = difflib.get_close_matches(query, unique_titles, n=topn, cutoff=cutoff)
    merged = []
    for t in subs + close:
        if t not in merged:
            merged.append(t)
    return merged[:topn]

# Simple content neighbor (genre overlap + title token overlap)
@st.cache_data
def content_neighbors_by_seed(seed_mid, topk=100):
    if seed_mid not in movies['movieId'].values:
        return []
    seed_row = movies[movies['movieId'] == seed_mid].iloc[0]
    seed_genres = str(seed_row.get('genres','')).lower().split('|')
    seed_tokens = set([t.strip() for g in seed_genres for t in g.split() if t.strip()])
    scores = []
    # vectorized-ish loop
    for _, r in movies.iterrows():
        mid = int(r['movieId']) if 'movieId' in r else None
        if mid == seed_mid or pd.isna(mid):
            continue
        g = str(r.get('genres','')).lower().split('|')
        gset = set([t.strip() for gg in g for t in gg.split() if t.strip()])
        overlap = len(seed_tokens.intersection(gset))
        if overlap > 0:
            # small title token boost
            title_tokens = set(r['title'].lower().split())
            score = overlap + 0.1 * len(title_tokens.intersection(seed_tokens))
            scores.append((mid, float(score)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]

def get_item_item_neighbors(seed_mid, topk=100):
    if not has_cf or seed_mid not in item_map:
        return []
    col = item_map[seed_mid]
    seed_vec = item_factors[col]
    sims = item_factors.dot(seed_vec)
    sims[col] = -np.inf
    k = min(topk, len(sims)-1)
    if k <= 0:
        return []
    top_idx = np.argpartition(-sims, range(k))[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(int(rev_item_map[idx]), float(sims[idx])) for idx in top_idx]

def aggregate_from_seeds(seed_mids, mode='hybrid', topn=20):
    scores = defaultdict(float)
    for mid in seed_mids:
        # content neighbors
        for cmid, sc in content_neighbors_by_seed(mid, topk=topn*5):
            scores[cmid] += float(sc)
        # cf neighbors
        if mode == 'hybrid' and has_cf:
            for cmid, sc in get_item_item_neighbors(mid, topk=topn*3):
                scores[cmid] += 0.7 * float(sc)
    # remove seeds
    for s in seed_mids:
        scores.pop(s, None)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:topn]

# ------------------ UI ------------------
st.title("🎬 Movie & Series Recommender (Offline)")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Inputs")
    st.write("Type movies you watched (comma-separated). Use the search box to find titles.")
    query = st.text_input("Search a title (press Enter):")
    watched_text = st.text_area("Or paste watched movie titles (comma-separated):", height=100)
    lang_pref = st.text_input("Preferred language (e.g. Hindi) — optional")
    series_pref = st.selectbox("Series preference", ["Both", "Only movies", "Only TV series"])
    mode = st.radio("Mode", ["Hybrid (recommended)", "Content-only"])
    topn = st.number_input("Number of recommendations", min_value=1, max_value=50, value=10, step=1)
    if st.button("Recommend"):
        # build seed titles from watched_text or query
        seed_titles = []
        if watched_text and watched_text.strip():
            seed_titles = [s.strip() for s in watched_text.split(',') if s.strip()]
        elif query and query.strip():
            seed_titles = [query.strip()]
        else:
            st.error("Please enter at least one movie (search or paste list).")
            st.stop()

        # resolve titles -> movieIds
        seed_mids = []
        unresolved = []
        for t in seed_titles:
            matches = find_matches(t, topn=6)
            if not matches:
                unresolved.append(t)
                continue
            chosen = matches[0]
            rows = movies[movies['title'] == chosen]
            if len(rows) == 0:
                # fallback contains search
                rows = movies[movies['title'].str.contains(chosen, case=False, na=False)]
            if len(rows) == 0:
                unresolved.append(t)
                continue
            mid = int(rows.iloc[0]['movieId'])
            seed_mids.append(mid)

        if unresolved:
            st.warning("Could not match: " + ", ".join(unresolved))

        if len(seed_mids) == 0:
            st.error("No valid seed movies found. Try different titles.")
            st.stop()

        mode_name = 'hybrid' if mode.startswith('Hybrid') else 'content'
        recs = aggregate_from_seeds(seed_mids, mode=mode_name, topn=topn*3)

        # apply filters
        filtered = []
        for mid, score in recs:
            row = movies[movies['movieId'] == mid]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            # language
            if lang_pref and pd.notna(r.get('language','')) and lang_pref.strip().lower() not in str(r['language']).lower():
                continue
            # series filter
            is_series = bool(r.get('is_series', False)) or detect_series(r['title'])
            if series_pref == "Only movies" and is_series:
                continue
            if series_pref == "Only TV series" and not is_series:
                continue
            filtered.append((mid, score))

        if not filtered:
            st.warning("No recommendations after filters. Relax filters or try other seeds.")
            st.stop()

        filtered = filtered[:topn]
        st.success(f"Top {len(filtered)} recommendations")
        for rank, (mid, sc) in enumerate(filtered, start=1):
            row = movies[movies['movieId']==mid].iloc[0]
            meta = []
            if 'genres' in row and pd.notna(row['genres']) and row['genres'] != '':
                meta.append(row['genres'])
            if 'language' in row and pd.notna(row['language']):
                meta.append(str(row['language']))
            if 'is_series' in row and row['is_series']:
                meta.append("Series")
            meta_s = " | ".join(meta) if meta else ""
            st.markdown(f"**{rank}. {row['title']}**  —  score: {sc:.3f}")
            if meta_s:
                st.write(meta_s)

with col2:
    st.subheader("Search results")
    if query and query.strip():
        matches = find_matches(query, topn=15)
        if not matches:
            st.write("No matches found.")
        else:
            for m in matches:
                rows = movies[movies['title'] == m]
                if len(rows) > 0:
                    r = rows.iloc[0]
                    s = f"{r['title']} (movieId: {int(r['movieId'])})"
                    if 'genres' in r and pd.notna(r['genres']) and r['genres'] != '':
                        s += f" — {r['genres']}"
                    st.write(s)
    else:
        st.info("Type a title above and press Enter to see matching movies.")

# ------------------ Footer / debug ------------------
st.markdown("---")
st.write("Debug info:")
file_status = {
    "movies_csv": os.path.exists(MOVIES_CSV),
    "user_factors": os.path.exists(USER_FACTORS_NPY),
    "item_factors": os.path.exists(ITEM_FACTORS_NPY),
    "user_map": os.path.exists(USER_MAP_PKL),
    "item_map": os.path.exists(ITEM_MAP_PKL),
    "rev_item_map": os.path.exists(REV_ITEM_MAP_PKL),
    "cf_enabled": has_cf
}
st.json(file_status)
