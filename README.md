# VibeNet: do friends listen to the same music?
DSE3150 Graph Visualization project.

Purpose: show whether friends share music taste, which friend groups ("taste tribes") exist, and who connects them.
Data: LastFM Asia social network (Rozemberczki & Sarkar, 2020, FEATHER paper), collected from the public Last.fm API in March 2020.
7,624 listeners, 27,806 mutual follows, 7,842 distinct artists liked, 18 country codes. CC BY 4.0.
Source: snap.stanford.edu/data/feather-lastfm-social.html (files also in github.com/benedekrozemberczki/FEATHER).
Note: the dataset gives artist and country IDs only, not names.

Graph: undirected, simple, finite, 1 connected component, 0 isolated vertices, density 0.00096.

Run: pip install -r requirements.txt, then streamlit run app.py
Rebuild data: python preprocess.py (downloads the raw files, computes PageRank, betweenness, Louvain tribes, taste overlap, layout)
Deploy: push app.py, requirements.txt and data/vibenet.json (keep the data/ folder) to GitHub, then share.streamlit.io with app.py

Pipeline: raw CSV/JSON -> pandas/NetworkX metrics -> data/vibenet.json -> Plotly views -> Streamlit tabs
Design: Gestalt proximity (layout), similarity (color), connectedness (edges); overload control by top-N, filters and dimming; group technique: Louvain tribes and the tribe-by-country matrix.
