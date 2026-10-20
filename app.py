from pathlib import Path
import json, random, numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go, streamlit as st

st.set_page_config("VibeNet", layout="wide")
PAL = px.colors.qualitative.Bold
COLS = ["id", "country", "tribe", "friends", "pagerank", "betweenness", "cross", "taste", "liked", "x", "y"]

@st.cache_data
def load():
    data_file = Path(__file__).resolve().parent / "vibenet.json"
with data_file.open("r", encoding="utf-8") as f:
    D = json.load(f)
df = pd.DataFrame(D["nodes"], columns=COLS)
df["tribe"] = df.tribe.map(lambda i: f"Tribe {i+1}" if i < 8 else "Other"); df["country"] = "Country " + df.country.astype(str).str.zfill(2)
 return D, df
D, df = load(); P = D["props"]

st.sidebar.header("Controls")
color_by = st.sidebar.radio("Color by", ["tribe", "country"])
size_by = st.sidebar.radio("Size by", ["pagerank", "friends", "betweenness"])
top_n = st.sidebar.slider("Listeners shown (top by PageRank)", 100, 1500, 500, 100)
min_f = st.sidebar.slider("Min friends", 0, 50, 0)
cset = st.sidebar.multiselect("Countries", sorted(df.country.unique()), default=sorted(df.country.unique()))
q = st.sidebar.text_input("Search listener ID")

st.title("VibeNet")
st.caption("Do friends listen to the same music? Listeners are vertices; a mutual follow on Last.fm is an edge.")
c = st.columns(6)
for col, (k, v) in zip(c, [("Listeners", f"{P['vertices']:,}"), ("Friendships", f"{P['edges']:,}"), ("Density", P["density"]),
        ("Components", P["components"]), ("Taste tribes", P["communities"]), ("Modularity", P["modularity"])]): col.metric(k, v)
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["About", "Network", "Tribes", "Taste", "Analytics", "Insights"])

with tab0:
    st.markdown(f"""**Purpose:** find out whether friends share music taste, which friend groups ("taste tribes") exist, and who connects them.

**Dataset:** LastFM Asia social network (Rozemberczki & Sarkar, 2020; SNAP). Each vertex is a Last.fm user from Asia, each edge a mutual follow. Each user has a list of liked artists ({P['artists']:,} artists in total) and a country code. The data was collected in March 2020. The dataset gives artist and country IDs only, not names.

**How to read it:** close dots are friends; color is the tribe (friend group found by Louvain) or country; bigger dots are more central. Hover for details, search a listener ID, or filter by country.""")
    st.caption("Source: snap.stanford.edu/data/feather-lastfm-social.html (CC BY 4.0 on UCI)")

with tab1:
    st.caption("Each dot is a listener and each line a mutual follow; tribes form clusters and bridges link them.")
    sub = df[df.x.notna() & (df.friends >= min_f) & df.country.isin(cset)].nlargest(top_n, "pagerank")
    if sub.empty: st.warning("No listeners match these filters."); st.stop()
    vis = set(sub.id); pos = dict(zip(df.id, zip(df.x, df.y)))
    E = [(a, b) for a, b in D["edges"] if a in vis and b in vis]
    if len(E) > 6000: random.seed(1); E = random.sample(E, 6000); st.caption("Showing 6,000 random friendships for speed.")
    focus = None
    if q.strip().isdigit() and int(q) in vis: i = int(q); focus = {i} | {b if a == i else a for a, b in E if i in (a, b)}
    elif q.strip(): st.sidebar.warning("ID not in the current view.")
    ex, ey = [], []
    for a, b in E: ex += [pos[a][0], pos[b][0], None]; ey += [pos[a][1], pos[b][1], None]
    v = sub[size_by].astype(float); keys = sorted(sub[color_by].unique()); cmap = {k: PAL[i % len(PAL)] for i, k in enumerate(keys)}
    fig = go.Figure([go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=.4, color="#666"), opacity=.3, hoverinfo="none"),
        go.Scatter(x=sub.x, y=sub.y, mode="markers", hoverinfo="text",
            text=[f"Listener {r.id}<br>{r.tribe} | {r.country}<br>Friends: {r.friends}<br>Taste match with friends: {r.taste:.2f}" for r in sub.itertuples()],
            marker=dict(size=5 + 22 * np.sqrt(v / v.max()), color=[cmap[k] for k in sub[color_by]],
                        opacity=[1 if focus is None or i in focus else .08 for i in sub.id]))])
    fig.update_layout(template="plotly_dark", height=650, showlegend=False, dragmode="pan", xaxis=dict(visible=False),
                      yaxis=dict(visible=False), margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
    st.markdown("  ".join(f"<span style='color:{c}'>●</span> {k}" for k, c in cmap.items()), unsafe_allow_html=True)

with tab2:
    st.caption("The biggest taste tribes: size, main country code, how much of the tribe shares it, and signature artist IDs.")
    T = pd.DataFrame(D["tribes"]); T["signature_artists"] = T.signature_artists.map(lambda a: ", ".join(f"#{x}" for x in a))
    st.dataframe(T, hide_index=True)
    st.caption("Tribes vs countries: bright cells mean a tribe is mostly from one country, so friend groups follow geography.")
    labs = [f"Tribe {i+1}" for i in range(8)] + ["Other"]
    st.plotly_chart(px.imshow(np.array(D["tribe_country"]), x=[f"C{i:02d}" for i in range(18)], y=labs, template="plotly_dark",
        color_continuous_scale="Viridis"), width="stretch")

with tab3:
    st.caption("Taste overlap: the share of liked artists two listeners have in common; friends sit higher than random pairs.")
    H = D["taste_hist"]; hd = pd.DataFrame({"overlap": H["bins"] * 2, "listener pairs": H["friends"] + H["strangers"],
        "pair type": ["friends"] * len(H["bins"]) + ["random strangers"] * len(H["bins"])})
    st.plotly_chart(px.bar(hd, x="overlap", y="listener pairs", color="pair type", barmode="group", template="plotly_dark"), width="stretch")
    st.caption("Each listener's average taste overlap with their friends, against how many friends they have.")
    st.plotly_chart(px.scatter(df, x="friends", y="taste", color="tribe", log_x=True, opacity=.5, template="plotly_dark"), width="stretch")

with tab4:
    st.caption("Degree distribution: most listeners have a few friends while a handful have over a hundred.")
    dd = pd.DataFrame({"friends": range(len(D["deg"])), "listeners": D["deg"]}).query("listeners > 0 and friends > 0")
    st.plotly_chart(px.scatter(dd, x="friends", y="listeners", log_x=True, log_y=True, template="plotly_dark"), width="stretch")
    st.caption("Most central listeners (PageRank), the likely hubs.")
    st.dataframe(df.nlargest(10, "pagerank")[["id", "tribe", "country", "friends", "pagerank"]], hide_index=True)
    st.caption("Top bridges: listeners on many shortest paths, with the share of friends outside their own tribe.")
    st.dataframe(df.nlargest(10, "betweenness")[["id", "tribe", "friends", "cross", "betweenness"]], hide_index=True)
    st.caption("Most-liked artists (IDs): how many of the listeners like them.")
    st.dataframe(pd.DataFrame(D["top_artists"], columns=["artist ID", "listeners"]), hide_index=True)

with tab5:
    st.caption("Plain-language findings computed from this network.")
    for line in D["insights"]: st.markdown(f"- {line}")
