import streamlit as st
import pandas as pd
from pathlib import Path
import re, unicodedata
from urllib.parse import quote_plus
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster

# ---------- Config ----------
st.set_page_config(page_title="Carte entreprises par NAF (départements + méthaniseurs)", layout="wide")
st.title("🗺️ Carte entreprises par NAF — sélection par département + couche Méthaniseurs")

ROOT = Path(__file__).parent
DIR_ENT = ROOT / "data" / "entreprises"
DIR_METH = ROOT / "data" / "methaniseurs"
METH_FILE = DIR_METH / "methaniseurs.csv"   # optionnel

# Colonnes SIRENE (selon ton fichier)
COLS = {
    "siret": "siret",
    "etat": "etatAdministratifEtablissement",
    "naf": "activitePrincipaleEtablissement",
    "enseigne1": "enseigne1Etablissement",
    "denom": "denominationUsuelleEtablissement",
    "lon": "longitude",
    "lat": "latitude",
    "adresse": "geo_adresse",
    "cp": "codePostalEtablissement",
    "commune": "libelleCommuneEtablissement",
    "siege": "etablissementSiege",
}

# ---------- Utils ----------
def _read_csv_auto(path):
    # tente séparateur auto, puis ";" si échec (certains exports)
    try:
        return pd.read_csv(path, sep=None, engine="python", low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=";", low_memory=False)

def _norm(s: str):
    if not isinstance(s, str): return ""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return re.sub(r"[^0-9A-Za-z ]+", " ", s).strip()

def build_pj_link(nom, adresse, cp, commune):
    terms = " ".join([str(x) for x in [nom, adresse, cp, commune] if x])
    return f"https://www.pagesjaunes.fr/recherche/{quote_plus(terms)}"

def build_gmaps_link(lat, lon, nom=None, adresse=None):
    if pd.notna(lat) and pd.notna(lon):
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    q = " ".join([str(x) for x in [nom, adresse] if x])
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(q)}"

# ---------- Découverte des fichiers ----------
@st.cache_data(show_spinner=False)
def list_dept_files():
    files = sorted(DIR_ENT.glob("geo_siret_*.csv"))
    out = []
    for f in files:
        m = re.search(r"geo_siret_([0-9]{2}|2A|2B)", f.name, re.IGNORECASE)
        code = m.group(1).upper() if m else f.stem
        out.append((code, f))
    return out

# ---------- Chargement multi-départements ----------
@st.cache_data(show_spinner=True)
def load_departments(selected_deps):
    frames = []
    for dep_code, f in list_dept_files():
        if dep_code in selected_deps:
            df = _read_csv_auto(f)
            df["__dep__"] = dep_code
            df["__source__"] = f.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ---------- Charge méthaniseurs (optionnel) ----------
@st.cache_data(show_spinner=False)
def load_methaniseurs():
    if not METH_FILE.exists():
        return None
    dfm = _read_csv_auto(METH_FILE)
    # colonnes attendues: nom, adresse, latitude, longitude (tolère variantes simples)
    def find_col(cands):
        for c in cands:
            if c in dfm.columns: return c
        return None
    c_lat = find_col(["latitude","lat","y"])
    c_lon = find_col(["longitude","lon","x"])
    c_nom = find_col(["nom","name","denomination","enseigne"])
    c_addr= find_col(["adresse","address","geo_adresse"])
    if not c_lat or not c_lon:
        return None
    dfm["lat"] = pd.to_numeric(dfm[c_lat].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    dfm["lon"] = pd.to_numeric(dfm[c_lon].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    dfm = dfm[dfm["lat"].notna() & dfm["lon"].notna()].copy()
    if c_nom and "nom" not in dfm.columns: dfm.rename(columns={c_nom:"nom"}, inplace=True)
    if c_addr and "adresse" not in dfm.columns: dfm.rename(columns={c_addr:"adresse"}, inplace=True)
    dfm["gmaps_url"] = dfm.apply(lambda r: build_gmaps_link(r.get("lat"), r.get("lon"), r.get("nom"), r.get("adresse")), axis=1)
    keep = [c for c in ["nom","adresse","lat","lon","gmaps_url"] if c in dfm.columns]
    return dfm[keep].copy()

# ---------- UI: Départements ----------
deps = list_dept_files()
dep_codes = [d for d,_ in deps]
if not deps:
    st.error("Aucun fichier trouvé dans data/entreprises/ (ex: geo_siret_01.csv).")
    st.stop()

st.subheader("1) Sélection des départements")
col1, col2 = st.columns([2,1])
with col1:
    all_deps = st.checkbox("Sélectionner tous les départements", value=True)
if all_deps:
    selected_deps = dep_codes
else:
    selected_deps = st.multiselect("Choisis un ou plusieurs départements", options=dep_codes, default=dep_codes[:5])

# ---------- Chargement ----------
df_raw = load_departments(selected_deps)
st.write(f"📥 Lignes chargées: **{len(df_raw):,}** (dep: {len(selected_deps)})")

# ---------- Garde colonnes utiles ----------
missing = [c for c in COLS.values() if c not in df_raw.columns]
if missing:
    st.warning("⚠️ Colonnes manquantes (vérifie l’export): " + ", ".join(missing))

# Nettoyage état administratif
df = df_raw.copy()
if COLS["etat"] in df.columns:
    df = df[df[COLS["etat"]].astype(str).str.upper().str.startswith("A")]  # "A" = actif

# Champs d'affichage
def coalesce_name(row):
    vals = [row.get(COLS["denom"]), row.get(COLS["enseigne1"])]
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
    return ""

# NAF list + filtre
if COLS["naf"] in df.columns:
    naf_clean = df[COLS["naf"]].astype(str).str.upper().str.replace(r"[^0-9A-Z.]", "", regex=True)
    codes_naf = sorted(naf_clean.unique())
    st.subheader("2) Codes NAF")
    defaults = [c for c in ["10.51C","47.29Z","1051C","4729Z"] if c in codes_naf]
    naf_select = st.multiselect("Choisis un ou plusieurs codes NAF à afficher",
                                options=codes_naf,
                                default=defaults or codes_naf[:10])
    if naf_select:
        df = df[naf_clean.isin(naf_select)]
else:
    st.info("Colonne NAF introuvable — aucun filtre NAF appliqué.")
    naf_select = []

# Optional: si tu veux ne garder que les sièges
only_siege = st.checkbox("Ne garder que les sièges (etablissementSiege=1)", value=False)
if only_siege and COLS["siege"] in df.columns:
    df = df[df[COLS["siege"]].astype(str).isin(["1","True","true","O","Oui"])]

# Coordonnées
if COLS["lat"] not in df.columns or COLS["lon"] not in df.columns:
    st.error("Colonnes latitude/longitude absentes. Tes CSV doivent être géolocalisés.")
    st.stop()

df["lat"] = pd.to_numeric(df[COLS["lat"]].astype(str).str.replace(",", ".", regex=False), errors="coerce")
df["lon"] = pd.to_numeric(df[COLS["lon"]].astype(str).str.replace(",", ".", regex=False), errors="coerce")
df = df[df["lat"].notna() & df["lon"].notna()].copy()

# Dataframe final pour export & carte
ent = pd.DataFrame({
    "siret": df.get(COLS["siret"], ""),
    "nom": df.apply(coalesce_name, axis=1),
    "adresse": df.get(COLS["adresse"], ""),
    "cp": df.get(COLS["cp"], "").astype(str),
    "commune": df.get(COLS["commune"], ""),
    "naf": df.get(COLS["naf"], ""),
    "lat": df["lat"],
    "lon": df["lon"],
    "__dep__": df["__dep__"],
    "__source__": df["__source__"],
})

# Liens
ent["pj_url"] = ent.apply(lambda r: build_pj_link(r["nom"], r["adresse"], r["cp"], r["commune"]), axis=1)
ent["gmaps_url"] = ent.apply(lambda r: build_gmaps_link(r["lat"], r["lon"], r["nom"], r["adresse"]), axis=1)

st.success(f"✅ Entreprises à afficher : **{len(ent):,}**")

# ---------- Couche Méthaniseurs ----------
st.subheader("3) Couche optionnelle : Méthaniseurs")
show_meth = st.checkbox("Afficher la couche 'Méthaniseurs' si data/methaniseurs/methaniseurs.csv est présent", value=METH_FILE.exists())
meth = load_methaniseurs() if show_meth else None
if show_meth and meth is None:
    st.info("Aucun fichier valide trouvé pour les méthaniseurs (attendu: nom, adresse, latitude, longitude).")

# ---------- Carte ----------
st.subheader("4) Carte")
m = folium.Map(location=[46.6, 2.4], zoom_start=6, tiles="OpenStreetMap")
cluster_ent = MarkerCluster(name="Entreprises").add_to(m)

for _, r in ent.iterrows():
    popup = f"""<b>{_norm(r.get('nom',''))}</b><br>
    {r.get('adresse','') or ''}<br>
    {(r.get('cp','') or '')} {(r.get('commune','') or '')}<br>
    Dép: {r.get('__dep__','')} | SIRET: {r.get('siret','') or ''}<br>
    NAF: {r.get('naf','') or ''}<br>
    <a href="{r.get('gmaps_url','')}" target="_blank">Google Maps</a> |
    <a href="{r.get('pj_url','')}" target="_blank">PagesJaunes</a>"""
    try:
        folium.Marker([float(r["lat"]), float(r["lon"])],
                      popup=popup,
                      icon=folium.Icon(color="blue", icon="briefcase", prefix="fa")).add_to(cluster_ent)
    except Exception:
        continue

if meth is not None and len(meth):
    cluster_m = MarkerCluster(name="Méthaniseurs").add_to(m)
    for _, r in meth.iterrows():
        popup = f"""<b>{_norm(str(r.get('nom','Méthaniseur')))}</b><br>
        {r.get('adresse','') or ''}<br>
        <a href="{r.get('gmaps_url','')}" target="_blank">Google Maps</a>"""
        try:
            folium.Marker([float(r["lat"]), float(r["lon"])],
                          popup=popup,
                          icon=folium.Icon(color="green", icon="leaf", prefix="fa")).add_to(cluster_m)
        except Exception:
            continue

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width=1200, height=700)

# ---------- Export ----------
st.subheader("5) Export CSV des données affichées")
csv_bytes = ent.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Télécharger les entreprises (CSV)", data=csv_bytes,
                   file_name="entreprises_filtrees.csv", mime="text/csv")

st.caption("💡 Ajoute/retire des CSV dans data/entreprises/, puis redeploie pour mettre à jour.")

