import streamlit as st
import pandas as pd
import unicodedata, re, zipfile
from io import BytesIO
from urllib.parse import quote_plus
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster

st.set_page_config(page_title="Carte entreprises (NAF) + couches", layout="wide")
st.title("🗺️ Carte entreprises par NAF + couches (ex. fromageries, méthaniseurs)")

# ---------- Helpers ----------
def _norm(s:str):
    if not isinstance(s, str): return ""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.lower()
    return re.sub(r"[^a-z0-9_]", "", s)

def _find_col(df, candidates):
    normmap = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if cand in normmap:
            return normmap[cand]
    return None

def _read_csv_auto(filelike):
    # détecte ; ou , automatiquement
    try:
        return pd.read_csv(filelike, sep=None, engine="python", low_memory=False)
    except Exception:
        try:
            filelike.seek(0)
        except Exception:
            pass
        return pd.read_csv(filelike, sep=";", low_memory=False)

@st.cache_data(show_spinner=False)
def load_many(files):
    frames=[]
    for f in files:
        name = getattr(f, "name", "upload")
        if name.lower().endswith(".zip"):
            z = zipfile.ZipFile(f)
            for n in z.namelist():
                if n.lower().endswith(".csv"):
                    with z.open(n) as zf:
                        df = _read_csv_auto(zf)
                        df["__source__"] = n
                        frames.append(df)
        else:
            df = _read_csv_auto(f)
            df["__source__"] = name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def build_pj_link(nom, adresse, cp, commune):
    # Lien recherche PagesJaunes (approx, mais efficace)
    terms = " ".join([str(x) for x in [nom, adresse, cp, commune] if x])
    return f"https://www.pagesjaunes.fr/recherche/{quote_plus(terms)}"

def build_gmaps_link(lat, lon, nom=None, adresse=None):
    if pd.notna(lat) and pd.notna(lon):
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    # fallback sur recherche texte
    q = " ".join([str(x) for x in [nom, adresse] if x])
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(q)}"

# ---------- UI: chargement des données entreprises ----------
st.subheader("1) Charge tes CSV d’entreprises (par département ou ZIP)")
st.caption("Astuce : si possible, utilise des CSV **déjà géolocalisés** (colonnes latitude/longitude).")
uploads = st.file_uploader("Dépose plusieurs CSV ou un ZIP", type=["csv","zip"], accept_multiple_files=True)

# ---------- UI: couche optionnelle méthaniseurs ----------
st.subheader("2) (Optionnel) Ajoute un CSV de méthaniseurs")
meth_file = st.file_uploader("CSV unique ou ZIP pour 'Méthaniseurs'", type=["csv","zip"], accept_multiple_files=False)

if not uploads:
    st.info("➡️ Commence par déposer tes CSV d’entreprises.")
    st.stop()

df_raw = load_many(uploads)
st.write(f"📥 Données entreprises chargées : **{len(df_raw):,} lignes**")

# ---------- Mapping de colonnes ----------
col_naf  = _find_col(df_raw, ["activiteprincipaleetablissement","naf","ape"])
col_lat  = _find_col(df_raw, ["latitude","lat","geo_lat","y"])
col_lon  = _find_col(df_raw, ["longitude","lon","geo_lon","x"])
col_name = _find_col(df_raw, ["denominationunitelegale","denomination","enseigne1etablissement","enseigne","nom"])
col_addr = _find_col(df_raw, ["geo_adresse","adresseetablissement","adresse"])
col_cp   = _find_col(df_raw, ["codepostaletablissement","code_postal","cp"])
col_com  = _find_col(df_raw, ["libellecommuneetablissement","commune","ville"])
col_siret= _find_col(df_raw, ["siret"])
col_etat = _find_col(df_raw, ["etatadministratifetablissement","etat"])

if not col_lat or not col_lon:
    st.error("❌ Colonnes latitude/longitude introuvables. Fourni des CSV géolocalisés ou ajoute un géocodage en amont.")
    st.stop()

# ---------- Filtre 'actifs' ----------
df = df_raw.copy()
if col_etat:
    df = df[df[col_etat].astype(str).str.upper().str.startswith("A")]

# ---------- UI NAF : menu déroulant multi-sélection ----------
if col_naf:
    naf_series = df[col_naf].astype(str).str.upper().str.replace(r"[^0-9A-Z.]", "", regex=True)
    codes_naf_disponibles = sorted(naf_series.unique())
    # pré-selectionne 10.51C et 47.29Z si présents
    preselect = [c for c in ["10.51C","47.29Z","1051C","4729Z"] if c in codes_naf_disponibles]
    sel = st.multiselect("3) Choisis un ou plusieurs codes NAF à afficher",
                         options=codes_naf_disponibles,
                         default=preselect or codes_naf_disponibles[:10])
    if sel:
        df = df[naf_series.isin(sel)]
else:
    st.warning("⚠️ Colonne NAF non trouvée. Tous les enregistrements seront affichés.")

# ---------- Normalisation coords ----------
lat_raw = df[col_lat].astype(str).str.replace(",", ".", regex=False)
lon_raw = df[col_lon].astype(str).str.replace(",", ".", regex=False)
df["_lat"] = pd.to_numeric(lat_raw, errors="coerce")
df["_lon"] = pd.to_numeric(lon_raw, errors="coerce")
df = df.dropna(subset=["_lat","_lon"])

# ---------- Dataframe final entreprises ----------
keep_cols = [col_siret,col_name,col_addr,col_cp,col_com,col_naf,col_lat,col_lon,"__source__"]
keep = [c for c in keep_cols if c]
ent = df[keep].rename(columns={
    col_siret:"siret", col_name:"nom", col_addr:"adresse", col_cp:"cp",
    col_com:"commune", col_naf:"naf", col_lat:"lat", col_lon:"lon"
}).copy()
ent["pj_url"] = ent.apply(lambda r: build_pj_link(r.get("nom"), r.get("adresse"), r.get("cp"), r.get("commune")), axis=1)
ent["gmaps_url"] = ent.apply(lambda r: build_gmaps_link(r.get("lat"), r.get("lon"), r.get("nom"), r.get("adresse")), axis=1)

st.success(f"✅ Entreprises à afficher : **{len(ent):,}**")

# ---------- (Optionnel) Méthaniseurs ----------
meth = None
if meth_file:
    dfm = load_many([meth_file])
    col_lat_m = _find_col(dfm, ["latitude","lat","y"])
    col_lon_m = _find_col(dfm, ["longitude","lon","x"])
    name_m    = _find_col(dfm, ["nom","name","enseigne","denomination"])
    addr_m    = _find_col(dfm, ["adresse","address","geo_adresse"])
    if col_lat_m and col_lon_m:
        latm = pd.to_numeric(dfm[col_lat_m].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        lonm = pd.to_numeric(dfm[col_lon_m].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        dfm = dfm[latm.notna() & lonm.notna()].copy()
        dfm["lat"] = latm
        dfm["lon"] = lonm
        dfm["nom"] = dfm.get(name_m, "")
        dfm["adresse"] = dfm.get(addr_m, "")
        dfm["gmaps_url"] = dfm.apply(lambda r: build_gmaps_link(r.get("lat"), r.get("lon"), r.get("nom"), r.get("adresse")), axis=1)
        meth = dfm[["nom","adresse","lat","lon","gmaps_url","__source__"]].copy()
        st.info(f"➕ Couche 'Méthaniseurs' chargée : **{len(meth):,}** points")
    else:
        st.warning("⚠️ Méthaniseurs : colonnes lat/lon introuvables → couche ignorée.")

# ---------- Carte ----------
st.subheader("4) Carte interactive")
m = folium.Map(location=[46.6, 2.4], zoom_start=6, tiles="OpenStreetMap")

# Cluster entreprises
cluster_ent = MarkerCluster(name="Entreprises").add_to(m)
for _, r in ent.iterrows():
    popup = f"""<b>{r.get('nom','')}</b><br>
    {r.get('adresse','') or ''}<br>{(str(r.get('cp',''))+' '+str(r.get('commune',''))).strip()}<br>
    SIRET: {r.get('siret','') or ''}<br>NAF: {r.get('naf','') or ''}<br>
    <a href="{r.get('gmaps_url','')}" target="_blank">Google Maps</a> | 
    <a href="{r.get('pj_url','')}" target="_blank">PagesJaunes</a>
    """
    try:
        folium.Marker([float(r["lat"]), float(r["lon"])],
                      popup=popup,
                      icon=folium.Icon(color="blue", icon="briefcase", prefix="fa")).add_to(cluster_ent)
    except Exception:
        continue

# Cluster méthaniseurs
if meth is not None and len(meth):
    cluster_m = MarkerCluster(name="Méthaniseurs").add_to(m)
    for _, r in meth.iterrows():
        popup = f"""<b>{r.get('nom','(Méthaniseur)')}</b><br>
        {r.get('adresse','') or ''}<br>
        <a href="{r.get('gmaps_url','')}" target="_blank">Google Maps</a>
        """
        try:
            folium.Marker([float(r["lat"]), float(r["lon"])],
                          popup=popup,
                          icon=folium.Icon(color="green", icon="leaf", prefix="fa")).add_to(cluster_m)
        except Exception:
            continue

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width=1200, height=700)

# ---------- Exports ----------
st.subheader("5) Exports (données actuellement affichées)")
csv_ent = ent.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Télécharger entreprises (CSV)", data=csv_ent, file_name="entreprises_filtrees.csv", mime="text/csv")

if meth is not None and len(meth):
    csv_m = meth.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger méthaniseurs (CSV)", data=csv_m, file_name="methaniseurs.csv", mime="text/csv")

st.caption("💡 Si le volume > ~100k points, pense à filtrer par NAF ou charger par zone pour garder de bonnes perfs.")
