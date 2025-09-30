import streamlit as st
import pandas as pd
from pathlib import Path
import re, unicodedata
from urllib.parse import quote_plus
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster

# Pour les Parquet filtrés
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

# ---------- Config ----------
st.set_page_config(page_title="Carte entreprises par NAF (départements + méthaniseurs)", layout="wide")
st.title("🗺️ Carte entreprises par NAF — sélection par département + couche Méthaniseurs")

ROOT = Path(__file__).parent
DIR_ENT = ROOT / "data" / "entreprises"
DIR_METH = ROOT / "data" / "methaniseurs"

# Colonnes SIRENE (selon tes fichiers)
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

NEEDED_COLS = [
    COLS["siret"], COLS["etat"], COLS["naf"], COLS["enseigne1"], COLS["denom"],
    COLS["lon"], COLS["lat"], COLS["adresse"], COLS["cp"], COLS["commune"], COLS["siege"]
]

DEPT_RE = re.compile(r"geo_siret_([0-9]{2}|[0-9]{3}|2A|2B)", re.IGNORECASE)  # 2ch, DOM à 3ch, 2A/2B

# ---------- Utils ----------
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

def coalesce_name(row):
    for c in (COLS["denom"], COLS["enseigne1"]):
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            return v
    return ""

# ---------- Fichiers disponibles ----------
@st.cache_data(show_spinner=False)
def files_by_dep():
    """Retourne {dep: [Paths...]} en groupant toutes les parts/suffixes."""
    files = []
    for ext in (".parquet", ".csv.gz", ".zip", ".csv"):
        files.extend(DIR_ENT.glob(f"geo_siret_*{ext}"))
    out = {}
    for f in files:
        m = DEPT_RE.search(f.name)
        code = (m.group(1).upper() if m else f.stem)
        out.setdefault(code, []).append(f)
    # tri stable pour affichage
    out = {k: sorted(v, key=lambda p: p.name) for k, v in out.items()}
    return out

# ---------- Découverte NAF (optionnel) ----------
@st.cache_data(show_spinner=True)
def discover_naf_codes(selected_deps: tuple[str, ...]) -> list[str]:
    """Lit uniquement la colonne NAF des départements sélectionnés (Parquet en pushdown, CSV en échantillon/chunks)."""
    fb = files_by_dep()
    naf = set()
    for dep in selected_deps:
        for f in fb.get(dep, []):
            name = f.name.lower()
            try:
                if name.endswith(".parquet"):
                    dset = ds.dataset([str(f)], format="parquet")
                    t = dset.to_table(columns=[COLS["naf"]])
                    s = pd.Series(t[COLS["naf"]].to_pandas(dtype="string", types_mapper=pd.ArrowDtype))
                    naf.update(s.astype("string").str.upper().str.replace(r"[^0-9A-Z.]", "", regex=True).dropna().unique())
                elif name.endswith(".csv.gz") or name.endswith(".gz") or name.endswith(".csv") or name.endswith(".zip"):
                    # lecture partielle / chunks pour éviter la RAM
                    seps = [None, ";", ",", "\t"]; encs = ["utf-8", "latin1"]
                    read_ok = False
                    for sep in seps:
                        for enc in encs:
                            try:
                                kw = dict(usecols=[COLS["naf"]], encoding=enc, on_bad_lines="skip")
                                if name.endswith(".csv.gz") or name.endswith(".gz"):
                                    kw["compression"] = "gzip"
                                elif name.endswith(".zip"):
                                    kw["compression"] = "zip"
                                if sep is None:
                                    kw.update(sep=None, engine="python")
                                else:
                                    kw.update(sep=sep)
                                it = pd.read_csv(f, chunksize=150_000, **kw)
                                cnt = 0
                                for ch in it:
                                    s = ch[COLS["naf"]].astype("string").str.upper().str.replace(r"[^0-9A-Z.]", "", regex=True)
                                    naf.update(s.dropna().unique())
                                    cnt += len(ch)
                                    if cnt >= 600_000:  # plafonner le scan par fichier
                                        break
                                read_ok = True
                                break
                            except Exception:
                                continue
                        if read_ok: break
                else:
                    continue
            except Exception:
                continue
    codes = sorted([c for c in naf if c], key=lambda x: (len(x), x))
    return codes[:2000]  # limite raisonnable pour l’UI

# ---------- Chargement filtré ----------
def _filter_in_pandas(df: pd.DataFrame, naf_set: set[str], only_siege: bool) -> pd.DataFrame:
    # Evite le SettingWithCopyWarning + travaille sur une copie
    df = df.copy()

    if COLS["etat"] in df.columns:
        df = df[df[COLS["etat"]].astype(str).str.upper().str.startswith("A")]

    if naf_set and COLS["naf"] in df.columns:
        naf_clean = df[COLS["naf"]].astype(str).str.upper().str.replace(r"[^0-9A-Z.]", "", regex=True)
        df = df[naf_clean.isin(list(naf_set))]

    if only_siege and COLS["siege"] in df.columns:
        df = df[df[COLS["siege"]].astype(str).isin(["1","True","true","O","Oui"])]

    # Coords en numérique
    if COLS["lat"] in df.columns and COLS["lon"] in df.columns:
        df.loc[:, COLS["lat"]] = pd.to_numeric(df[COLS["lat"]].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        df.loc[:, COLS["lon"]] = pd.to_numeric(df[COLS["lon"]].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        df = df[df[COLS["lat"]].notna() & df[COLS["lon"]].notna()]

    return df


def load_filtered(selected_deps: list[str], naf_selected: list[str], only_siege: bool) -> pd.DataFrame:
    fb = files_by_dep()
    naf_set = {re.sub(r"[^0-9A-Z.]", "", c.upper()) for c in naf_selected if c}
    frames = []

    needed = [c for c in NEEDED_COLS if c]  # toutes les colonnes utiles
    for dep in selected_deps:
        files = fb.get(dep, [])
        if not files:
            continue

        # 1) Parquet (filtre pushdown sur NAF si possible)
        pq_files = [str(p) for p in files if p.suffix.lower() == ".parquet"]
        if pq_files:
            # Filtre pushdown NAF + colonnes minimales
            filt = None
            if naf_set:
                filt = pc.field(COLS["naf"]).isin(list(naf_set))
            dset = ds.dataset(pq_files, format="parquet")
            cols = [c for c in needed if c in dset.schema.names]
            try:
                tbl = dset.to_table(columns=cols, filter=filt)
                df = tbl.to_pandas()
                df["__dep__"] = dep
                df["__source__"] = "parquet"
                df = _filter_in_pandas(df, naf_set=set(), only_siege=only_siege)  # état/siege/coords côté pandas
                if not df.empty:
                    frames.append(df)
            except Exception:
                pass

        # 2) CSV-like (chunks + filtre en flux)
        csv_files = [p for p in files if p.suffix.lower() in (".csv", ".gz", ".zip") or p.name.lower().endswith(".csv.gz")]
        for f in csv_files:
            name = f.name.lower()
            seps = [None, ";", ",", "\t"]; encs = ["utf-8","latin1"]
            ok = False
            for sep in seps:
                for enc in encs:
                    try:
                        kw = dict(usecols=[c for c in needed if c], encoding=enc, on_bad_lines="skip", chunksize=200_000)
                        if name.endswith(".csv.gz") or name.endswith(".gz"):
                            kw["compression"] = "gzip"
                        elif name.endswith(".zip"):
                            kw["compression"] = "zip"
                        if sep is None:
                            kw.update(sep=None, engine="python")
                        else:
                            kw.update(sep=sep)
                        for ch in pd.read_csv(f, **kw):
                            ch["__dep__"] = dep
                            ch["__source__"] = f.name
                            ch = _filter_in_pandas(ch, naf_set=naf_set, only_siege=only_siege)
                            if not ch.empty:
                                frames.append(ch)
                        ok = True
                        break
                    except Exception:
                        continue
                if ok: break

    if not frames:
        return pd.DataFrame(columns=[c for c in needed] + ["__dep__","__source__"])
    return pd.concat(frames, ignore_index=True)

# ---------- Méthaniseurs ----------
def _find_meth_file() -> Path | None:
    for p in [DIR_METH / "methaniseurs.parquet", DIR_METH / "methaniseurs.csv.gz", DIR_METH / "methaniseurs.csv"]:
        if p.exists(): return p
    for ext in (".parquet", ".csv.gz", ".csv"):
        found = list(DIR_METH.glob(f"*{ext}"))
        if found: return found[0]
    return None

@st.cache_data(show_spinner=False)
def load_methaniseurs():
    p = _find_meth_file()
    if not p: return None
    n = p.name.lower()
    if n.endswith(".parquet"):
        dfm = pd.read_parquet(p)
    elif n.endswith(".csv.gz") or n.endswith(".gz"):
        dfm = pd.read_csv(p, compression="gzip")
    else:
        dfm = pd.read_csv(p)
    def pick(cols, cands):
        for c in cands:
            if c in cols: return c
        return None
    c_lat = pick(dfm.columns, ["latitude","lat","y"])
    c_lon = pick(dfm.columns, ["longitude","lon","x"])
    c_nom = pick(dfm.columns, ["nom","name","denomination","enseigne"])
    c_addr= pick(dfm.columns, ["adresse","address","geo_adresse"])
    if not c_lat or not c_lon: return None
    dfm["lat"] = pd.to_numeric(dfm[c_lat].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    dfm["lon"] = pd.to_numeric(dfm[c_lon].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    dfm = dfm[dfm["lat"].notna() & dfm["lon"].notna()].copy()
    if c_nom and "nom" not in dfm.columns: dfm.rename(columns={c_nom:"nom"}, inplace=True)
    if c_addr and "adresse" not in dfm.columns: dfm.rename(columns={c_addr:"adresse"}, inplace=True)
    dfm["gmaps_url"] = dfm.apply(lambda r: build_gmaps_link(r.get("lat"), r.get("lon"), r.get("nom"), r.get("adresse")), axis=1)
    keep = [c for c in ["nom","adresse","lat","lon","gmaps_url"] if c in dfm.columns]
    return dfm[keep].copy()

# ===================== UI =====================

fb = files_by_dep()
if not fb:
    st.error("Aucun fichier trouvé dans data/entreprises/ (ex: geo_siret_01.parquet).")
    st.stop()

all_deps = sorted(fb.keys(), key=lambda x: (len(x), x))

st.subheader("1) Sélection des départements")
selected_deps = st.multiselect(
    "Choisis 1 à N départements (évite 'Tous' pour la mémoire)",
    options=all_deps,
    default=[],
)

st.subheader("2) Codes NAF")

# Multiselect alimenté par un scan (on ne scanne que si l'utilisateur le demande)
colA, colB = st.columns([2,1])
with colA:
    naf_input = st.text_input("Saisis des codes NAF (séparés par des virgules)", value="")
with colB:
    scan_click = st.button("Scanner les codes NAF (colonne NAF)")

# On mémorise la liste scannée dans la session pour éviter de rescanner à chaque rerun
if "naf_options" not in st.session_state:
    st.session_state["naf_options"] = []

if scan_click:
    if not selected_deps:
        st.warning("Sélectionne d'abord au moins un département pour scanner les codes NAF.")
    else:
        with st.spinner("Scan des codes NAF sur les départements sélectionnés…"):
            st.session_state["naf_options"] = discover_naf_codes(tuple(selected_deps))

naf_select_ms = st.multiselect(
    "…ou choisis dans la liste déroulante (issue du scan)",
    options=st.session_state["naf_options"],
    default=[]
)

# Fusion des codes saisis + sélectionnés
naf_typed = [re.sub(r"[^0-9A-Z.]", "", c.upper()) for c in naf_input.split(",")]
naf_typed = [c for c in naf_typed if c]
naf_final = sorted(set(naf_typed) | set(naf_select_ms))

# Checkbox siège
only_siege = st.checkbox("Ne garder que les sièges (etablissementSiege=1)", value=False)

# --- Boutons persistants ---
if "go" not in st.session_state:
    st.session_state["go"] = False

st.subheader("3) Charger les données filtrées")
col_go, col_reset = st.columns([1,1])
with col_go:
    if st.button("Charger la carte"):
        st.session_state["go"] = True
with col_reset:
    if st.button("Réinitialiser"):
        st.session_state["go"] = False
        st.session_state["naf_options"] = []

# --- Affichage conditionnel persistant ---
if st.session_state["go"]:
    if not selected_deps:
        st.warning("Sélectionne au moins un département.")
        st.stop()

    with st.spinner("Chargement filtré…"):
        df = load_filtered(selected_deps, naf_final, only_siege)

    if df.empty:
        st.info("Aucune ligne avec ces filtres (NAF, siège, coordonnées) dans les départements sélectionnés.")
        st.stop()

    # Construction tableau final (avec loc pour éviter SettingWithCopy)
    df = df.copy()
    df.loc[:, "lat"] = pd.to_numeric(df[COLS["lat"]], errors="coerce")
    df.loc[:, "lon"] = pd.to_numeric(df[COLS["lon"]], errors="coerce")

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

    st.success(f"✅ Entreprises à afficher : **{len(ent):,}**")
    if len(ent) > 50_000:
        st.warning("Beaucoup de points à afficher. J’affiche un échantillon de 50 000 pour garder la carte fluide.")
        ent = ent.sample(50_000, random_state=1)

    # …puis TON bloc carte/folium + export, inchangé…
else:
    st.info("💡 Sélectionne 1–n départements, choisis des codes NAF (scan ou saisie), puis clique *Charger la carte*.")

    # Couche Méthaniseurs
    st.subheader("4) Couche optionnelle : Méthaniseurs")
    meth_file = _find_meth_file()
    show_meth = st.checkbox(
        "Afficher la couche 'Méthaniseurs' (si un fichier est présent dans data/methaniseurs/)",
        value=bool(meth_file)
    )
    meth = load_methaniseurs() if show_meth else None
    if show_meth and meth is None:
        st.info("Aucun fichier valide trouvé pour les méthaniseurs (attendu: nom, adresse, lat, lon).")

    # Carte
    st.subheader("5) Carte")
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

    # Export
    st.subheader("6) Export CSV des données affichées")
    csv_bytes = ent.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger les entreprises (CSV)", data=csv_bytes,
                       file_name="entreprises_filtrees.csv", mime="text/csv")
else:
    st.info("💡 Sélectionne d’abord 1–n départements, saisis (ou scanne) des codes NAF, puis clique *Charger la carte*.")

