import streamlit as st
import pandas as pd
from pathlib import Path
import re, unicodedata
from urllib.parse import quote_plus
from io import BytesIO
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster

# Parquet / Arrow
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

# ==================== CONFIG ====================
st.set_page_config(page_title="Carte entreprises par NAF (départements + méthaniseurs)", layout="wide")
st.title("🗺️ Carte entreprises par NAF — sélection par département + couche Méthaniseurs")

ROOT     = Path(__file__).parent
DIR_ENT  = ROOT / "data" / "entreprises"
DIR_METH = ROOT / "data" / "methaniseurs"
DIR_UL   = ROOT / "data" / "unite_legale" / "ul_parts"   # <-- UL partitions

# Colonnes SIRENE (selon tes fichiers entreprises)
COLS = {
    "siret":   "siret",
    "etat":    "etatAdministratifEtablissement",
    "naf":     "activitePrincipaleEtablissement",
    "enseigne1":"enseigne1Etablissement",
    "denom":   "denominationUsuelleEtablissement",
    "lon":     "longitude",
    "lat":     "latitude",
    "adresse": "geo_adresse",
    "cp":      "codePostalEtablissement",
    "commune": "libelleCommuneEtablissement",
    "siege":   "etablissementSiege",
}

NEEDED_COLS = [
    COLS["siret"], COLS["etat"], COLS["naf"], COLS["enseigne1"], COLS["denom"],
    COLS["lon"], COLS["lat"], COLS["adresse"], COLS["cp"], COLS["commune"], COLS["siege"]
]

DEPT_RE = re.compile(r"geo_siret_([0-9]{2}|[0-9]{3}|2A|2B)", re.IGNORECASE)

# ---------- NAF (Division -> sous-classes) ----------
NAF_DIVISIONS = {
    "01 — Culture et production animale, chasse et services annexes": [
        ("0111Z", "Culture de céréales (à l'exception du riz), de légumineuses et de graines oléagineuses"),
        ("0112Z", "Culture du riz"),
        ("0113Z", "Culture de légumes, de melons, de racines et de tubercules"),
        ("0114Z", "Culture de la canne à sucre"),
        ("0115Z", "Culture du tabac"),
        ("0116Z", "Culture de plantes à fibres"),
        ("0119Z", "Autres cultures non permanentes"),
        ("0121Z", "Culture de la vigne"),
        ("0122Z", "Culture de fruits tropicaux et subtropicaux"),
        ("0123Z", "Culture d'agrumes"),
        ("0124Z", "Culture de fruits à pépins et à noyau"),
        ("0125Z", "Culture d'autres fruits d'arbres ou d'arbustes et de fruits à coque"),
        ("0126Z", "Culture de fruits oléagineux"),
        ("0127Z", "Culture de plantes à boissons"),
        ("0128Z", "Culture de plantes à épices, aromatiques, médicinales et pharmaceutiques"),
        ("0129Z", "Autres cultures permanentes"),
        ("0130Z", "Reproduction de plantes"),
        ("0141Z", "Élevage de vaches laitières"),
        ("0142Z", "Élevage d'autres bovins et de buffles"),
        ("0143Z", "Élevage de chevaux et d'autres équidés"),
        ("0144Z", "Élevage de chameaux et d'autres camélidés"),
        ("0145Z", "Élevage d'ovins et de caprins"),
        ("0146Z", "Élevage de porcins"),
        ("0147Z", "Élevage de volailles"),
        ("0149Z", "Élevage d'autres animaux"),
        ("0150Z", "Culture et élevage associés"),
        ("0161Z", "Activités de soutien aux cultures"),
        ("0162Z", "Activités de soutien à la production animale"),
        ("0163Z", "Traitement primaire des récoltes"),
        ("0164Z", "Traitement des semences"),
        ("0170Z", "Chasse, piégeage et services annexes"),
    ],
    "02 — Sylviculture et exploitation forestière": [
        ("0210Z", "Sylviculture et autres activités forestières"),
        ("0220Z", "Exploitation forestière"),
        ("0230Z", "Récolte de produits forestiers non ligneux poussant à l'état sauvage"),
        ("0240Z", "Services de soutien à l'exploitation forestière"),
    ],
    "03 — Pêche et aquaculture": [
        ("0311Z", "Pêche en mer"),
        ("0312Z", "Pêche en eau douce"),
        ("0321Z", "Aquaculture en mer"),
        ("0322Z", "Aquaculture en eau douce"),
    ],
    "10 — Industries alimentaires": [
        ("1011Z", "Transformation et conservation de la viande de boucherie"),
        ("1012Z", "Transformation et conservation de la viande de volaille"),
        ("1013A", "Préparation industrielle de produits à base de viande"),
        ("1013B", "Charcuterie"),
        ("1020Z", "Transformation et conservation de poisson, de crustacés et de mollusques"),
        ("1031Z", "Transformation et conservation de pommes de terre"),
        ("1032Z", "Préparation de jus de fruits et légumes"),
        ("1039A", "Autre transformation et conservation de légumes"),
        ("1039B", "Transformation et conservation de fruits"),
        ("1041B", "Fabrication d'huiles et graisses raffinées"),
        ("1042Z", "Fabrication de margarine et graisses comestibles similaires"),
        ("1051A", "Fabrication de lait liquide et de produits frais"),
        ("1051B", "Fabrication de beurre"),
        ("1051C", "Fabrication de fromage"),
        ("1051D", "Fabrication d'autres produits laitiers"),
        ("1052Z", "Fabrication de glaces et sorbets"),
        ("1061A", "Meunerie"),
        ("1061B", "Autres activités du travail des grains"),
        ("1062Z", "Fabrication de produits amylacés"),
        ("1071A", "Fabrication industrielle de pain et de pâtisserie fraîche"),
        ("1071B", "Cuisson de produits de boulangerie"),
        ("1071C", "Boulangerie et boulangerie-pâtisserie"),
        ("1071D", "Pâtisserie"),
        ("1072Z", "Fabrication de biscuits, biscottes et pâtisseries de conservation"),
        ("1073Z", "Fabrication de pâtes alimentaires"),
        ("1081Z", "Fabrication de sucre"),
        ("1082Z", "Fabrication de cacao, chocolat et de produits de confiserie"),
        ("1083Z", "Transformation du thé et du café"),
        ("1084Z", "Fabrication de condiments et assaisonnements"),
        ("1085Z", "Fabrication de plats préparés"),
        ("1086Z", "Fabrication d'aliments homogénéisés et diététiques"),
        ("1089Z", "Fabrication d'autres produits alimentaires n.c.a."),
        ("1091Z", "Fabrication d'aliments pour animaux de ferme"),
        ("1092Z", "Fabrication d'aliments pour animaux de compagnie"),
    ],
    "11 — Fabrication de boissons": [
        ("1101Z", "Production de boissons alcooliques distillées"),
        ("1102A", "Fabrication de vins effervescents"),
        ("1102B", "Vinification"),
        ("1103Z", "Fabrication de cidre et de vins de fruits"),
        ("1104Z", "Production d'autres boissons fermentées non distillées"),
        ("1105Z", "Fabrication de bière"),
        ("1106Z", "Fabrication de malt"),
        ("1107A", "Industrie des eaux de table"),
        ("1107B", "Production de boissons rafraîchissantes"),
    ],
    "16 — Travail du bois, liège, vannerie, sparterie (hors meubles)": [
        ("1610A", "Sciage et rabotage du bois, hors imprégnation"),
        ("1610B", "Imprégnation du bois"),
        ("1621Z", "Fabrication de placage et de panneaux de bois"),
        ("1622Z", "Fabrication de parquets assemblés"),
        ("1623Z", "Fabrication de charpentes et d'autres menuiseries"),
        ("1624Z", "Fabrication d'emballages en bois"),
        ("1629Z", "Objets en bois, liège, vannerie et sparterie"),
    ],
    "17 — Industrie du papier et du carton": [
        ("1711Z", "Fabrication de pâte à papier"),
        ("1712Z", "Fabrication de papier et de carton"),
        ("1721A", "Fabrication de carton ondulé"),
        ("1721B", "Fabrication de cartonnages"),
        ("1721C", "Fabrication d'emballages en papier"),
        ("1722Z", "Articles en papier à usage sanitaire ou domestique"),
        ("1723Z", "Articles de papeterie"),
        ("1724Z", "Papiers peints"),
        ("1729Z", "Autres articles en papier ou en carton"),
    ],
    "31 — Fabrication de meubles": [
        ("3101Z", "Meubles de bureau et de magasin"),
        ("3102Z", "Meubles de cuisine"),
        ("3103Z", "Matelas"),
        ("3109A", "Sièges d'ameublement d'intérieur"),
        ("3109B", "Autres meubles et industries connexes"),
    ],
    "35 — Électricité, gaz, vapeur et air conditionné": [
        ("3521Z", "Production de combustibles gazeux"),
    ],
    "38 — Déchets : collecte, traitement, élimination, récupération": [
        ("3811Z", "Collecte des déchets non dangereux"),
        ("3812Z", "Collecte des déchets dangereux"),
        ("3821Z", "Traitement et élimination des déchets non dangereux"),
        ("3822Z", "Traitement et élimination des déchets dangereux"),
        ("3831Z", "Démantèlement d'épaves"),
        ("3832Z", "Récupération de déchets triés"),
    ],
    "46 — Commerce de gros (sauf auto/moto)": [
        ("4611Z", "Intermédiaires du commerce (MP agricoles, animaux, textiles, semi-finis)"),
        ("4617A", "Centrales d'achat alimentaires"),
        ("4613Z", "Intermédiaires commerce de gros en bois et matériaux de construction"),
        ("4631Z", "Commerce de gros de fruits et légumes"),
        ("4621Z", "Gros de céréales, tabac non manufacturé, semences, aliments pour bétail"),
        ("4633Z", "Gros de produits laitiers, œufs, huiles et matières grasses comestibles"),
        ("4639B", "Gros alimentaire non spécialisé"),
        ("4673A", "Gros de bois et matériaux de construction"),
        ("4677Z", "Gros de déchets et débris"),
    ],
    "47 — Commerce de détail (sauf auto/moto)": [
        ("4711B", "Commerce d'alimentation générale"),
        ("4711F", "Hypermarchés"),
        ("4721Z", "Détail de fruits et légumes en magasin spécialisé"),
        ("4781Z", "Détail alimentaire sur éventaires et marchés"),
    ],
    "56 — Restauration": [
        ("5610A", "Restauration traditionnelle"),
        ("5610B", "Cafétérias et autres libres-services"),
        ("5610C", "Restauration de type rapide"),
        ("5621Z", "Services des traiteurs"),
        ("5629A", "Restauration collective sous contrat"),
        ("5629B", "Autres services de restauration n.c.a."),
    ],
}


# ==================== UTILS ====================
def _norm(s: str):
    if not isinstance(s, str): return ""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return re.sub(r"[^0-9A-Za-z ]+", " ", s).strip()

def _slugify(s: str) -> str:
    """Slug pour PagesJaunes: sans accents, minuscule, tirets."""
    if not isinstance(s, str): return ""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^0-9A-Za-z]+", "-", s).strip("-").lower()
    s = re.sub(r"-{2,}", "-", s)
    return s

def build_gmaps_fiche(nom, adresse, cp, commune, siret=None):
    """
    Lien 'fiche' Google: on force une recherche précise avec "Nom exact" + adresse + CP + commune + SIRET.
    Ça maximise les chances d’ouvrir directement la fiche établissement.
    """
    parts = []
    if isinstance(nom, str) and nom.strip():
        parts.append(f'"{nom.strip()}"')  # guillemets = match exact du nom
    if isinstance(adresse, str) and adresse.strip():
        parts.append(adresse.strip())
    if isinstance(cp, str) and cp.strip():
        parts.append(cp.strip())
    if isinstance(commune, str) and commune.strip():
        parts.append(commune.strip())
    if isinstance(siret, str) and siret.strip():
        parts.append(f"SIRET {siret.strip()}")
    q = " ".join(parts) if parts else ""
    if not q:
        q = "France"
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(q)}"

def build_gmaps_point(lat, lon, nom=None):
    """Pin direct lat/lon sur Google Maps (secours si la fiche n’est pas reconnue)."""
    if pd.notna(lat) and pd.notna(lon):
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    # fallback recherche par nom (si pas de coords)
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(str(nom or '').strip())}"

def build_pj_links(nom, adresse, cp, commune):
    """
    Deux variantes pour PagesJaunes :
    - URL 'jolie': /recherche/<ville-dept>/<quoiqui>
    - URL querystring de secours: ?quoiqui=&ou=
    On renvoie un tuple (pj_pretty, pj_qs).
    """
    nom = (nom or "").strip()
    adresse = (adresse or "").strip()
    cp = (str(cp or "")).strip()
    commune = (commune or "").strip()

    # ville-dept pour slug 'joli' (ex: paris-75, lyon-69)
    dept = cp[:2] if cp else ""
    ville_slug = _slugify(commune) + (f"-{dept}" if dept else "")
    quoiqui_slug = _slugify(nom) if nom else _slugify(adresse) or "entreprise"

    pj_pretty = f"https://www.pagesjaunes.fr/recherche/{ville_slug}/{quoiqui_slug}".rstrip("/")

    # Fallback plus tolérant
    quoiqui = nom if nom else " ".join([adresse, commune]).strip()
    ou = " ".join([cp, commune]).strip() or commune or "France"
    pj_qs = f"https://www.pagesjaunes.fr/recherche?quoiqui={quote_plus(quoiqui)}&ou={quote_plus(ou)}"

    return pj_pretty, pj_qs

def coalesce_name_etab(row):
    for c in (COLS["denom"], COLS["enseigne1"]):
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

# ==================== FICHIERS DISPONIBLES ====================
@st.cache_data(show_spinner=False)
def files_by_dep():
    files = []
    for ext in (".parquet", ".csv.gz", ".zip", ".csv"):
        files.extend(DIR_ENT.glob(f"geo_siret_*{ext}"))
    out = {}
    for f in files:
        m = DEPT_RE.search(f.name)
        code = (m.group(1).upper() if m else f.stem)
        out.setdefault(code, []).append(f)
    out = {k: sorted(v, key=lambda p: p.name) for k, v in out.items()}
    return out

# ==================== DÉCOUVERTE NAF (OPTIONNEL) ====================
@st.cache_data(show_spinner=True)
def discover_naf_codes(selected_deps: tuple[str, ...]) -> list[str]:
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
                    naf.update(
                        s.astype("string").str.upper().str.replace(r"[^0-9A-Z.]", "", regex=True).dropna().unique()
                    )
                elif name.endswith(".csv.gz") or name.endswith(".gz") or name.endswith(".csv") or name.endswith(".zip"):
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
                                    if cnt >= 600_000:
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
    return codes[:2000]

# ==================== CHARGEMENT FILTRÉ (ENTREPRISES) ====================
def _filter_in_pandas(df: pd.DataFrame, naf_set: set[str], only_siege: bool) -> pd.DataFrame:
    df = df.copy()
    if COLS["etat"] in df.columns:
        df = df[df[COLS["etat"]].astype(str).str.upper().str.startswith("A")]
    if naf_set and COLS["naf"] in df.columns:
        naf_clean = df[COLS["naf"]].astype(str).str.upper().str.replace(r"[^0-9A-Z.]", "", regex=True)
        df = df[naf_clean.isin(list(naf_set))]
    if only_siege and COLS["siege"] in df.columns:
        df = df[df[COLS["siege"]].astype(str).isin(["1","True","true","O","Oui"])]
    if COLS["lat"] in df.columns and COLS["lon"] in df.columns:
        df.loc[:, COLS["lat"]] = pd.to_numeric(df[COLS["lat"]].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        df.loc[:, COLS["lon"]] = pd.to_numeric(df[COLS["lon"]].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        df = df[df[COLS["lat"]].notna() & df[COLS["lon"]].notna()]
    return df

def load_filtered(selected_deps: list[str], naf_selected: list[str], only_siege: bool) -> pd.DataFrame:
    fb = files_by_dep()
    naf_set = {re.sub(r"[^0-9A-Z.]", "", c.upper()) for c in naf_selected if c}
    frames = []

    needed = [c for c in NEEDED_COLS if c]
    for dep in selected_deps:
        files = fb.get(dep, [])
        if not files:
            continue

        # 1) Parquet
        pq_files = [str(p) for p in files if p.suffix.lower() == ".parquet"]
        if pq_files:
            filt = pc.field(COLS["naf"]).isin(list(naf_set)) if naf_set else None
            dset = ds.dataset(pq_files, format="parquet")
            cols = [c for c in needed if c in dset.schema.names]
            try:
                tbl = dset.to_table(columns=cols, filter=filt)
                df = tbl.to_pandas()
                df["__dep__"] = dep
                df["__source__"] = "parquet"
                df = _filter_in_pandas(df, naf_set=set(), only_siege=only_siege)
                if not df.empty:
                    frames.append(df)
            except Exception:
                pass

        # 2) CSV-like
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

# ==================== UNITE LEGALE : LECTURE & JOINTURE ====================
UL_NAME_COLS = [
    "denominationUniteLegale",
    "denominationUsuelle1UniteLegale",
    "denominationUsuelle2UniteLegale",
    "denominationUsuelle3UniteLegale",
    "sigleUniteLegale",
    "nomUsageUniteLegale",
    "nomUniteLegale",
    "prenom1UniteLegale",
    "prenomUsuelUniteLegale",
    "pseudonymeUniteLegale",
]

def _best_ul_name(row: pd.Series) -> str:
    for c in ["denominationUniteLegale","denominationUsuelle1UniteLegale","denominationUsuelle2UniteLegale","denominationUsuelle3UniteLegale","sigleUniteLegale"]:
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            return v.strip()
    prenom = (row.get("prenom1UniteLegale") or row.get("prenomUsuelUniteLegale") or "").strip()
    nom    = (row.get("nomUsageUniteLegale") or row.get("nomUniteLegale") or "").strip()
    if nom or prenom:
        return (prenom + " " + nom).strip()
    v = row.get("pseudonymeUniteLegale")
    return (v or "").strip()

@st.cache_data(show_spinner=False)
def load_ul_names_for(sirens: list[str]) -> pd.DataFrame:
    if not sirens:
        return pd.DataFrame(columns=["siren","nom_ul"])
    if not DIR_UL.exists():
        st.warning("Dossier data/unite_legale/ul_parts introuvable.")
        return pd.DataFrame(columns=["siren","nom_ul"])
    parts = sorted(DIR_UL.glob("*.parquet"))
    if not parts:
        st.warning("Aucun fichier parquet dans data/unite_legale/ul_parts.")
        return pd.DataFrame(columns=["siren","nom_ul"])

    dset = ds.dataset([str(p) for p in parts], format="parquet")
    cols = ["siren"] + [c for c in UL_NAME_COLS if c in dset.schema.names]

    CHUNK = 50_000
    frames = []
    for i in range(0, len(sirens), CHUNK):
        chunk = sirens[i:i+CHUNK]
        filt = pc.field("siren").isin(pa.array(chunk, type=pa.string()))
        tbl  = dset.to_table(columns=cols, filter=filt)
        df   = tbl.to_pandas()
        if not df.empty:
            df["nom_ul"] = df.apply(_best_ul_name, axis=1)
            frames.append(df[["siren","nom_ul"]])

    if not frames:
        return pd.DataFrame(columns=["siren","nom_ul"])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["siren"], keep="first")
    return out

# ==================== METHANISEURS ====================
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
    dfm["gmaps_url"] = dfm.apply(lambda r: build_gmaps_point(r.get("lat"), r.get("lon"), r.get("nom")), axis=1)
    keep = [c for c in ["nom","adresse","lat","lon","gmaps_url"] if c in dfm.columns]
    return dfm[keep].copy()

# ==================== UI ====================
# --- état UI par défaut (évite les KeyError au 1er run) ---
if "go" not in st.session_state:
    st.session_state["go"] = False
if "naf_options" not in st.session_state:
    st.session_state["naf_options"] = []

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

# --- A) Saisie libre + Scan (inchangé) ---
colA, colB = st.columns([2,1])
with colA:
    naf_input = st.text_input("Saisis des codes NAF (séparés par des virgules)", value="")
with colB:
    scan_click = st.button("Scanner les codes NAF (colonne NAF)")

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

# --- B) Sélection par Division → Secteurs ---
st.markdown("**Sélection guidée par Division / Secteurs (codes NAF avec libellés)**")
colD, colS = st.columns([1,2])
with colD:
    divisions_sel = st.multiselect(
        "Division(s) (tu peux en choisir plusieurs)",
        options=list(NAF_DIVISIONS.keys()),
        default=[]
    )

# liste des options (code, libellé) en fonction des divisions choisies
sect_options = []
for d in divisions_sel:
    sect_options.extend(NAF_DIVISIONS.get(d, []))

# dédoublonner par code en gardant le 1er libellé rencontré
code2label = {}
for code, label in sect_options:
    if code not in code2label:
        code2label[code] = label
sect_options_unique = [(c, l) for c, l in sorted(code2label.items())]

with colS:
    secteurs_sel = st.multiselect(
        "Sous-classes NAF (des divisions sélectionnées)",
        options=sect_options_unique,
        format_func=lambda x: f"{x[0]} — {x[1]}",
        default=[]
    )

# Option "tout prendre" pour les divisions cochées
take_all = False
if divisions_sel and sect_options_unique:
    take_all = st.checkbox("Sélectionner toutes les sous-classes des divisions choisies")

naf_from_div = [c for (c, _) in sect_options_unique] if take_all else [c for (c, _) in secteurs_sel]

# --- Fusion des 3 sources : saisie libre + scan + divisions/secteurs ---
naf_typed = [re.sub(r"[^0-9A-Z.]", "", c.upper()) for c in naf_input.split(",")]
naf_typed = [c for c in naf_typed if c]

naf_final = sorted(set(naf_typed) | set(naf_select_ms) | set(naf_from_div))
st.caption(f"🧩 Codes NAF retenus ({len(naf_final)}): {', '.join(naf_final) if naf_final else '—'}")

# 2bis) Options de filtrage
only_siege = st.checkbox("Ne garder que les sièges (etablissementSiege=1)", value=False)


st.subheader("3) Charger les données filtrées")
col_go, col_reset = st.columns([1,1])
with col_go:
    if st.button("Charger la carte"):
        st.session_state["go"] = True
with col_reset:
    if st.button("Réinitialiser"):
        st.session_state["go"] = False
        st.session_state["naf_options"] = []

# ==================== RUN ====================
if st.session_state.get("go", False):
    if not selected_deps:
        st.warning("Sélectionne au moins un département.")
        st.stop()

    with st.spinner("Chargement filtré (entreprises)…"):
        df = load_filtered(selected_deps, naf_final, only_siege)

    if df.empty:
        st.info("Aucune ligne avec ces filtres (NAF, siège, coordonnées) dans les départements sélectionnés.")
        st.stop()

    # === Base entreprises (avant jointure) ===
    df = df.copy()
    df.loc[:, "lat"] = pd.to_numeric(df[COLS["lat"]], errors="coerce")
    df.loc[:, "lon"] = pd.to_numeric(df[COLS["lon"]], errors="coerce")

    ent = pd.DataFrame({
        "siret":    df.get(COLS["siret"], "").astype(str),
        "nom_etab": df.apply(coalesce_name_etab, axis=1),
        "adresse":  df.get(COLS["adresse"], ""),
        "cp":       df.get(COLS["cp"], "").astype(str),
        "commune":  df.get(COLS["commune"], ""),
        "naf":      df.get(COLS["naf"], ""),
        "lat":      df["lat"],
        "lon":      df["lon"],
        "__dep__":  df["__dep__"],
        "__source__": df["__source__"],
    })

    # Contrôles avant jointure
    rows_before        = len(ent)
    siret_uniq_before  = ent["siret"].nunique()
    ent["siren"]       = ent["siret"].str.slice(0, 9)
    siren_uniq_before  = ent["siren"].nunique()

    # === Jointure UL ===
    with st.spinner("Jointure des noms d’Unité Légale…"):
        sirens_need = sorted(ent["siren"].dropna().unique().tolist())
        ul = load_ul_names_for(sirens_need)  # ['siren','nom_ul'] unique

    ent = ent.merge(ul, on="siren", how="left")

    # Nom final : priorité UL, repli sur établissement
    ent["nom_affiche"] = ent["nom_ul"]
    mask_vide = ent["nom_affiche"].isna() | (ent["nom_affiche"].astype(str).str.strip()=="")
    ent.loc[mask_vide, "nom_affiche"] = ent.loc[mask_vide, "nom_etab"]

    # Liens (Google Maps + PagesJaunes) — version fiche + secours
    ent["gmaps_fiche"] = ent.apply(lambda r: build_gmaps_fiche(r["nom_affiche"], r["adresse"], r["cp"], r["commune"], r["siret"]), axis=1)
    ent["gmaps_point"] = ent.apply(lambda r: build_gmaps_point(r["lat"], r["lon"], r["nom_affiche"]), axis=1)
    pj_links = ent.apply(lambda r: build_pj_links(r["nom_affiche"], r["adresse"], r["cp"], r["commune"]), axis=1, result_type="expand")
    ent["pj_url"]     = pj_links[0]  # jolie
    ent["pj_url_qs"]  = pj_links[1]  # fallback

    # Contrôles post-jointure
    rows_after        = len(ent)
    siret_uniq_after  = ent["siret"].nunique()
    siren_uniq_after  = ent["siren"].nunique()

    ok_rows  = (rows_before == rows_after)
    ok_siret = (siret_uniq_before == siret_uniq_after)
    ok_siren = (siren_uniq_before == siren_uniq_after)

    if ok_rows and ok_siret and ok_siren:
        st.success(f"✅ Chargé: {rows_after:,} lignes | SIRET uniques: {siret_uniq_after:,} | SIREN uniques: {siren_uniq_after:,} (aucune perte, aucun doublon).")
    else:
        st.error("❌ Incohérence détectée entre avant et après jointure (vérifie les sources UL).")
        st.write({
            "rows_before": rows_before, "rows_after": rows_after,
            "siret_uniq_before": siret_uniq_before, "siret_uniq_after": siret_uniq_after,
            "siren_uniq_before": siren_uniq_before, "siren_uniq_after": siren_uniq_after,
        })

    # ---------- Couche Méthaniseurs ----------
    st.subheader("4) Couche optionnelle : Méthaniseurs")
    meth_file = _find_meth_file()
    show_meth = st.checkbox(
        "Afficher la couche 'Méthaniseurs' (si un fichier est présent dans data/methaniseurs/)",
        value=bool(meth_file)
    )
    meth = load_methaniseurs() if show_meth else None
    if show_meth and meth is None:
        st.info("Aucun fichier valide trouvé pour les méthaniseurs (attendu: nom, adresse, lat, lon).")

    # ---------- Carte ----------
    st.subheader("5) Carte")
    m = folium.Map(location=[46.6, 2.4], zoom_start=6, tiles="OpenStreetMap")
    cluster_ent = MarkerCluster(name="Entreprises").add_to(m)

    for _, r in ent.iterrows():
        popup = f"""<b>{_norm(r.get('nom_affiche',''))}</b><br>
        {r.get('adresse','') or ''}<br>
        {(r.get('cp','') or '')} {(r.get('commune','') or '')}<br>
        Dép: {r.get('__dep__','')} | SIRET: {r.get('siret','') or ''}<br>
        NAF: {r.get('naf','') or ''}<br>
        <a href="{r.get('gmaps_fiche','')}" target="_blank">Google (fiche)</a> |
        <a href="{r.get('gmaps_point','')}" target="_blank">Google (point)</a> |
        <a href="{r.get('pj_url','')}" target="_blank">PagesJaunes</a> |
        <a href="{r.get('pj_url_qs','')}" target="_blank">PJ (recherche)</a>"""
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

    # ---------- Export CSV ----------
    st.subheader("6) Export CSV des données affichées")
    cols_export = [
        "siret","siren","nom_affiche","nom_ul","nom_etab",
        "adresse","cp","commune","naf","lat","lon",
        "__dep__","__source__",
        "gmaps_fiche","gmaps_point","pj_url","pj_url_qs"
    ]
    cols_export = [c for c in cols_export if c in ent.columns]
    csv_bytes = ent[cols_export].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger les entreprises (CSV)", data=csv_bytes,
                       file_name="entreprises_filtrees.csv", mime="text/csv")

    # ---------- Export CARTE HTML ----------
    st.subheader("7) Exporter la carte (HTML)")
    # Rend tout le HTML (incl. clusters et popups). Pas besoin d’écrire un fichier temporaire.
    html_str = m.get_root().render()
    st.download_button(
        "🗺️ Télécharger la carte (HTML)",
        data=html_str.encode("utf-8"),
        file_name="carte_entreprises.html",
        mime="text/html"
    )

else:
    st.info("💡 Sélectionne d’abord 1–n départements, saisis (ou scanne) des codes NAF, puis clique *Charger la carte*.")

