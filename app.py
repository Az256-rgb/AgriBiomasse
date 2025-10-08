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
        ("01.11Z", "Culture de céréales (à l'exception du riz), de légumineuses et de graines oléagineuses"),
        ("01.12Z", "Culture du riz"),
        ("01.13Z", "Culture de légumes, de melons, de racines et de tubercules"),
        ("01.14Z", "Culture de la canne à sucre"),
        ("01.15Z", "Culture du tabac"),
        ("01.16Z", "Culture de plantes à fibres"),
        ("01.19Z", "Autres cultures non permanentes"),
        ("01.21Z", "Culture de la vigne"),
        ("01.22Z", "Culture de fruits tropicaux et subtropicaux"),
        ("01.23Z", "Culture d'agrumes"),
        ("01.24Z", "Culture de fruits à pépins et à noyau"),
        ("01.25Z", "Culture d'autres fruits d'arbres ou d'arbustes et de fruits à coque"),
        ("01.26Z", "Culture de fruits oléagineux"),
        ("01.27Z", "Culture de plantes à boissons"),
        ("01.28Z", "Culture de plantes à épices, aromatiques, médicinales et pharmaceutiques"),
        ("01.29Z", "Autres cultures permanentes"),
        ("01.30Z", "Reproduction de plantes"),
        ("01.41Z", "Élevage de vaches laitières"),
        ("01.42Z", "Élevage d'autres bovins et de buffles"),
        ("01.43Z", "Élevage de chevaux et d'autres équidés"),
        ("01.44Z", "Élevage de chameaux et d'autres camélidés"),
        ("01.45Z", "Élevage d'ovins et de caprins"),
        ("01.46Z", "Élevage de porcins"),
        ("01.47Z", "Élevage de volailles"),
        ("01.49Z", "Élevage d'autres animaux"),
        ("01.50Z", "Culture et élevage associés"),
        ("01.61Z", "Activités de soutien aux cultures"),
        ("01.62Z", "Activités de soutien à la production animale"),
        ("01.63Z", "Traitement primaire des récoltes"),
        ("01.64Z", "Traitement des semences"),
        ("01.70Z", "Chasse, piégeage et services annexes"),
    ],
    "02 — Sylviculture et exploitation forestière": [
        ("02.10Z", "Sylviculture et autres activités forestières"),
        ("02.20Z", "Exploitation forestière"),
        ("02.30Z", "Récolte de produits forestiers non ligneux poussant à l'état sauvage"),
        ("02.40Z", "Services de soutien à l'exploitation forestière"),
    ],
    "03 — Pêche et aquaculture": [
        ("03.11Z", "Pêche en mer"),
        ("03.12Z", "Pêche en eau douce"),
        ("03.21Z", "Aquaculture en mer"),
        ("03.22Z", "Aquaculture en eau douce"),
    ],
    "10 — Industries alimentaires": [
        ("10.11Z", "Transformation et conservation de la viande de boucherie"),
        ("10.12Z", "Transformation et conservation de la viande de volaille"),
        ("10.13A", "Préparation industrielle de produits à base de viande"),
        ("10.13B", "Charcuterie"),
        ("10.20Z", "Transformation et conservation de poisson, de crustacés et de mollusques"),
        ("10.31Z", "Transformation et conservation de pommes de terre"),
        ("10.32Z", "Préparation de jus de fruits et légumes"),
        ("10.39A", "Autre transformation et conservation de légumes"),
        ("10.39B", "Transformation et conservation de fruits"),
        ("10.41B", "Fabrication d'huiles et graisses raffinées"),
        ("10.42Z", "Fabrication de margarine et graisses comestibles similaires"),
        ("10.51A", "Fabrication de lait liquide et de produits frais"),
        ("10.51B", "Fabrication de beurre"),
        ("10.51C", "Fabrication de fromage"),
        ("10.51D", "Fabrication d'autres produits laitiers"),
        ("10.52Z", "Fabrication de glaces et sorbets"),
        ("10.61A", "Meunerie"),
        ("10.61B", "Autres activités du travail des grains"),
        ("10.62Z", "Fabrication de produits amylacés"),
        ("10.71A", "Fabrication industrielle de pain et de pâtisserie fraîche"),
        ("10.71B", "Cuisson de produits de boulangerie"),
        ("10.71C", "Boulangerie et boulangerie-pâtisserie"),
        ("10.71D", "Pâtisserie"),
        ("10.72Z", "Fabrication de biscuits, biscottes et pâtisseries de conservation"),
        ("10.73Z", "Fabrication de pâtes alimentaires"),
        ("10.81Z", "Fabrication de sucre"),
        ("10.82Z", "Fabrication de cacao, chocolat et de produits de confiserie"),
        ("10.83Z", "Transformation du thé et du café"),
        ("10.84Z", "Fabrication de condiments et assaisonnements"),
        ("10.85Z", "Fabrication de plats préparés"),
        ("10.86Z", "Fabrication d'aliments homogénéisés et diététiques"),
        ("10.89Z", "Fabrication d'autres produits alimentaires n.c.a."),
        ("10.91Z", "Fabrication d'aliments pour animaux de ferme"),
        ("10.92Z", "Fabrication d'aliments pour animaux de compagnie"),
    ],
    "11 — Fabrication de boissons": [
        ("11.01Z", "Production de boissons alcooliques distillées"),
        ("11.02A", "Fabrication de vins effervescents"),
        ("11.02B", "Vinification"),
        ("11.03Z", "Fabrication de cidre et de vins de fruits"),
        ("11.04Z", "Production d'autres boissons fermentées non distillées"),
        ("11.05Z", "Fabrication de bière"),
        ("11.06Z", "Fabrication de malt"),
        ("11.07A", "Industrie des eaux de table"),
        ("11.07B", "Production de boissons rafraîchissantes"),
    ],
    "16 — Travail du bois, liège, vannerie, sparterie (hors meubles)": [
        ("16.10A", "Sciage et rabotage du bois, hors imprégnation"),
        ("16.10B", "Imprégnation du bois"),
        ("16.21Z", "Fabrication de placage et de panneaux de bois"),
        ("16.22Z", "Fabrication de parquets assemblés"),
        ("16.23Z", "Fabrication de charpentes et d'autres menuiseries"),
        ("16.24Z", "Fabrication d'emballages en bois"),
        ("16.29Z", "Objets en bois, liège, vannerie et sparterie"),
    ],
    "17 — Industrie du papier et du carton": [
        ("17.11Z", "Fabrication de pâte à papier"),
        ("17.12Z", "Fabrication de papier et de carton"),
        ("17.21A", "Fabrication de carton ondulé"),
        ("17.21B", "Fabrication de cartonnages"),
        ("17.21C", "Fabrication d'emballages en papier"),
        ("17.22Z", "Articles en papier à usage sanitaire ou domestique"),
        ("17.23Z", "Articles de papeterie"),
        ("17.24Z", "Papiers peints"),
        ("17.29Z", "Autres articles en papier ou en carton"),
    ],
    "31 — Fabrication de meubles": [
        ("31.01Z", "Meubles de bureau et de magasin"),
        ("31.02Z", "Meubles de cuisine"),
        ("31.03Z", "Matelas"),
        ("31.09A", "Sièges d'ameublement d'intérieur"),
        ("31.09B", "Autres meubles et industries connexes"),
    ],
    "35 — Électricité, gaz, vapeur et air conditionné": [
        ("35.21Z", "Production de combustibles gazeux"),
    ],
    "38 — Déchets : collecte, traitement, élimination, récupération": [
        ("38.11Z", "Collecte des déchets non dangereux"),
        ("38.12Z", "Collecte des déchets dangereux"),
        ("38.21Z", "Traitement et élimination des déchets non dangereux"),
        ("38.22Z", "Traitement et élimination des déchets dangereux"),
        ("38.31Z", "Démantèlement d'épaves"),
        ("38.32Z", "Récupération de déchets triés"),
    ],
    "46 — Commerce de gros (sauf auto/moto)": [
        ("46.11Z", "Intermédiaires du commerce (MP agricoles, animaux, textiles, semi-finis)"),
        ("46.17A", "Centrales d'achat alimentaires"),
        ("46.13Z", "Intermédiaires commerce de gros en bois et matériaux de construction"),
        ("46.31Z", "Commerce de gros de fruits et légumes"),
        ("46.21Z", "Gros de céréales, tabac non manufacturé, semences, aliments pour bétail"),
        ("46.33Z", "Gros de produits laitiers, œufs, huiles et matières grasses comestibles"),
        ("46.39B", "Gros alimentaire non spécialisé"),
        ("46.73A", "Gros de bois et matériaux de construction"),
        ("46.77Z", "Gros de déchets et débris"),
    ],
    "47 — Commerce de détail (sauf auto/moto)": [
        ("47.11B", "Commerce d'alimentation générale"),
        ("47.11F", "Hypermarchés"),
        ("47.21Z", "Détail de fruits et légumes en magasin spécialisé"),
        ("47.81Z", "Détail alimentaire sur éventaires et marchés"),
    ],
    "56 — Restauration": [
        ("56.10A", "Restauration traditionnelle"),
        ("56.10B", "Cafétérias et autres libres-services"),
        ("56.10C", "Restauration de type rapide"),
        ("56.21Z", "Services des traiteurs"),
        ("56.29A", "Restauration collective sous contrat"),
        ("56.29B", "Autres services de restauration n.c.a."),
    ],
}


# ==================== UTILS ====================
def canon_naf(x) -> str:
    if not isinstance(x, str):
        x = "" if x is None else str(x)
    return re.sub(r"[^0-9A-Z]", "", x.upper())  # enlève les points, espaces, etc.

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
                    s = pd.Series(t[COLS["naf"]].to_pandas())
                    naf.update(s.astype("string").map(canon_naf).dropna().unique())
                
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
                                    s = ch[COLS["naf"]].astype("string").map(canon_naf)
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
        naf_clean = df[COLS["naf"]].astype(str).map(canon_naf)
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

    # --- IMPORTANT : naf_set est bien défini ici, en canonique ---
    naf_set = {canon_naf(c) for c in naf_selected if c}
    frames = []

    needed = [c for c in NEEDED_COLS if c]
    for dep in selected_deps:
        files = fb.get(dep, [])
        if not files:
            continue

        # 1) Parquet
        pq_files = [str(p) for p in files if p.suffix.lower() == ".parquet"]
        if pq_files:
            dset = ds.dataset(pq_files, format="parquet")
            cols = [c for c in needed if c in dset.schema.names]

            # --- Filtre NAF côté Arrow : on normalise la colonne NAF (upper + enlève tout sauf [0-9A-Z]) ---
            filt = None
            if naf_set:
                try:
                    naf_field = pc.field(COLS["naf"]).cast(pa.string())
                    naf_upper = pc.utf8_upper(naf_field)
                    naf_norm  = pc.replace_substring_regex(naf_upper, pattern=r"[^0-9A-Z]", replacement="")
                    filt = pc.is_in(naf_norm, value_set=pa.array(sorted(naf_set), type=pa.string()))
                except Exception:
                    # Si la version de PyArrow ne supporte pas replace_substring_regex → on lira sans filtre
                    # et on filtrera ensuite en Pandas (plus lent mais robuste).
                    filt = None

            try:
                tbl = dset.to_table(columns=cols, filter=filt)
                df = tbl.to_pandas()
                df["__dep__"] = dep
                df["__source__"] = "parquet"

                # Si on a filtré NAF côté Arrow, on remet naf_set vide ici.
                # Sinon (filt None), on filtre en Pandas avec naf_set.
                df = _filter_in_pandas(df, naf_set=set() if filt is not None else naf_set, only_siege=only_siege)
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
                            # Pour CSV on laisse le filtre Pandas gérer NAF (canon_naf)
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
    """
    Retourne ['siren','nom_ul','statutDiffusionUniteLegale','unitePurgeeUniteLegale'].
    - Cherche d'abord data/unite_legale/ul_parts/*.parquet
    - Sinon, prend n'importe quel *.parquet dans data/unite_legale/
    - Détecte la colonne siren (casse/tpe), gère int vs string.
    """
    base_dir = ROOT / "data" / "unite_legale"

    if not sirens or not base_dir.exists():
        return pd.DataFrame(columns=["siren","nom_ul","statutDiffusionUniteLegale","unitePurgeeUniteLegale"])

    parts = sorted((base_dir / "ul_parts").glob("*.parquet"))
    if not parts:
        parts = sorted(base_dir.glob("*.parquet"))

    if not parts:
        return pd.DataFrame(columns=["siren","nom_ul","statutDiffusionUniteLegale","unitePurgeeUniteLegale"])

    dset = ds.dataset([str(p) for p in parts], format="parquet")

    # trouve la colonne siren quel que soit le nom exact / casse
    siren_col = next((c for c in dset.schema.names if c.lower() == "siren"), None)
    if not siren_col:
        return pd.DataFrame(columns=["siren","nom_ul","statutDiffusionUniteLegale","unitePurgeeUniteLegale"])

    t = dset.schema.field(siren_col).type
    UL_cols_wanted = [
        "denominationUniteLegale","denominationUsuelle1UniteLegale","denominationUsuelle2UniteLegale",
        "denominationUsuelle3UniteLegale","sigleUniteLegale","nomUsageUniteLegale","nomUniteLegale",
        "prenom1UniteLegale","prenomUsuelUniteLegale","pseudonymeUniteLegale",
        "statutDiffusionUniteLegale","unitePurgeeUniteLegale"
    ]
    cols = [siren_col] + [c for c in UL_cols_wanted if c in dset.schema.names]

    # normalise sirens demandés
    sirens = [re.sub(r"\D", "", s or "")[:9].zfill(9) for s in sirens if s]

    CHUNK = 60_000
    out = []

    def best_name(df):
        nom = (
            df.get("denominationUniteLegale").fillna("")
              .replace(r"^\s*$", pd.NA, regex=True)
              .fillna(df.get("denominationUsuelle1UniteLegale"))
              .fillna(df.get("denominationUsuelle2UniteLegale"))
              .fillna(df.get("denominationUsuelle3UniteLegale"))
              .fillna(df.get("sigleUniteLegale"))
        )
        prenom = df.get("prenom1UniteLegale", "").fillna("")
        nompp  = df.get("nomUsageUniteLegale", df.get("nomUniteLegale", "")).fillna("")
        nom = nom.fillna((prenom + " " + nompp).str.strip()).replace("", pd.NA)
        nom = nom.fillna(df.get("pseudonymeUniteLegale"))
        return nom

    for i in range(0, len(sirens), CHUNK):
        chunk = sirens[i:i+CHUNK]

        if pa.types.is_integer(t):
            ints = []
            for s in chunk:
                try: ints.append(int(s))
                except: pass
            if not ints: continue
            f = ds.field(siren_col).isin(pa.array(ints, type=t))
        else:
            variants = list({*chunk, *[s.lstrip("0") or "0" for s in chunk]})
            f = ds.field(siren_col).cast(pa.string()).isin(pa.array(variants, type=pa.string()))

        tbl = dset.to_table(columns=cols, filter=f)
        if tbl.num_rows == 0:
            continue
        df = tbl.to_pandas()

        # harmonise 'siren' → string 9
        df["siren"] = (
            df[siren_col].astype("string")
              .str.replace(r"\D", "", regex=True)
              .str.zfill(9).str[:9]
        )
        df["nom_ul"] = best_name(df)
        out.append(df[["siren","nom_ul","statutDiffusionUniteLegale","unitePurgeeUniteLegale"]])

    if not out:
        return pd.DataFrame(columns=["siren","nom_ul","statutDiffusionUniteLegale","unitePurgeeUniteLegale"])

    return pd.concat(out, ignore_index=True).drop_duplicates("siren", keep="first")

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
naf_typed = [canon_naf(c) for c in naf_input.split(",") if c.strip()]
naf_select_ms = [canon_naf(c) for c in st.session_state["naf_options"] if c]
naf_from_div = [canon_naf(c) for (c, _) in (sect_options_unique if take_all else secteurs_sel)]
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
        
    with st.expander("🔎 Debug NAF"):
        st.write("Échantillon NAF brut :", df[COLS["naf"]].astype(str).head(10).tolist())
        st.write("NAF sélectionnés (canon) :", sorted(list({canon_naf(x) for x in naf_final})))
    
    if df.empty:
        st.info("Aucune ligne avec ces filtres (NAF, siège, coordonnées) dans les départements sélectionnés.")
        st.stop()

    # === Base entreprises (avant jointure) ===
    df = df.copy()
    df.loc[:, "lat"] = pd.to_numeric(df[COLS["lat"]], errors="coerce")
    df.loc[:, "lon"] = pd.to_numeric(df[COLS["lon"]], errors="coerce")

    # Normalisation SIRET → SIREN (évite pertes de zéros, formats bizarres)
    siret_str = (
        df.get(COLS["siret"], "").astype("string")
          .str.replace(r"\D", "", regex=True)  # garde chiffres only
          .str.zfill(14).str[:14]
    )
    ent = pd.DataFrame({
        "siret":    siret_str,
        "nom_etab": df.apply(coalesce_name_etab, axis=1),
        "adresse":  df.get(COLS["adresse"], ""),
        "cp":       df.get(COLS["cp"], "").astype(str),
        "commune":  df.get(COLS["commune"], ""),
        "naf":      df.get(COLS["naf"], ""),
        "lat":      pd.to_numeric(df["lat"], errors="coerce"),
        "lon":      pd.to_numeric(df["lon"], errors="coerce"),
        "__dep__":  df["__dep__"],
        "__source__": df["__source__"],
    })
    ent["siren"] = ent["siret"].str[:9].str.zfill(9)

    # Contrôles avant jointure
    rows_before        = len(ent)
    siret_uniq_before  = ent["siret"].nunique()
    siren_uniq_before  = ent["siren"].nunique()

    # === Jointure UL ===
    with st.spinner("Jointure des noms d’Unité Légale…"):
        sirens_need = sorted(ent["siren"].dropna().unique().tolist())
        ul = load_ul_names_for(sirens_need)  # ['siren','nom_ul'] unique

    ent = ent.merge(ul, on="siren", how="left")

    # ✅ À GARDER (une seule fois)
    ent["nom_affiche"] = (
        ent["nom_ul"].fillna("")
        .replace(r"^\s*$", pd.NA, regex=True)
        .fillna(ent["nom_etab"])
    ).fillna("Nom non diffusible")

    # Diagnostic utile
    ul_vide = ent["nom_ul"].isna() | (ent["nom_ul"].astype(str).str.strip() == "")
    non_O   = (ent.get("statutDiffusionUniteLegale")  # peut ne pas exister si pas dans UL
                 .astype(str).ne("O") if "statutDiffusionUniteLegale" in ent.columns else (ul_vide & False))
    ul_purg = (ent.get("unitePurgeeUniteLegale")
                 .astype(str).isin(["true","True","1"]) if "unitePurgeeUniteLegale" in ent.columns else (ul_vide & False))

    st.caption(
        f"🔎 UL sans nom: {int(ul_vide.sum()):,} | "
        f"non-diffusibles: {int((ul_vide & non_O).sum()):,} | "
        f"UL purgées: {int((ul_vide & ul_purg).sum()):,} | "
        f"taux de match UL: {1 - (ul_vide.sum()/max(len(ent),1)):.1%}"
    )

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

