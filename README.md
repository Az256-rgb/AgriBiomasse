# Carte entreprises par NAF (France) — Streamlit

Application **Streamlit** pour afficher sur une carte interactive les entreprises françaises (par **codes NAF**) à partir de fichiers **par département** stockés **dans ce dépôt GitHub**.
Une couche **“Méthaniseurs”** (optionnelle) peut être ajoutée. Export CSV des points affichés.

---

## ✨ Fonctionnalités

* Sélection **par département** (tous ou au cas par cas).
* Filtre **par code NAF** via menu déroulant (liste détectée depuis vos fichiers).
* Popups avec **SIRET, adresse**, liens **Google Maps** et **PagesJaunes**.
* **Export CSV** des entreprises actuellement affichées.
* **Couche “Méthaniseurs”** optionnelle (CSV/Parquet).
* Compatible **Parquet (.parquet)**, **CSV.gz (.csv.gz)** et **CSV**.

---

## 🗂️ Structure du dépôt

```
fromageries-map/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ data/
│  ├─ entreprises/
│  │  ├─ geo_siret_01.parquet
│  │  ├─ geo_siret_02.parquet
│  │  ├─ ...
│  │  ├─ geo_siret_2A.parquet
│  │  └─ geo_siret_95.parquet
│  └─ methaniseurs/
│     └─ methaniseurs.parquet   # (optionnel) ou .csv
└─ tools/
   └─ shrink_to_parquet.py      # script local pour convertir CSV → Parquet
```

> **Nom des fichiers entreprises** : `geo_siret_XX.*` avec `XX ∈ {01..95, 2A, 2B}`.
> **Formats acceptés** : `.parquet` (recommandé), `.csv.gz`, `.csv`.

---

## 📦 Données attendues (entreprises)

Vos fichiers par département suivent le schéma SIRENE (colonnes principales utilisées par l’app) :

* `siret`
* `etatAdministratifEtablissement` *(filtre “Actif”)*
* `activitePrincipaleEtablissement` *(code NAF)*
* `enseigne1Etablissement`, `denominationUsuelleEtablissement` *(nom affiché)*
* `longitude`, `latitude` *(obligatoires)*
* `geo_adresse` *(adresse affichée)*
* `codePostalEtablissement`, `libelleCommuneEtablissement`
* `etablissementSiege` *(option “sièges uniquement”)*

**Méthaniseurs (optionnel)** : colonnes simples `nom`, `adresse`, `latitude`, `longitude` (ou variantes proches).

---

## 🚀 Déployer en ligne (Streamlit Community Cloud)

1. **Créer le dépôt** (ou forker/copier ce répertoire).
2. Vérifier que `data/entreprises/` contient vos `geo_siret_XX.parquet` (ou `.csv.gz` / `.csv`).
3. Aller sur **[https://streamlit.io](https://streamlit.io) → Sign in → Deploy an app**.
4. Sélectionner votre repo, branche `main`, script `app.py` → **Deploy**.
5. Une URL publique `https://…streamlit.app` est générée : partagez-la à vos collègues.

> **Astuce taille** : GitHub limite ~100 MB/fichier (upload web ≈ 25 MB). Utilisez **Parquet** (compressé) via `tools/shrink_to_parquet.py` pour réduire fortement la taille.

---

## 🧑‍💻 Lancer en local

```bash
# Python 3.10+ recommandé
pip install -r requirements.txt
streamlit run app.py
# Ouvrez http://localhost:8501
```

---

## 🕹️ Utilisation

1. **Sélection des départements** : tous ou multi-sélection.
2. **Codes NAF** : choisissez un ou plusieurs codes proposés (détectés dans vos fichiers).
3. **Options** :

   * *Ne garder que les sièges* (`etablissementSiege=1`)
   * *Afficher la couche “Méthaniseurs”* si le fichier existe.
4. **Carte** : points clusterisés ; popup avec infos + liens **Google Maps** / **PagesJaunes**.
5. **Export** : bouton **Télécharger CSV** des **données affichées**.

---

## 📉 Performances & bonnes pratiques

* **Parquet** (Snappy) recommandé : lecture rapide et fichiers 10–20× plus petits que CSV.
* **Réduire les colonnes** aux champs listés ci-dessus (moins d’E/S, temps de chargement réduit).
* Éviter de charger des **millions** de points d’un coup : filtrer par départements et NAF.

---

## 🛠️ Préparer les données (conversion CSV → Parquet)

Si vos CSV bruts sont lourds, utilisez le script fourni :

```bash
mkdir -p raw data/entreprises
# placez vos CSV bruts (>100 Mo) dans raw/
python tools/shrink_to_parquet.py
# des geo_siret_XX.parquet optimisés seront écrits dans data/entreprises/
```

Le script :

* lit par **chunks**,
* conserve uniquement les **colonnes utiles**,
* normalise le **code NAF**,
* filtre les lignes sans coordonnées,
* écrit en **Parquet** compressé.

---

## 🔒 Données & conformité

* Données issues de SIRENE : veillez à respecter les règles de diffusion (CNIL, réutilisation).
* Évitez de publier des informations personnelles non nécessaires.

---

## 📜 Licence

Ce projet peut fonctionner sous **MIT** (à adapter selon votre préférence).

---

## 🤝 Contributions

Issues/PR bienvenues : corrections, améliorations UI, nouveaux filtres (rayon géographique, recherche texte, libellés NAF, etc.).

---

*Bon déploiement !*
