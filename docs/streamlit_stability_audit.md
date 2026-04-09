# Audit de stabilité Streamlit — AgriBiomasse

_Date : 2026-04-09_

Ce document liste les points de fragilité observés dans `app.py` avec une priorisation orientée **stabilité en production Streamlit/Streamlit Cloud**.

## Critique

1. **Filtre NAF de la multiselect du scan ignoré dans le chargement final**
   - Le code récupère la sélection utilisateur dans `naf_select_ms = st.multiselect(...)`, mais la variable est ensuite écrasée avec **toutes** les options scannées (`st.session_state["naf_options"]`) au lieu des choix effectifs.
   - Risque : chargement massif involontaire (plus de départements + plus de NAF) ⇒ mémoire et temps explosent.

2. **Passage en Pandas trop large après lecture Arrow (`to_table(...).to_pandas()`)**
   - Même avec filtre NAF, les autres filtres (`etat`, `siege`, lat/lon valides) sont appliqués ensuite en Pandas.
   - Risque : pic mémoire élevé avant réduction des données.

3. **Rendu carte Folium avec un marker par ligne (`iterrows`)**
   - Un marker + popup HTML par entreprise, puis rendu complet `st_folium`.
   - Risque : freeze navigateur, payload HTML trop lourd, plantage de session Streamlit.

## Important

4. **Absence de garde-fou de volumétrie avant rendu carte**
   - Pas de limite stricte du nombre de points affichés ou exportés.
   - Risque : un filtre large suffit à saturer RAM/CPU.

5. **`load_filtered` non caché**
   - Chaque rerun après clic “Charger la carte” relance les lectures disque si un widget change.
   - Risque : latence et coûts I/O répétés.

6. **Calculs ligne-à-ligne coûteux (`apply(axis=1)`)**
   - `coalesce_name_etab`, liens Google/PJ exécutés pour chaque ligne.
   - Risque : CPU élevé sur gros DataFrame.

7. **Code mort après `return` dans `load_ul_names_for`**
   - Bloc non exécuté conservé après un `return`.
   - Risque : maintenance difficile, confusion, dette technique.

## Optionnel

8. **Constantes métier volumineuses dans `app.py` (NAF_DIVISIONS)**
   - Rend le fichier monolithique et plus lent à parser.

9. **Script `tools/shrink_to_parquet.py` vide**
   - Le README recommande ce script, mais le fichier est vide.
   - Risque : confusion opérationnelle (pipeline data non reproductible).

10. **Data bundle très volumineux pour Streamlit Cloud**
    - Le repo local pèse plusieurs Go.
    - Risque : build lent/instable, limites de stockage/mémoire Cloud.

## Plan 3 étapes (sans refonte)

### Étape 1 — Stabilisation immédiate
- Corriger l’utilisation de la multiselect NAF (prendre seulement les codes choisis).
- Ajouter une limite de sécurité configurable (ex. `MAX_POINTS=20_000`) + message utilisateur avant rendu carte.
- Ajouter une option “Aperçu rapide” (échantillon ou top N) par défaut quand volume élevé.

### Étape 2 — Performance
- Mettre `load_filtered` en cache (`@st.cache_data`) avec clés d’entrée strictes.
- Pousser davantage de filtres côté Arrow (état actif, siège, coordonnées non nulles si colonnes présentes).
- Réduire les `apply(axis=1)` en remplaçant par opérations vectorisées quand possible.

### Étape 3 — Nouvelles fonctionnalités
- Ajouter mode carte “densité / agrégation” (heatmap ou clusters pré-agrégés).
- Ajouter recherche directe SIREN/SIRET avec focus carte.
- Ajouter “profil département” (compte par NAF, actifs, sièges) avant affichage des points.

## Streamlit Cloud — recommandations explicites
- Ne pas embarquer tout le dataset dans le repo de déploiement; privilégier stockage externe (S3/GCS/Blob) + chargement ciblé.
- Garder uniquement Parquet partitionné et colonnes minimales nécessaires.
- Mettre des limites strictes (points max, taille export max, timeout de scan NAF).
- Utiliser `st.cache_data(ttl=...)` sur lectures coûteuses et invalider proprement sur changement de sources.

## API SIRENE — approche progressive et robuste
- Commencer par une intégration **complémentaire** (fallback/enrichissement ciblé), pas un remplacement total des fichiers locaux.
- Cas d’usage recommandé :
  1. Validation/complément ponctuel sur un petit lot de SIREN/SIRET,
  2. Synchronisation incrémentale nocturne,
  3. Cache local des réponses API (évite quotas et latence).
- Ajouter un circuit simple “si API indisponible ⇒ continuer sur données locales” pour préserver la robustesse.
