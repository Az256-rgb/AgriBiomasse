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

## Addendum — plan minimal (3 problèmes critiques uniquement)

### 1) Bug de sélection NAF (sélection utilisateur ignorée)
- **Fichier à modifier** : `app.py`
- **Zone exacte** : bloc UI NAF autour des lignes `naf_select_ms = st.multiselect(...)` puis fusion `naf_final`.
- **Petit changement recommandé** :
  - Conserver la valeur retournée par la multiselect (ex: `naf_selected_from_scan`) ;
  - Dans la fusion finale, utiliser cette sélection au lieu de `st.session_state["naf_options"]`.
- **Taille du patch** : **très légère** (quelques lignes).
- **Risque de régression** : faible (la logique devient conforme à l’UI).
- **Vérifications manuelles** :
  1. Scanner 2 départements, sélectionner seulement 1–2 NAF dans la liste scannée.
  2. Vérifier que la caption “Codes NAF retenus” contient seulement ces NAF.
  3. Vérifier que le volume chargé baisse par rapport à “tous les NAF scannés”.

### 2) Surcharge de rendu carte (un marker par ligne)
- **Fichier à modifier** : `app.py`
- **Zone exacte** : section carte avant la boucle `for _, r in df_src.iterrows():`.
- **Petit changement recommandé** :
  - Ajouter un garde-fou `MAX_POINTS_MAP` (ex: 20_000) ;
  - Si `len(ent)` dépasse le seuil, afficher `st.warning(...)` + ne pas construire tous les markers (ou ne garder qu’un échantillon explicite).
- **Taille du patch** : **légère**.
- **Risque de régression** : faible à moyen (des utilisateurs verront moins de points d’un coup, mais l’app ne plantera plus).
- **Vérifications manuelles** :
  1. Cas petit volume (< seuil) : comportement inchangé.
  2. Cas gros volume (> seuil) : warning visible, app reste réactive, pas de crash navigateur.
  3. Export CSV toujours fonctionnel (si conservé hors garde-fou carte).

### 3) Pic mémoire lors du chargement (Arrow → Pandas trop large)
- **Fichier à modifier** : `app.py`
- **Zone exacte** : `load_filtered(...)`, partie parquet.
- **Petit changement recommandé** :
  - Ajouter des filtres Arrow simples et robustes avant `to_pandas()` :
    - `etatAdministratifEtablissement` actif (si colonne présente),
    - `etablissementSiege` si option active (si colonne présente),
    - `latitude`/`longitude` non nulles (si colonnes présentes).
  - Garder le fallback Pandas actuel pour compatibilité.
- **Taille du patch** : **moyenne légère** (quelques conditions supplémentaires dans la construction du filtre).
- **Risque de régression** : moyen (hétérogénéité des schémas/fichiers départementaux).
- **Vérifications manuelles** :
  1. Lancer sur 1 département puis 5+ départements, comparer temps de chargement.
  2. Vérifier que les résultats restent cohérents (actifs, sièges, coordonnées valides).
  3. Vérifier le fallback si un fichier a un schéma partiel.

## Micro-plan d’exécution (sans refactor)
1. **Patch #1 (NAF)** : corriger la variable utilisée dans `naf_final`.
2. **Patch #2 (carte)** : ajouter seuil de sécurité avant création des markers.
3. **Patch #3 (Arrow)** : ajouter filtres “sûrs” côté Arrow + conserver filtre Pandas en sécurité.
4. **Validation rapide** : test manuel 3 scénarios (petit / moyen / gros volume) + export CSV + reset UI.

Ce plan reste strictement minimal et localisé à `app.py`, sans changement d’architecture globale.
