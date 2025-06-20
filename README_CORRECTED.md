# 🎓 Système Hybride: Recommandations de Carrière & Prédiction de Salaire

## 📋 Résumé du Projet

Ce projet implémente un **système hybride d'intelligence artificielle** qui combine plusieurs techniques d'apprentissage automatique pour :

1. **Recommander des choix de carrière optimaux** basés sur le profil de l'étudiant
2. **Prédire le salaire de départ** en tenant compte des interactions complexes
3. **Analyser l'équité** pour corriger les biais potentiels

## 🎯 Problématique

*Comment un modèle prédictif basé sur les caractéristiques académiques, professionnelles et personnelles des étudiants peut-il simultanément recommander des choix de carrière optimaux et estimer leur salaire de départ, tout en tenant compte des interactions complexes entre les compétences, le domaine d'étude, les opportunités de réseautage et les préférences individuelles, afin de maximiser leur satisfaction professionnelle et leur réussite financière ?*

## 🔬 Méthodologie Hybride

### 1. 🎯 Clustering K-Means 
- **Objectif** : Segmenter les étudiants en profils types similaires
- **Variables** : GPA, Stages, Compétences techniques/relationnelles, Réseautage, Salaire
- **Résultat** : 4 clusters identifiés avec caractéristiques distinctes

### 2. 🤖 Modèles de Régression
- **Random Forest** et **XGBoost** pour prédire le salaire de départ
- **Performance** : R² ≈ 0.77, RMSE ≈ 0.48 (sur données standardisées)
- **Variables importantes** : Stages, Compétences techniques, Expérience, Réseautage

### 3. 💡 Système de Recommandation Hybride
- **Clustering** : Position par rapport aux profils types
- **Analyse d'impact** : Simulation de l'effet des améliorations
- **Recommandations** : Actions prioritaires personnalisées et hiérarchisées

### 4. ⚖️ Analyse d'Équité
- **Détection** des biais (Genre, Localisation)
- **Correction** avec Fairlearn (Demographic Parity)
- **Résultat** : Réduction du biais géographique de 0.87 à 0.03

## 📊 Résultats Clés

### Profils des 4 Clusters Identifiés

- **Cluster 0** : **Performants avec Expérience Pratique** (33.8%)
  - Beaucoup de stages, bonnes compétences techniques, salaire élevé

- **Cluster 1** : **Sociables en Développement Technique** (33.3%)
  - Excellentes soft skills, peu de stages, potentiel d'amélioration

- **Cluster 2** : **Réseautés aux Compétences Mixtes** (18.7%)
  - Bon réseautage, compétences techniques correctes, faibles soft skills

- **Cluster 3** : **Profils en Développement** (14.2%)
  - Compétences techniques très faibles, nécessite accompagnement intensif

### Facteurs d'Influence Majeurs
1. **Stages** (Internships_Completed) - Impact le plus élevé
2. **Compétences techniques** - Essentielles dans l'économie numérique
3. **Expérience professionnelle** - Chaque année compte
4. **Réseautage** - Peut compenser des lacunes académiques

### Analyse d'Équité
- ✅ **Genre** : Équité confirmée (écart < 1%)
- ⚠️ **Localisation** : Biais initial détecté et corrigé
- 📈 **Amélioration** : Demographic Parity Difference 0.87 → 0.03

## 🚀 Structure du Projet

```
projet/
├── education_career_success_g.csv          # Données originales
├── education_career_success_encoded.csv    # Données encodées (multi-label + one-hot)
├── education_career_sucess.ipynb          # Notebook d'analyse complète
├── app_corrected.py                       # Application Streamlit corrigée
├── test_app.py                           # Script de test
├── CORRECTIONS_APPLIQUEES.md             # Documentation des corrections
└── README.md                             # Ce fichier
```

## 🛠️ Installation et Utilisation

### Prérequis
```bash
pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib
pip install xgboost fairlearn shap joblib
```

### Lancement de l'Application
```bash
streamlit run app_corrected.py
```

### Test de l'Application
```bash
python test_app.py
```

## 📱 Fonctionnalités de l'Application

### 🏠 Accueil
- Présentation de la problématique et méthodologie
- Métriques clés du projet
- Découvertes principales

### 📊 Exploration
- Aperçu des données originales et encodées
- Statistiques descriptives
- Analyse des salaires par domaine

### 🎯 Clustering
- Visualisation des 4 profils types
- Analyse radar des clusters
- Distribution et interprétation détaillée

### 🤖 Prédiction
- Performances du modèle Random Forest
- Importance des variables
- Interface de prédiction instantanée

### 💡 Recommandations Hybrides
- Saisie complète du profil utilisateur
- Identification du cluster d'appartenance
- Recommandations prioritaires personnalisées
- Plan d'action à 6 mois
- Estimation de gain potentiel

## 🔧 Corrections Appliquées

### Problèmes Identifiés dans l'Application Originale
1. **Mauvaise gestion des données encodées** - L'app ne comprenait pas la structure après encodage multi-label
2. **Variables manquantes** pour la prédiction - Incompatibilité entre features du modèle
3. **Système de recommandation simpliste** - Ne correspondait pas à l'approche hybride du notebook
4. **Absence de cohérence** entre notebook et application

### Solutions Implémentées
1. **Chargement dual** : Données originales (exploration) + encodées (prédiction)
2. **Pipeline correct** : Respect de l'encodage multi-label (langues, certifications) + one-hot
3. **Système hybride** : Clustering + Random Forest + analyse comparative
4. **Interface complète** : 47 variables encodées prises en compte
5. **Tests automatisés** : Validation du fonctionnement

## 📈 Impact et Applications

### Pour les Étudiants
- **Orientation personnalisée** basée sur l'IA
- **Recommandations actionables** avec plan détaillé
- **Prédiction fiable** du salaire de départ
- **Identification** des axes d'amélioration prioritaires

### Pour les Institutions
- **Segmentation** des étudiants en 4 profils types
- **Correction des biais** géographiques identifiés
- **Accompagnement ciblé** selon le profil
- **Métriques d'impact** mesurables

## 🎯 Recommandations Stratégiques

### Basées sur les Résultats
1. **Priorité absolue aux stages** - Facteur #1 d'influence sur le salaire
2. **Développement des compétences techniques** via projets concrets
3. **Cultivation du réseau professionnel** dès la première année
4. **Correction des inégalités géographiques** (focus étudiants ruraux)

## 👥 Équipe et Contributions

- **TAMSIR Aboubacar** : BD complète, analyse exploratoire, modélisation, Streamlit
- **BELEM Zakaria** : BD normale, analyse exploratoire, modélisation, Streamlit  
- **Amina OUEDRAOGO** : Contexte, problématique, objectifs

## 📝 Méthodologie Scientifique

### Algorithmes Utilisés
- **K-Means** : Clustering non supervisé (4 clusters optimaux)
- **Random Forest** : Régression supervisée (200 arbres, profondeur 10)
- **XGBoost** : Régression alternative pour comparaison
- **Fairlearn** : Correction des biais avec contrainte de parité démographique

### Métriques d'Évaluation
- **R²** : Variance expliquée (≈ 0.77)
- **RMSE** : Erreur quadratique moyenne 
- **Score de Silhouette** : Qualité du clustering
- **Demographic Parity Difference** : Mesure d'équité

### Validation
- **Validation croisée** 5-fold pour la robustesse
- **Train/Test split** 80/20 pour l'évaluation
- **Tests automatisés** pour la fiabilité du code

## 🚀 Perspectives d'Amélioration

1. **Modèles avancés** : Réseaux de neurones, Gradient Boosting optimisé
2. **Variables supplémentaires** : Données économiques, sectorielles
3. **Interface mobile** : Application smartphone pour accessibility
4. **API REST** : Intégration dans d'autres systèmes
5. **Suivi longitudinal** : Validation des prédictions dans le temps

---

*Système développé dans le cadre du cours "Types d'apprentissage ML" - S6 AI*
