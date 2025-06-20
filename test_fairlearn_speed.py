#!/usr/bin/env python3
"""
Script de test pour vérifier la vitesse d'exécution de la fonction Fairlearn optimisée
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    from fairlearn.metrics import demographic_parity_difference
    FAIRLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ Fairlearn non installé")
    FAIRLEARN_AVAILABLE = False

def load_data():
    """Chargement des données"""
    try:
        df_original = pd.read_csv('education_career_success_g.csv')
        df_encoded = pd.read_csv('education_career_success_encoded.csv')
        return df_original, df_encoded
    except FileNotFoundError:
        print("❌ Fichiers de données non trouvés")
        return None, None

def test_fairlearn_speed():
    """Test de vitesse de la fonction Fairlearn optimisée"""
    
    if not FAIRLEARN_AVAILABLE:
        print("❌ Fairlearn non disponible")
        return
    
    print("🚀 Test de Vitesse Fairlearn Ultra-Optimisé")
    print("=" * 50)
    
    # Chargement des données
    df_original, df_encoded = load_data()
    if df_encoded is None:
        return
    
    print(f"📊 Données chargées: {len(df_encoded):,} observations")
    
    # Début du chronométrage
    start_time = time.time()
    
    try:
        # Échantillonnage ultra-réduit
        sample_size = min(3000, len(df_encoded))
        df_sample = df_encoded.sample(n=sample_size, random_state=42)
        print(f"🎯 Échantillon utilisé: {sample_size:,} observations")
        
        # Features importantes seulement
        important_features = [
            'GPA', 'Internships_Completed', 'Extracurricular_Activities',
            'Leadership_Positions', 'Networking_Events_Attended', 'Personal_Projects',
            'Location_International', 'Location_Rural', 'Location_Urban'
        ]
        
        available_features = [f for f in important_features if f in df_sample.columns]
        print(f"🔧 Features utilisées: {len(available_features)}")
        
        X = df_sample[available_features]
        y = df_sample['Starting_Salary']
        
        # Variable sensible (vectorisée)
        if 'Location_Rural' in df_sample.columns:
            sensitive_feature = np.where(df_sample['Location_Rural'] == 1, 'Rural',
                                       np.where(df_sample.get('Location_Urban', 0) == 1, 'Urban', 'International'))
        else:
            sensitive_feature = ['International'] * len(df_sample)
        
        sensitive_feature = pd.Series(sensitive_feature, index=df_sample.index)
        
        # Binarisation
        salary_threshold = y.median()
        y_binary = (y > salary_threshold).astype(int)
        
        # Split
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y_binary, sensitive_feature, test_size=0.2, random_state=42
        )
        
        print(f"✂️ Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Modèle non équitable (référence)
        unfair_model = LogisticRegression(max_iter=50, solver='liblinear', random_state=42)
        unfair_model.fit(X_train, y_train)
        y_pred_unfair = unfair_model.predict(X_test)
        unfair_dp_diff = demographic_parity_difference(y_test, y_pred_unfair, sensitive_features=s_test)
        
        # Modèle équitable (chronométré)
        fairlearn_start = time.time()
        
        base_model = LogisticRegression(max_iter=50, solver='liblinear', random_state=42)
        constraint = DemographicParity()
        fair_model = ExponentiatedGradient(
            base_model, 
            constraint,
            eps=0.2,
            max_iter=5,
            nu=1e-6
        )
        
        fair_model.fit(X_train, y_train, sensitive_features=s_train)
        y_pred_fair = fair_model.predict(X_test)
        fair_dp_diff = demographic_parity_difference(y_test, y_pred_fair, sensitive_features=s_test)
        
        fairlearn_time = time.time() - fairlearn_start
        
        # Fin du chronométrage total
        total_time = time.time() - start_time
        
        # Résultats
        print("\n📈 RÉSULTATS:")
        print(f"⏱️  Temps total: {total_time:.2f}s")
        print(f"⚡ Temps Fairlearn: {fairlearn_time:.2f}s")
        print(f"📊 DP Difference avant: {unfair_dp_diff:.3f}")
        print(f"✅ DP Difference après: {fair_dp_diff:.3f}")
        
        improvement = ((abs(unfair_dp_diff) - abs(fair_dp_diff)) / abs(unfair_dp_diff)) * 100 if abs(unfair_dp_diff) > 0 else 0
        print(f"🎯 Amélioration: {improvement:.1f}%")
        
        # Groupes uniques
        unique_groups = np.unique(s_test)
        print(f"🏢 Groupes géographiques: {list(unique_groups)}")
        
        # Performance par groupe
        for group in unique_groups:
            group_mask = s_test == group
            if np.sum(group_mask) > 0:
                unfair_rate = np.mean(y_pred_unfair[group_mask])
                fair_rate = np.mean(y_pred_fair[group_mask])
                print(f"   {group}: {unfair_rate:.1%} → {fair_rate:.1%}")
        
        # Verdict de vitesse
        if total_time < 10:
            print("🚀 EXCELLENT: Ultra-rapide (< 10s)")
        elif total_time < 30:
            print("✅ BON: Rapide (< 30s)")
        else:
            print("⚠️ LENT: Peut être optimisé davantage")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print(f"⏱️ Temps avant erreur: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    test_fairlearn_speed()
