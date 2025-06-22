#!/usr/bin/env python3
"""
Test rapide des deux analyses d'équité (géographie + genre)
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    from fairlearn.metrics import demographic_parity_difference
    FAIRLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ Fairlearn non installé")
    FAIRLEARN_AVAILABLE = False

def test_both_equity_analyses():
    """Test des deux analyses d'équité"""
    
    if not FAIRLEARN_AVAILABLE:
        print("❌ Fairlearn non disponible")
        return
    
    print("🚀 Test Complet : Équité Géographie + Genre")
    print("=" * 55)
    
    # Chargement des données
    try:
        df_encoded = pd.read_csv('education_career_success_encoded.csv')
        print(f"📊 Données chargées: {len(df_encoded):,} observations")
    except FileNotFoundError:
        print("❌ Fichier non trouvé")
        return
    
    # Début du chronométrage global
    start_total = time.time()
    
    # === ANALYSE GÉOGRAPHIE ===
    print("\n🏢 ANALYSE GÉOGRAPHIE...")
    start_geo = time.time()
    
    # Échantillonnage
    sample_size = min(3000, len(df_encoded))
    df_sample = df_encoded.sample(n=sample_size, random_state=42)
    
    # Features et variables
    important_features = ['GPA', 'Internships_Completed', 'Extracurricular_Activities',
                         'Leadership_Positions', 'Networking_Events_Attended', 'Personal_Projects']
    available_features = [f for f in important_features if f in df_sample.columns]
    
    X = df_sample[available_features]
    y = df_sample['Starting_Salary']
    
    # Variable sensible géographie
    sensitive_geo = np.where(df_sample['Location_Rural'] == 1, 'Rural',
                           np.where(df_sample.get('Location_Urban', 0) == 1, 'Urban', 'International'))
    sensitive_geo = pd.Series(sensitive_geo, index=df_sample.index)
    
    # Binarisation et split
    y_binary = (y > y.median()).astype(int)
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y_binary, sensitive_geo, test_size=0.2, random_state=42
    )
    
    # Modèle équitable géographie
    base_model = LogisticRegression(max_iter=50, solver='liblinear', random_state=42)
    constraint = DemographicParity()
    fair_model_geo = ExponentiatedGradient(base_model, constraint, eps=0.2, max_iter=5)
    fair_model_geo.fit(X_train, y_train, sensitive_features=s_train)
    
    y_pred_geo = fair_model_geo.predict(X_test)
    dp_diff_geo = demographic_parity_difference(y_test, y_pred_geo, sensitive_features=s_test)
    
    time_geo = time.time() - start_geo
    print(f"   ✅ Géographie terminée en {time_geo:.2f}s - DP Diff: {dp_diff_geo:.3f}")
    
    # === ANALYSE GENRE ===
    print("\n👥 ANALYSE GENRE...")
    start_gender = time.time()
    
    if 'Gender_Male' in df_sample.columns:
        # Variable sensible genre
        sensitive_gender = np.where(df_sample['Gender_Male'] == 1, 'Male', 'Female')
        sensitive_gender = pd.Series(sensitive_gender, index=df_sample.index)
        
        # Split pour genre
        X_train_g, X_test_g, y_train_g, y_test_g, s_train_g, s_test_g = train_test_split(
            X, y_binary, sensitive_gender, test_size=0.2, random_state=42
        )
        
        # Modèle équitable genre
        fair_model_gender = ExponentiatedGradient(base_model, constraint, eps=0.2, max_iter=5)
        fair_model_gender.fit(X_train_g, y_train_g, sensitive_features=s_train_g)
        
        y_pred_gender = fair_model_gender.predict(X_test_g)
        dp_diff_gender = demographic_parity_difference(y_test_g, y_pred_gender, sensitive_features=s_test_g)
        
        time_gender = time.time() - start_gender
        print(f"   ✅ Genre terminé en {time_gender:.2f}s - DP Diff: {dp_diff_gender:.3f}")
        
        gender_available = True
    else:
        print("   ❌ Variable Gender_Male non trouvée")
        gender_available = False
        time_gender = 0
        dp_diff_gender = 0
    
    # === RÉSULTATS GLOBAUX ===
    total_time = time.time() - start_total
    
    print(f"\n📈 RÉSULTATS GLOBAUX:")
    print(f"⏱️  Temps total: {total_time:.2f}s")
    print(f"🏢 Géographie: {time_geo:.2f}s")
    if gender_available:
        print(f"👥 Genre: {time_gender:.2f}s")
    
    print(f"\n🎯 MÉTRIQUES D'ÉQUITÉ:")
    print(f"🏢 DP Diff Géographie: {dp_diff_geo:.3f} ({'✅ Excellent' if abs(dp_diff_geo) < 0.05 else '🟡 Bon' if abs(dp_diff_geo) < 0.1 else '🚨 Critique'})")
    
    if gender_available:
        print(f"👥 DP Diff Genre: {dp_diff_gender:.3f} ({'✅ Excellent' if abs(dp_diff_gender) < 0.05 else '🟡 Bon' if abs(dp_diff_gender) < 0.1 else '🚨 Critique'})")
        
        # Score global
        geo_score = 100 if abs(dp_diff_geo) < 0.05 else 50 if abs(dp_diff_geo) < 0.1 else 0
        gender_score = 100 if abs(dp_diff_gender) < 0.05 else 50 if abs(dp_diff_gender) < 0.1 else 0
        global_score = (geo_score + gender_score) / 2
        
        print(f"\n🌟 SCORE GLOBAL D'ÉQUITÉ: {global_score:.0f}/100")
        if global_score >= 90:
            print("🎉 EXCELLENT: Modèle équitable sur toutes les dimensions !")
        elif global_score >= 70:
            print("✅ BON: Équité satisfaisante")
        else:
            print("⚠️ À AMÉLIORER: Biais significatifs détectés")
    
    # Verdict de vitesse
    if total_time < 5:
        print(f"\n🚀 PERFORMANCE: Ultra-rapide (< 5s)")
    elif total_time < 10:
        print(f"\n✅ PERFORMANCE: Rapide (< 10s)")
    else:
        print(f"\n⚠️ PERFORMANCE: Peut être optimisé davantage")

if __name__ == "__main__":
    test_both_equity_analyses()
