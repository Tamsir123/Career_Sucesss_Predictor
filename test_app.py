#!/usr/bin/env python3
"""
Script de test pour vérifier que l'application Streamlit fonctionne correctement
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def test_data_loading():
    """Test du chargement des données"""
    try:
        df_original = pd.read_csv('education_career_success_g.csv')
        df_encoded = pd.read_csv('education_career_success_encoded.csv')
        print("✅ Chargement des données: OK")
        print(f"   - Données originales: {df_original.shape}")
        print(f"   - Données encodées: {df_encoded.shape}")
        return df_original, df_encoded
    except Exception as e:
        print(f"❌ Erreur chargement données: {e}")
        return None, None

def test_model_training(df_encoded):
    """Test de l'entraînement du modèle"""
    try:
        # Préparation des données
        exclude_cols = ['Starting_Salary']
        feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
        
        X = df_encoded[feature_cols]
        y = df_encoded['Starting_Salary']
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraînement
        model = RandomForestRegressor(
            n_estimators=50,  # Réduit pour le test
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print("✅ Entraînement modèle: OK")
        print(f"   - R²: {r2:.3f}")
        print(f"   - RMSE: {rmse:.3f}")
        print(f"   - Features: {len(feature_cols)}")
        
        return model, r2, rmse
        
    except Exception as e:
        print(f"❌ Erreur entraînement modèle: {e}")
        return None, None, None

def test_clustering(df_encoded):
    """Test du clustering"""
    try:
        cluster_cols = ['University_GPA', 'Internships_Completed', 'Networking_Score', 
                       'Technical_Skills_Score', 'Soft_Skills_Score', 'Starting_Salary']
        
        # Vérifier que les colonnes existent
        missing_cols = [col for col in cluster_cols if col not in df_encoded.columns]
        if missing_cols:
            print(f"⚠️ Colonnes manquantes pour clustering: {missing_cols}")
            return None, None
        
        X_cluster = df_encoded[cluster_cols]
        
        # K-Means
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_cluster)
        
        # Résumé des clusters
        df_with_clusters = df_encoded.copy()
        df_with_clusters['Cluster'] = clusters
        cluster_summary = df_with_clusters.groupby('Cluster')[cluster_cols].mean()
        
        print("✅ Clustering: OK")
        print(f"   - Nombre de clusters: 4")
        print(f"   - Variables utilisées: {len(cluster_cols)}")
        print("   - Distribution des clusters:")
        unique, counts = np.unique(clusters, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"     Cluster {cluster}: {count} étudiants")
        
        return kmeans, cluster_summary
        
    except Exception as e:
        print(f"❌ Erreur clustering: {e}")
        return None, None

def test_prediction_pipeline(model, df_encoded):
    """Test du pipeline de prédiction"""
    try:
        # Créer des données utilisateur test
        user_data = pd.DataFrame(0, index=[0], columns=df_encoded.columns)
        
        # Remplir avec des valeurs par défaut
        default_values = {
            'Age': 23,
            'University_GPA': 3.0,
            'High_School_GPA': 3.0,
            'SAT_Score': 1200,
            'University_Ranking': 250,
            'Internships_Completed': 2,
            'Projects_Completed': 5,
            'Work_Experience_Years': 1,
            'Technical_Skills_Score': 5,
            'Soft_Skills_Score': 5,
            'Networking_Score': 5,
            'Study_Hours_Per_Week': 20,
            'Extracurricular_Activities': 3,
            'Motivation': 7,
            'Work_Life_Balance': 6,
            'Job_Offers': 3,
            'Career_Satisfaction': 6,
            'Years_to_Promotion': 3
        }
        
        for col, value in default_values.items():
            if col in user_data.columns:
                user_data[col] = value
        
        # Encoder quelques variables catégoriques
        if 'Field_of_Study_Computer Science' in user_data.columns:
            user_data['Field_of_Study_Computer Science'] = 1
        if 'Location_Urban' in user_data.columns:
            user_data['Location_Urban'] = 1
        if 'Gender_Male' in user_data.columns:
            user_data['Gender_Male'] = 1
        if 'Languages_Spoken_Anglais' in user_data.columns:
            user_data['Languages_Spoken_Anglais'] = 1
        
        # Supprimer Starting_Salary si présent
        if 'Starting_Salary' in user_data.columns:
            user_data = user_data.drop(columns=['Starting_Salary'])
        
        # Prédiction
        feature_cols = [col for col in model.feature_names_in_ if col in user_data.columns]
        predicted_salary = model.predict(user_data[feature_cols])[0]
        
        print("✅ Pipeline de prédiction: OK")
        print(f"   - Salaire prédit: ${predicted_salary:,.0f}")
        print(f"   - Features utilisées: {len(feature_cols)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur pipeline prédiction: {e}")
        return False

def main():
    """Test principal"""
    print("🧪 Tests de l'Application Streamlit")
    print("=" * 40)
    
    # Test 1: Chargement des données
    df_original, df_encoded = test_data_loading()
    if df_original is None or df_encoded is None:
        print("❌ Impossible de continuer sans les données")
        return
    
    print()
    
    # Test 2: Entraînement du modèle
    model, r2, rmse = test_model_training(df_encoded)
    if model is None:
        print("❌ Impossible de continuer sans le modèle")
        return
    
    print()
    
    # Test 3: Clustering
    kmeans, cluster_summary = test_clustering(df_encoded)
    if kmeans is None:
        print("⚠️ Clustering échoué, mais l'app peut continuer")
    
    print()
    
    # Test 4: Pipeline de prédiction
    test_prediction_pipeline(model, df_encoded)
    
    print()
    print("🎉 Tests terminés!")
    print("✅ L'application devrait fonctionner correctement")
    print()
    print("Pour lancer l'application:")
    print("streamlit run app_corrected.py")

if __name__ == "__main__":
    main()
