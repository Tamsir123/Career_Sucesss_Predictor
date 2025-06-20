import streamlit as st
import pandas as pd
import numpy as np

def load_encoded_data():
    """Charge les données encodées comme dans le notebook"""
    try:
        # Essayer de charger le fichier encodé d'abord
        df = pd.read_csv('education_career_success_encoded.csv')
        st.success("✅ Données encodées chargées avec succès")
        return df
    except FileNotFoundError:
        try:
            # Sinon charger et encoder à la volée
            df = pd.read_csv('education_career_success_g.csv')
            df = encode_data_like_notebook(df)
            st.success("✅ Données chargées et encodées à la volée")
            return df
        except FileNotFoundError:
            st.error("❌ Aucun fichier de données trouvé")
            return None

def encode_data_like_notebook(df):
    """Encode les données exactement comme dans votre notebook"""
    
    # Supprimer Student_ID
    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
    
    # Gestion des valeurs manquantes (comme dans le notebook)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in ['Languages_Spoken', 'Certifications']:
            df[col].fillna('', inplace=True)
        else:
            df[col].fillna('Unknown', inplace=True)
    
    # Encodage multi-label pour Languages_Spoken
    if 'Languages_Spoken' in df.columns:
        languages_split = df['Languages_Spoken'].str.split(',')
        all_languages = set()
        for lang_list in languages_split:
            if isinstance(lang_list, list):
                all_languages.update(lang_list)
        
        for language in all_languages:
            if language.strip():  # Ignorer les chaînes vides
                df[f'Languages_Spoken_{language.strip()}'] = df['Languages_Spoken'].str.contains(language.strip(), na=False).astype(int)
    
    # Encodage multi-label pour Certifications  
    if 'Certifications' in df.columns:
        for i in range(6):  # 0 à 5 certifications
            df[f'Certifications_{i}'] = (df['Certifications'] == i).astype(int)
    
    # Encodage one-hot pour les autres variables catégoriques
    categorical_to_encode = ['Field_of_Study', 'Location', 'Gender', 'Current_Job_Level', 
                           'Entrepreneurship', 'Remote_Work']
    
    for col in categorical_to_encode:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
    
    # Supprimer les colonnes catégoriques originales
    cols_to_drop = ['Languages_Spoken', 'Certifications'] + categorical_to_encode
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    return df

def display_data_info(df):
    """Affiche les informations sur les données"""
    
    st.markdown("### 📊 Informations sur les Données")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔢 Nombre d'échantillons", len(df))
    
    with col2:
        st.metric("📋 Nombre de variables", len(df.columns))
    
    with col3:
        n_encoded = len([col for col in df.columns if '_' in col])
        st.metric("🔄 Variables encodées", n_encoded)
    
    # Vérification des types de données
    st.markdown("### 🧮 Types de Données")
    type_counts = df.dtypes.value_counts()
    
    for dtype, count in type_counts.items():
        st.write(f"- **{dtype}** : {count} variables")
    
    # Aperçu des données
    st.markdown("### 👀 Aperçu des Données Encodées")
    st.dataframe(df.head(), use_container_width=True)
    
    # Variables par catégorie
    st.markdown("### 🏷️ Variables par Catégorie")
    
    original_vars = [col for col in df.columns if '_' not in col]
    encoded_vars = [col for col in df.columns if '_' in col]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Variables Originales :**")
        for var in original_vars[:10]:  # Limiter l'affichage
            st.write(f"- {var}")
        if len(original_vars) > 10:
            st.write(f"... et {len(original_vars) - 10} autres")
    
    with col2:
        st.markdown("**Variables Encodées :**")
        for var in encoded_vars[:10]:  # Limiter l'affichage
            st.write(f"- {var}")
        if len(encoded_vars) > 10:
            st.write(f"... et {len(encoded_vars) - 10} autres")

def main():
    st.set_page_config(page_title="🔧 Test des Données Encodées", layout="wide")
    
    st.title("🔧 Test des Données Encodées")
    st.markdown("Cette page teste le chargement et l'encodage des données comme dans votre notebook.")
    
    # Charger les données
    df = load_encoded_data()
    
    if df is not None:
        display_data_info(df)
        
        # Test du clustering
        st.markdown("### 🎯 Test du Clustering (4 clusters)")
        
        cluster_cols = ['University_GPA', 'Internships_Completed', 'Technical_Skills_Score', 
                       'Soft_Skills_Score', 'Networking_Score', 'Starting_Salary']
        
        available_cluster_cols = [col for col in cluster_cols if col in df.columns]
        
        if len(available_cluster_cols) >= 3:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            X_cluster = df[available_cluster_cols]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            kmeans = KMeans(n_clusters=4, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            
            st.success("✅ Clustering réussi avec 4 clusters")
            st.write("Distribution des clusters :")
            for i, count in cluster_counts.items():
                st.write(f"- **Cluster {i}** : {count} étudiants ({count/len(df)*100:.1f}%)")
        else:
            st.error(f"❌ Variables manquantes pour le clustering. Trouvées : {available_cluster_cols}")
        
        # Test du modèle de prédiction
        st.markdown("### 🤖 Test du Modèle de Prédiction")
        
        if 'Starting_Salary' in df.columns:
            exclude_cols = ['Starting_Salary']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if len(feature_cols) > 5:
                st.success(f"✅ {len(feature_cols)} variables disponibles pour la prédiction")
                
                # Afficher les variables les plus importantes (si encodées)
                important_vars = [col for col in feature_cols if any(keyword in col for keyword in 
                                ['Field_of_Study', 'Location', 'Gender', 'Technical_Skills', 'Internships'])]
                
                if important_vars:
                    st.write("Variables clés détectées :")
                    for var in important_vars[:10]:
                        st.write(f"- {var}")
            else:
                st.error(f"❌ Pas assez de variables pour la prédiction ({len(feature_cols)} trouvées)")
        else:
            st.error("❌ Variable cible 'Starting_Salary' manquante")

if __name__ == "__main__":
    main()
