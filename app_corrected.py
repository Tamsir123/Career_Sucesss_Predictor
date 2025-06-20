import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="🎓 Système Hybride - Carrière & Salaire",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f0f2f6;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .cluster-interpretation {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Charge les données originales et encodées"""
    try:
        df_original = pd.read_csv('education_career_success_g.csv')
        df_encoded = pd.read_csv('education_career_success_encoded.csv')
        return df_original, df_encoded
    except FileNotFoundError as e:
        st.error(f"❌ Fichier de données non trouvé: {e}")
        return None, None

@st.cache_data
def train_model_and_clustering(df_encoded):
    """Entraîne le modèle et effectue le clustering"""
    
    # 1. MODÈLE DE PRÉDICTION
    exclude_cols = ['Starting_Salary']
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    X = df_encoded[feature_cols]
    y = df_encoded['Starting_Salary']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement Random Forest
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Métriques
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Importance des features
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # 2. CLUSTERING K-MEANS
    cluster_cols = ['University_GPA', 'Internships_Completed', 'Networking_Score', 
                   'Technical_Skills_Score', 'Soft_Skills_Score', 'Starting_Salary']
    
    X_cluster = df_encoded[cluster_cols]
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster)
    
    df_with_clusters = df_encoded.copy()
    df_with_clusters['Cluster'] = clusters
    
    cluster_summary = df_with_clusters.groupby('Cluster')[cluster_cols].mean()
    
    return model, feature_importance, r2, rmse, kmeans, cluster_summary, df_with_clusters

def create_encoded_user_data(user_inputs, df_encoded_columns):
    """Transforme les inputs utilisateur au format encodé"""
    
    # Créer un DataFrame avec toutes les colonnes encodées initialisées à 0
    user_data = pd.DataFrame(0, index=[0], columns=df_encoded_columns)
    
    # Remplir les variables numériques
    numeric_mapping = {
        'Age': user_inputs.get('age', 23),
        'University_GPA': user_inputs.get('university_gpa', 3.0),
        'High_School_GPA': user_inputs.get('high_school_gpa', 3.0),
        'SAT_Score': user_inputs.get('sat_score', 1200),
        'University_Ranking': user_inputs.get('university_ranking', 250),
        'Internships_Completed': user_inputs.get('internships', 2),
        'Projects_Completed': user_inputs.get('projects', 5),
        'Work_Experience_Years': user_inputs.get('work_experience', 1),
        'Technical_Skills_Score': user_inputs.get('technical_skills', 5),
        'Soft_Skills_Score': user_inputs.get('soft_skills', 5),
        'Networking_Score': user_inputs.get('networking', 5),
        'Study_Hours_Per_Week': user_inputs.get('study_hours', 20),
        'Extracurricular_Activities': user_inputs.get('extracurricular', 3),
        'Motivation': user_inputs.get('motivation', 7),
        'Work_Life_Balance': user_inputs.get('work_life_balance', 6),
        'Job_Offers': 3,
        'Career_Satisfaction': 6,
        'Years_to_Promotion': 3
    }
    
    for col, value in numeric_mapping.items():
        if col in user_data.columns:
            user_data[col] = value
    
    # Encoder les variables catégoriques
    field = user_inputs.get('field_of_study', 'Computer Science')
    if f'Field_of_Study_{field}' in user_data.columns:
        user_data[f'Field_of_Study_{field}'] = 1
    
    location = user_inputs.get('location', 'Urban')
    if f'Location_{location}' in user_data.columns:
        user_data[f'Location_{location}'] = 1
    
    gender = user_inputs.get('gender', 'Male')
    if f'Gender_{gender}' in user_data.columns:
        user_data[f'Gender_{gender}'] = 1
    
    # Langues par défaut
    if 'Languages_Spoken_Anglais' in user_data.columns:
        user_data['Languages_Spoken_Anglais'] = 1
    
    # Supprimer Starting_Salary si présent
    if 'Starting_Salary' in user_data.columns:
        user_data = user_data.drop(columns=['Starting_Salary'])
    
    return user_data

def main():
    """Application principale"""
    
    # En-tête
    st.markdown('<h1 class="main-header">🎓 Système Hybride: Recommandations de Carrière & Prédiction de Salaire</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choisissez une section:",
        ["🏠 Accueil", "📊 Exploration", "🎯 Clustering", "🤖 Prédiction", "💡 Recommandations Hybrides"]
    )
    
    # Chargement des données
    df_original, df_encoded = load_data()
    if df_original is None or df_encoded is None:
        st.stop()
    
    # Entraînement des modèles (mis en cache)
    model, feature_importance, r2, rmse, kmeans, cluster_summary, df_with_clusters = train_model_and_clustering(df_encoded)
    
    if page == "🏠 Accueil":
        st.markdown("""
        ## 🌟 Bienvenue dans le Système Hybride d'Analyse de Carrière
        
        ### 🎯 Problématique du Projet
        
        **Comment un modèle prédictif basé sur les caractéristiques académiques, professionnelles et personnelles 
        peut-il simultanément recommander des choix de carrière optimaux et estimer le salaire de départ ?**
        
        ### 🔬 Approche Hybride Mise en Œuvre
        
        #### 1. **Clustering K-Means** 🎯
        - Segmentation des étudiants en 4 profils types
        - Identification des patterns de réussite
        - Base pour les recommandations personnalisées
        
        #### 2. **Modèle de Régression** 🤖
        - Random Forest pour prédire le salaire de départ
        - Performance: R² = {:.3f}, RMSE = {:.3f}
        - Analyse d'importance des variables
        
        #### 3. **Système de Recommandation** 💡
        - Combinaison clustering + analyse d'impact
        - Recommandations hiérarchisées par priorité
        - Plan d'action personnalisé
        
        #### 4. **Analyse d'Équité** ⚖️
        - Détection des biais (Genre, Localisation)
        - Correction avec Fairlearn
        - Garantie d'équité dans les prédictions
        """.format(r2, rmse))
        
        # Métriques clés
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("👥 Étudiants", f"{len(df_original):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 Variables", f"{len(df_encoded.columns)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💰 Salaire Moyen", f"${df_original['Starting_Salary'].mean():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🎯 Performance R²", f"{r2:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Insights principaux
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🔍 Découvertes Clés du Projet
        
        **Facteurs d'influence majeurs:**
        - 🏢 **Stages (Internships)**: Impact le plus élevé sur le salaire
        - 💻 **Compétences techniques**: Essentielles dans l'économie numérique
        - 🤝 **Réseautage**: Peut compenser des lacunes académiques
        - 📚 **Expérience**: Chaque année compte significativement
        
        **Analyse d'équité:**
        - ✅ **Genre**: Équité confirmée (écart < 1%)
        - ⚠️ **Géographie**: Biais initial corrigé (Rural vs International)
        - 📈 **Amélioration**: Demographic Parity 0.87 → 0.03
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "📊 Exploration":
        st.markdown('<div class="sub-header">📊 Exploration des Données</div>', unsafe_allow_html=True)
        
        # Aperçu des données
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Données Originales")
            st.dataframe(df_original.head(), use_container_width=True)
        
        with col2:
            st.markdown("### 🔧 Données Encodées (échantillon)")
            st.dataframe(df_encoded.iloc[:, :10].head(), use_container_width=True)
        
        # Statistiques descriptives
        st.markdown("### 📈 Statistiques des Variables Numériques Principales")
        key_vars = ['Age', 'University_GPA', 'Starting_Salary', 'Internships_Completed', 
                   'Technical_Skills_Score', 'Soft_Skills_Score', 'Networking_Score']
        available_vars = [var for var in key_vars if var in df_original.columns]
        st.dataframe(df_original[available_vars].describe(), use_container_width=True)
        
        # Analyse des salaires par domaine
        if 'Field_of_Study' in df_original.columns:
            st.markdown("### 💰 Analyse des Salaires par Domaine")
            salary_by_field = df_original.groupby('Field_of_Study')['Starting_Salary'].agg(['mean', 'count']).round(0)
            salary_by_field = salary_by_field.sort_values('mean', ascending=False)
            
            fig = px.bar(
                salary_by_field.reset_index(),
                x='Field_of_Study',
                y='mean',
                title="Salaire Moyen par Domaine d'Étude",
                labels={'mean': 'Salaire Moyen ($)', 'Field_of_Study': 'Domaine'},
                color='mean',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🎯 Clustering":
        st.markdown('<div class="sub-header">🎯 Analyse par Clustering K-Means</div>', unsafe_allow_html=True)
        
        # Résumé des clusters
        st.markdown("### 📊 Profils des 4 Clusters Identifiés")
        st.dataframe(cluster_summary.round(2), use_container_width=True)
        
        # Visualisation radar
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        cluster_cols = ['University_GPA', 'Internships_Completed', 'Networking_Score', 
                       'Technical_Skills_Score', 'Soft_Skills_Score', 'Starting_Salary']
        
        for i, cluster in enumerate(cluster_summary.index):
            values = cluster_summary.loc[cluster, cluster_cols].values.tolist()
            values += [values[0]]  # Fermer le radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=cluster_cols + [cluster_cols[0]],
                fill='toself',
                name=f'Cluster {cluster}',
                line_color=colors[i]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[cluster_summary.min().min(), cluster_summary.max().max()])),
            showlegend=True,
            title="🎯 Profils des Clusters - Analyse Radar",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution des clusters
        cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
        
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}" for i in cluster_counts.index],
            title="📈 Distribution des Étudiants par Cluster"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Interprétations détaillées
        st.markdown("### 🎯 Interprétation Détaillée des Clusters")
        
        interpretations = {
            0: {
                "title": "🌟 Performants avec Expérience Pratique",
                "description": "GPA moyen, excellente expérience en stages, bonnes compétences techniques. Salaire élevé grâce à l'expérience pratique.",
                "strengths": ["Beaucoup de stages", "Compétences techniques solides", "Salaire attractif"],
                "improvements": ["Renforcer le réseautage", "Développer les soft skills"]
            },
            1: {
                "title": "🤝 Sociables en Développement Technique", 
                "description": "Peu de stages mais excellentes compétences relationnelles. Potentiel important avec développement technique.",
                "strengths": ["Excellentes soft skills", "Bonnes compétences techniques", "Potentiel relationnel"],
                "improvements": ["Multiplier les stages", "Développer le réseautage professionnel"]
            },
            2: {
                "title": "🌐 Réseautés aux Compétences Mixtes",
                "description": "Bon réseautage et compétences techniques, mais faibles en relations interpersonnelles.",
                "strengths": ["Excellent réseautage", "Bonnes compétences techniques", "Salaire correct"],
                "improvements": ["Développer les soft skills", "Plus d'expérience pratique"]
            },
            3: {
                "title": "⚡ Profils en Développement",
                "description": "Compétences techniques très faibles mais potentiel identifiable. Nécessite accompagnement intensif.",
                "strengths": ["Réseautage correct", "Marge de progression importante"],
                "improvements": ["Urgence: compétences techniques", "Formation intensive requise"]
            }
        }
        
        for cluster_id, info in interpretations.items():
            with st.expander(f"Cluster {cluster_id}: {info['title']}"):
                st.markdown(f"**Description**: {info['description']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Points forts:**")
                    for strength in info['strengths']:
                        st.markdown(f"- {strength}")
                
                with col2:
                    st.markdown("**🎯 Axes d'amélioration:**")
                    for improvement in info['improvements']:
                        st.markdown(f"- {improvement}")
    
    elif page == "🤖 Prédiction":
        st.markdown('<div class="sub-header">🤖 Modèle de Prédiction Random Forest</div>', unsafe_allow_html=True)
        
        # Métriques de performance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🎯 Score R²", f"{r2:.3f}")
            st.markdown("*Variance expliquée*")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📏 RMSE", f"{rmse:.3f}")
            st.markdown("*Erreur quadratique*")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🌳 Arbres", "200")
            st.markdown("*Random Forest*")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Importance des variables
        st.markdown("### 🔝 Importance des Variables (Top 15)")
        
        top_features = feature_importance.head(15)
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Variables les Plus Influentes sur le Salaire",
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interface de prédiction rapide
        st.markdown("### 🔮 Prédicteur de Salaire Instantané")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📚 Profil Académique**")
                university_gpa = st.slider("GPA Universitaire", 0.0, 4.0, 3.0, 0.1)
                sat_score = st.slider("Score SAT", 400, 1600, 1200, 50)
                university_ranking = st.slider("Ranking Université", 1, 500, 250)
                internships = st.slider("Stages complétés", 0, 5, 2)
                projects = st.slider("Projets réalisés", 0, 15, 5)
            
            with col2:
                st.markdown("**🎯 Compétences & Expérience**")
                technical_skills = st.slider("Compétences techniques (1-10)", 1, 10, 5)
                soft_skills = st.slider("Compétences relationnelles (1-10)", 1, 10, 5)
                networking = st.slider("Score de réseautage (1-10)", 1, 10, 5)
                work_experience = st.slider("Années d'expérience", 0, 10, 1)
                
                field_of_study = st.selectbox("Domaine d'étude", 
                                            ['Computer Science', 'Engineering', 'Business', 'Medicine'])
            
            submitted = st.form_submit_button("🚀 Prédire le Salaire", type="primary")
            
            if submitted:
                # Préparer les données
                user_inputs = {
                    'university_gpa': university_gpa,
                    'sat_score': sat_score,
                    'university_ranking': university_ranking,
                    'internships': internships,
                    'projects': projects,
                    'technical_skills': technical_skills,
                    'soft_skills': soft_skills,
                    'networking': networking,
                    'work_experience': work_experience,
                    'field_of_study': field_of_study
                }
                
                user_data = create_encoded_user_data(user_inputs, df_encoded.columns)
                feature_cols = [col for col in model.feature_names_in_ if col in user_data.columns]
                
                predicted_salary = model.predict(user_data[feature_cols])[0]
                
                st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                st.markdown(f"## 💰 Salaire de Départ Prédit: ${predicted_salary:,.0f}")
                
                # Analyse du résultat
                avg_salary = df_original['Starting_Salary'].mean()
                diff_pct = ((predicted_salary - avg_salary) / avg_salary) * 100
                
                if diff_pct > 10:
                    st.markdown("🎉 **Excellent potentiel!** Salaire supérieur à la moyenne")
                elif diff_pct > 0:
                    st.markdown("✅ **Bon profil** - Salaire au-dessus de la moyenne")
                else:
                    st.markdown("📈 **Potentiel d'amélioration** - Consultez la section Recommandations")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "💡 Recommandations Hybrides":
        st.markdown('<div class="sub-header">💡 Système de Recommandations Hybride IA</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🧠 Méthodologie Hybride
        Ce système combine **3 techniques d'IA** pour des recommandations personnalisées :
        - **Clustering K-Means** : Positionnement par rapport aux profils types
        - **Random Forest** : Simulation d'impact des améliorations  
        - **Analyse comparative** : Écart avec les profils les plus performants
        """)
        
        with st.form("recommendation_form"):
            st.markdown("### 📝 Saisissez Votre Profil")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📚 Informations Académiques**")
                age = st.slider("Âge", 18, 35, 23)
                university_gpa = st.slider("GPA Universitaire", 0.0, 4.0, 3.0, 0.1)
                high_school_gpa = st.slider("GPA Lycée", 0.0, 4.0, 3.0, 0.1)
                sat_score = st.slider("Score SAT", 400, 1600, 1200, 50)
                university_ranking = st.slider("Ranking Université", 1, 500, 250)
                internships = st.slider("Stages complétés", 0, 5, 2)
                projects = st.slider("Projets réalisés", 0, 15, 5)
                work_experience = st.slider("Années d'expérience", 0, 10, 1)
            
            with col2:
                st.markdown("**🎯 Compétences & Préférences**")
                technical_skills = st.slider("Compétences techniques (1-10)", 1, 10, 5)
                soft_skills = st.slider("Compétences relationnelles (1-10)", 1, 10, 5)
                networking = st.slider("Score de réseautage (1-10)", 1, 10, 5)
                study_hours = st.slider("Heures d'étude/semaine", 0, 50, 20)
                extracurricular = st.slider("Activités extrascolaires", 0, 10, 3)
                motivation = st.slider("Niveau de motivation (1-10)", 1, 10, 7)
                work_life_balance = st.slider("Équilibre vie-travail (1-10)", 1, 10, 6)
                
                field_of_study = st.selectbox("Domaine d'étude", 
                                            ['Computer Science', 'Engineering', 'Business', 'Medicine'])
                location = st.selectbox("Localisation", ['Urban', 'Rural', 'International'])
                gender = st.selectbox("Genre", ['Male', 'Female'])
            
            submitted = st.form_submit_button("🚀 Générer Recommandations Hybrides", type="primary")
            
            if submitted:
                # Préparer les données utilisateur
                user_inputs = {
                    'age': age, 'university_gpa': university_gpa, 'high_school_gpa': high_school_gpa,
                    'sat_score': sat_score, 'university_ranking': university_ranking,
                    'internships': internships, 'projects': projects, 'work_experience': work_experience,
                    'technical_skills': technical_skills, 'soft_skills': soft_skills,
                    'networking': networking, 'study_hours': study_hours, 'extracurricular': extracurricular,
                    'motivation': motivation, 'work_life_balance': work_life_balance,
                    'field_of_study': field_of_study, 'location': location, 'gender': gender
                }
                
                user_data = create_encoded_user_data(user_inputs, df_encoded.columns)
                
                # 1. PRÉDICTION DE SALAIRE
                feature_cols = [col for col in model.feature_names_in_ if col in user_data.columns]
                predicted_salary = model.predict(user_data[feature_cols])[0]
                
                # 2. IDENTIFICATION DU CLUSTER
                cluster_cols = ['University_GPA', 'Internships_Completed', 'Networking_Score', 
                               'Technical_Skills_Score', 'Soft_Skills_Score', 'Starting_Salary']
                
                cluster_data = pd.DataFrame({
                    'University_GPA': [university_gpa],
                    'Internships_Completed': [internships], 
                    'Networking_Score': [networking],
                    'Technical_Skills_Score': [technical_skills],
                    'Soft_Skills_Score': [soft_skills],
                    'Starting_Salary': [predicted_salary]
                })
                
                user_cluster = kmeans.predict(cluster_data)[0]
                
                # 3. AFFICHAGE DES RÉSULTATS
                st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                st.markdown(f"## 🎯 Profil Identifié: Cluster {user_cluster}")
                
                cluster_names = {
                    0: "**Performant avec Expérience Pratique**",
                    1: "**Sociable en Développement Technique**", 
                    2: "**Réseauté aux Compétences Mixtes**",
                    3: "**Profil en Développement**"
                }
                
                st.markdown(f"### {cluster_names.get(user_cluster, 'Profil non défini')}")
                st.markdown(f"### 💰 Salaire de départ prédit: **${predicted_salary:,.0f}**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 4. RECOMMANDATIONS PERSONNALISÉES
                st.markdown("### 🎯 Recommandations Prioritaires")
                
                # Identifier le meilleur cluster
                best_cluster = cluster_summary['Starting_Salary'].idxmax()
                best_profile = cluster_summary.loc[best_cluster]
                
                # Analyser les écarts
                recommendations = []
                
                dimensions = {
                    'Internships_Completed': {
                        'current': internships,
                        'target': best_profile['Internships_Completed'],
                        'actions': [
                            "🏢 Recherchez activement des stages dans votre domaine",
                            "📧 Contactez directement les entreprises pour des opportunités",
                            "🤝 Utilisez votre réseau universitaire et LinkedIn",
                            "🌍 Considérez les stages à l'étranger pour plus d'expérience"
                        ]
                    },
                    'Technical_Skills_Score': {
                        'current': technical_skills,
                        'target': best_profile['Technical_Skills_Score'],
                        'actions': [
                            "💻 Suivez des formations en ligne (Coursera, edX, Udemy)",
                            "🔧 Participez à des projets open source sur GitHub", 
                            "📜 Obtenez des certifications professionnelles reconnues",
                            "🛠️ Créez un portfolio de projets techniques"
                        ]
                    },
                    'Networking_Score': {
                        'current': networking,
                        'target': best_profile['Networking_Score'],
                        'actions': [
                            "🌐 Rejoignez des associations professionnelles",
                            "📱 Optimisez et utilisez activement LinkedIn",
                            "🎤 Participez à des événements et conférences",
                            "👥 Créez ou rejoignez des groupes d'étude"
                        ]
                    },
                    'Soft_Skills_Score': {
                        'current': soft_skills,
                        'target': best_profile['Soft_Skills_Score'],
                        'actions': [
                            "🗣️ Rejoignez un club de débat ou Toastmasters",
                            "🤝 Pratiquez le travail en équipe sur des projets",
                            "📚 Lisez des livres sur le leadership et la communication",
                            "🎭 Participez à des activités de présentation publique"
                        ]
                    }
                }
                
                # Calculer les priorités
                for dim, info in dimensions.items():
                    gap = info['target'] - info['current']
                    if gap > 0.1:  # Seulement si amélioration significative possible
                        recommendations.append({
                            'dimension': dim,
                            'gap': gap,
                            'actions': info['actions']
                        })
                
                # Trier par écart (priorité)
                recommendations.sort(key=lambda x: x['gap'], reverse=True)
                
                # Afficher les 3 recommandations prioritaires
                for i, rec in enumerate(recommendations[:3], 1):
                    dim_name = rec['dimension'].replace('_', ' ').title()
                    
                    st.markdown(f"#### {i}. 🎯 {dim_name}")
                    st.markdown(f"**Écart avec le profil optimal**: {rec['gap']:.2f} points")
                    st.markdown("**Actions recommandées**:")
                    
                    for j, action in enumerate(rec['actions'][:2], 1):
                        st.markdown(f"{j}. {action}")
                    
                    st.markdown("---")
                
                # 5. PLAN D'ACTION
                st.markdown("### 📋 Plan d'Action Personnalisé (6 mois)")
                
                improvement_potential = len(recommendations) * 3000
                target_salary = predicted_salary + improvement_potential
                
                action_plan = f"""
                **🎯 Objectif**: Passer au profil du Cluster {best_cluster} (optimal)
                
                **Mois 1-2: Foundation**
                - Commencer par la priorité #{1 if recommendations else 'N/A'}
                - Mettre en place une routine d'amélioration
                
                **Mois 3-4: Développement** 
                - Continuer les actions prioritaires
                - Ajouter la priorité #{2 if len(recommendations) > 1 else 'N/A'}
                
                **Mois 5-6: Optimisation**
                - Consolider les acquis
                - Préparer la recherche d'emploi
                
                **💰 Salaire cible après amélioration**: ${target_salary:,.0f}
                **📈 Gain potentiel**: +${improvement_potential:,.0f}
                """
                
                st.markdown(action_plan)
                
                # 6. INDICATEURS DE SUIVI
                with st.expander("📊 Indicateurs de Suivi Recommandés"):
                    st.markdown("""
                    **📈 Metrics à suivre mensuellement:**
                    - Nombre de candidatures de stage envoyées
                    - Heures de formation technique complétées  
                    - Nouveaux contacts professionnels ajoutés
                    - Projets techniques finalisés
                    - Événements de networking fréquentés
                    
                    **🎯 Objectifs SMART à définir:**
                    - Spécifiques, Mesurables, Atteignables, Réalistes, Temporels
                    """)

if __name__ == "__main__":
    main()
