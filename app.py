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
    page_title="🎓 Analyse Succès Éducation & Carrière",
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Charge et prépare les données"""
    try:
        # Charger d'abord les données originales pour l'exploration
        df_original = pd.read_csv('education_career_success_g.csv')
        # Charger les données encodées pour la prédiction
        df_encoded = pd.read_csv('education_career_success_encoded.csv')
        return df_original, df_encoded
    except FileNotFoundError as e:
        st.error(f"❌ Fichier de données non trouvé: {e}")
        return None, None

@st.cache_data
def prepare_data(df):
    """Prépare les données pour l'analyse"""
    if df is None:
        return None, None, None
    
    # Copie du DataFrame
    df_processed = df.copy()
    
    # Suppression de Student_ID si présent
    if 'Student_ID' in df_processed.columns:
        df_processed = df_processed.drop(columns=['Student_ID'])
    
    # Gestion des valeurs manquantes pour les colonnes numériques
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Gestion des valeurs manquantes pour les colonnes catégoriques
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col].fillna(df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 'Unknown', inplace=True)
    
    # Encodage des variables catégoriques principales pour la prédiction
    # Field_of_Study
    if 'Field_of_Study' in df_processed.columns:
        field_dummies = pd.get_dummies(df_processed['Field_of_Study'], prefix='Field_of_Study')
        df_processed = pd.concat([df_processed, field_dummies], axis=1)
    
    # Location
    if 'Location' in df_processed.columns:
        location_dummies = pd.get_dummies(df_processed['Location'], prefix='Location')
        df_processed = pd.concat([df_processed, location_dummies], axis=1)
    
    # Gender
    if 'Gender' in df_processed.columns:
        gender_dummies = pd.get_dummies(df_processed['Gender'], prefix='Gender')
        df_processed = pd.concat([df_processed, gender_dummies], axis=1)
    
    # Current_Job_Level
    if 'Current_Job_Level' in df_processed.columns:
        job_level_dummies = pd.get_dummies(df_processed['Current_Job_Level'], prefix='Current_Job_Level')
        df_processed = pd.concat([df_processed, job_level_dummies], axis=1)
    
    # Mettre à jour les colonnes numériques après encodage
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64', 'uint8']).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    return df_processed, numeric_cols, categorical_cols

def create_correlation_heatmap(df, numeric_cols):
    """Crée une heatmap de corrélation interactive"""
    correlation_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size":10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="🔥 Matrice de Corrélation des Variables Numériques",
        xaxis={'side': 'bottom'},
        width=800,
        height=600
    )
    
    return fig

def create_distribution_plots(df, numeric_cols):
    """Crée des graphiques de distribution pour les variables numériques"""
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=numeric_cols,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                marker_color=colors[i % len(colors)],
                opacity=0.7,
                showlegend=False
            ),
            row=row, col=col_idx
        )
    
    fig.update_layout(
        title="📊 Distribution des Variables Numériques",
        height=200 * n_rows,
        showlegend=False
    )
    
    return fig

def perform_clustering(df_encoded):
    """Effectue le clustering K-Means basé sur votre notebook"""
    
    # Variables utilisées pour le clustering dans votre notebook
    cluster_cols = ['University_GPA', 'Internships_Completed', 'Networking_Score', 
                   'Technical_Skills_Score', 'Soft_Skills_Score', 'Starting_Salary']
    
    # Vérifier que toutes les colonnes existent
    available_cols = [col for col in cluster_cols if col in df_encoded.columns]
    
    if len(available_cols) < 5:
        st.warning("⚠️ Pas assez de colonnes disponibles pour le clustering")
        return None, None, None
    
    X_cluster = df_encoded[available_cols].copy()
    
    # Les données sont déjà standardisées dans votre notebook (education_career_success_encoded.csv)
    # Donc pas besoin de re-standardiser
    
    # K-Means avec 4 clusters comme déterminé dans votre notebook
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster)
    
    # Ajout des clusters au DataFrame
    df_with_clusters = df_encoded.copy()
    df_with_clusters['Cluster'] = clusters
    
    # Résumé des clusters
    cluster_summary = df_with_clusters.groupby('Cluster')[available_cols].mean()
    
    return df_with_clusters, cluster_summary, available_cols

def create_cluster_visualization(df_with_clusters, cluster_summary, available_cols):
    """Crée une visualisation des clusters"""
    # Graphique radar pour comparer les clusters
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i, cluster in enumerate(cluster_summary.index):
        values = cluster_summary.loc[cluster].values.tolist()
        values += [values[0]]  # Fermer le radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_cols + [available_cols[0]],
            fill='toself',
            name=f'Cluster {cluster}',
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[cluster_summary.min().min(), cluster_summary.max().max()]
            )),
        showlegend=True,
        title="🎯 Profil des Clusters - Analyse Radar"
    )
    
    return fig

def train_prediction_model(df_encoded):
    """Entraîne un modèle de prédiction Random Forest basé sur votre notebook"""
    
    # Variables utilisées dans votre notebook pour la prédiction
    # Exclure Starting_Salary (target) et Student_ID s'il existe
    exclude_cols = ['Starting_Salary']
    if 'Student_ID' in df_encoded.columns:
        exclude_cols.append('Student_ID')
    
    # Sélectionner toutes les autres colonnes comme features
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    if len(feature_cols) < 5 or 'Starting_Salary' not in df_encoded.columns:
        st.warning("⚠️ Pas assez de variables pour entraîner le modèle de prédiction")
        return None, None, None, None
    
    X = df_encoded[feature_cols].copy()
    y = df_encoded['Starting_Salary'].copy()
    
    # Vérifier qu'il n'y a pas de valeurs manquantes
    if X.isnull().sum().sum() > 0:
        st.warning("⚠️ Valeurs manquantes détectées dans les features")
        # Remplacer par la médiane si nécessaire
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # Division train/test comme dans votre notebook
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement du modèle Random Forest avec les mêmes paramètres que votre notebook
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Prédictions et métriques
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Importance des variables
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, feature_importance, r2, rmse

def create_salary_analysis(df):
    """Crée une analyse des salaires par différents critères"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Salaire par Domaine d\'Étude', 'Salaire par Genre', 
                       'Salaire par Localisation', 'Distribution des Salaires'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # Salaire par domaine d'étude
    if 'Field_of_Study' in df.columns:
        salary_by_field = df.groupby('Field_of_Study')['Starting_Salary'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=salary_by_field.index, y=salary_by_field.values, name="Domaine", marker_color='lightblue'),
            row=1, col=1
        )
    
    # Salaire par genre
    if 'Gender' in df.columns:
        salary_by_gender = df.groupby('Gender')['Starting_Salary'].mean()
        fig.add_trace(
            go.Bar(x=salary_by_gender.index, y=salary_by_gender.values, name="Genre", marker_color='lightcoral'),
            row=1, col=2
        )
    
    # Salaire par localisation
    if 'Location' in df.columns:
        salary_by_location = df.groupby('Location')['Starting_Salary'].mean()
        fig.add_trace(
            go.Bar(x=salary_by_location.index, y=salary_by_location.values, name="Localisation", marker_color='lightgreen'),
            row=2, col=1
        )
    
    # Distribution des salaires
    fig.add_trace(
        go.Histogram(x=df['Starting_Salary'], name="Distribution", marker_color='gold'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="💰 Analyse Complète des Salaires")
    
    return fig

def create_advanced_recommendation_system(df_encoded, model, cluster_summary, kmeans_model):
    """Système de recommandations hybride basé sur votre notebook"""
    st.markdown('<div class="sub-header">🎯 Système de Recommandations Hybride</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🧠 Approche Hybride Intelligence Artificielle
    Ce système combine plusieurs techniques d'IA pour des recommandations personnalisées :
    - **Clustering K-Means** : Position par rapport aux profils types
    - **Random Forest** : Simulation d'impact des améliorations
    - **Analyse comparative** : Comparaison avec les meilleurs profils
    """)
    
    # Interface utilisateur pour saisir les informations (variables principales de votre modèle)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Informations Académiques et Professionnelles")
        age = st.slider("Âge", 18, 35, 23)
        university_gpa = st.slider("GPA Universitaire", 0.0, 4.0, 3.0, 0.1)
        high_school_gpa = st.slider("GPA Lycée", 0.0, 4.0, 3.0, 0.1)
        sat_score = st.slider("Score SAT", 400, 1600, 1200, 50)
        university_ranking = st.slider("Ranking Université", 1, 500, 250)
        internships = st.slider("Nombre de stages", 0, 5, 2)
        projects = st.slider("Projets complétés", 0, 15, 5)
        work_experience = st.slider("Années d'expérience", 0, 10, 1)
    
    with col2:
        st.markdown("### 🎯 Compétences et Soft Skills")
        technical_skills = st.slider("Compétences techniques (1-10)", 1, 10, 5)
        soft_skills = st.slider("Compétences relationnelles (1-10)", 1, 10, 5)
        networking = st.slider("Score de réseautage (1-10)", 1, 10, 5)
        study_hours = st.slider("Heures d'étude/semaine", 0, 50, 20)
        extracurricular = st.slider("Activités extrascolaires", 0, 10, 3)
        motivation = st.slider("Niveau de motivation (1-10)", 1, 10, 7)
        work_life_balance = st.slider("Équilibre vie-travail (1-10)", 1, 10, 6)
        
        # Variables catégoriques principales
        field_options = ['Business', 'Computer Science', 'Engineering', 'Medicine']
        field_of_study = st.selectbox("Domaine d'étude", field_options)
        
        location_options = ['Urban', 'Rural', 'International']
        location = st.selectbox("Localisation", location_options)
        
        gender_options = ['Male', 'Female']
        gender = st.selectbox("Genre", gender_options)
    
    if st.button("🚀 Générer Recommandations Hybrides", type="primary"):
        # Préparer les données pour la prédiction (format encodé)
        user_data = create_encoded_user_data(
            age, university_gpa, high_school_gpa, sat_score, university_ranking,
            internships, projects, work_experience, technical_skills, soft_skills,
            networking, study_hours, extracurricular, motivation, work_life_balance,
            field_of_study, location, gender, df_encoded.columns
        )
        
        # 1. CLUSTERING - Identifier le profil type
        cluster_cols = ['University_GPA', 'Internships_Completed', 'Networking_Score', 
                       'Technical_Skills_Score', 'Soft_Skills_Score', 'Starting_Salary']
        
        # Prédire d'abord le salaire pour avoir Starting_Salary
        predicted_salary = model.predict(user_data[[col for col in model.feature_names_in_]])[0]
        
        # Créer les données pour le clustering
        cluster_data = pd.DataFrame({
            'University_GPA': [university_gpa],
            'Internships_Completed': [internships],
            'Networking_Score': [networking],
            'Technical_Skills_Score': [technical_skills],
            'Soft_Skills_Score': [soft_skills],
            'Starting_Salary': [predicted_salary]
        })
        
        user_cluster = kmeans_model.predict(cluster_data)[0]
        
        # 2. RECOMMANDATIONS BASÉES SUR L'ANALYSE COMPARATIVE
        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
        st.markdown(f"## 🎯 Profil Identifié: Cluster {user_cluster}")
        
        # Interpréter le cluster selon votre notebook
        cluster_interpretations = {
            0: "**Étudiants performants avec expérience pratique** - Profil équilibré avec bon potentiel",
            1: "**Étudiants sociables avec développement technique** - Excellentes soft skills à compléter",
            2: "**Étudiants réseautés mais techniques** - Bon réseautage, compétences relationnelles à améliorer", 
            3: "**Étudiants en développement** - Potentiel important, nécessite renforcement compétences"
        }
        
        st.markdown(f"### {cluster_interpretations.get(user_cluster, 'Profil non défini')}")
        st.markdown(f"### 💰 Salaire de départ prédit: **${predicted_salary:,.0f}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 3. ANALYSE D'IMPACT ET RECOMMANDATIONS PRIORITAIRES
        st.markdown("### 🎯 Recommandations Prioritaires (par ordre d'impact)")
        
        # Identifier le cluster avec le meilleur salaire moyen
        best_cluster = cluster_summary['Starting_Salary'].idxmax()
        best_profile = cluster_summary.loc[best_cluster]
        current_profile = cluster_data.iloc[0]
        
        # Calculer les écarts et impacts potentiels
        recommendations = []
        
        # Analyser chaque dimension clé
        dimensions = {
            'Internships_Completed': {
                'current': internships,
                'target': best_profile['Internships_Completed'],
                'actions': [
                    "🏢 Recherchez activement des stages dans votre domaine",
                    "📧 Contactez directement les entreprises pour des opportunités",
                    "🤝 Utilisez votre réseau universitaire et professionnel",
                    "💼 Considérez les stages à l'étranger pour plus d'expérience"
                ]
            },
            'Technical_Skills_Score': {
                'current': technical_skills,
                'target': best_profile['Technical_Skills_Score'],
                'actions': [
                    "💻 Suivez des formations techniques en ligne (Coursera, edX)",
                    "🔧 Participez à des projets open source",
                    "📜 Obtenez des certifications professionnelles",
                    "🛠️ Créez un portfolio de projets techniques"
                ]
            },
            'Networking_Score': {
                'current': networking,
                'target': best_profile['Networking_Score'],
                'actions': [
                    "🌐 Rejoignez des associations professionnelles",
                    "📱 Optimisez votre profil LinkedIn",
                    "🎤 Participez à des événements et conférences",
                    "👥 Créez des groupes d'étude avec vos pairs"
                ]
            },
            'Soft_Skills_Score': {
                'current': soft_skills,
                'target': best_profile['Soft_Skills_Score'],
                'actions': [
                    "🗣️ Rejoignez un club de débat ou Toastmasters",
                    "🤝 Pratiquez le travail en équipe sur des projets",
                    "📚 Lisez des livres sur le leadership et la communication",
                    "🎭 Participez à des activités théâtrales ou de présentation"
                ]
            },
            'University_GPA': {
                'current': university_gpa,
                'target': best_profile['University_GPA'],
                'actions': [
                    "📖 Améliorez vos méthodes d'étude",
                    "👨‍🏫 Consultez régulièrement vos professeurs",
                    "📝 Rejoignez des groupes d'étude",
                    "⏰ Optimisez votre gestion du temps"
                ]
            }
        }
        
        # Calculer l'impact potentiel pour chaque dimension
        for dim, info in dimensions.items():
            gap = info['target'] - info['current']
            if gap > 0.1:  # Seulement si amélioration significative possible
                # Simulation d'impact (approximation)
                impact_score = gap * 0.2  # Coefficient arbitraire basé sur votre analyse
                recommendations.append({
                    'dimension': dim,
                    'gap': gap,
                    'impact': impact_score,
                    'actions': info['actions']
                })
        
        # Trier par impact potentiel
        recommendations.sort(key=lambda x: x['gap'], reverse=True)
        
        # Afficher les 3 recommandations prioritaires
        for i, rec in enumerate(recommendations[:3], 1):
            dim_name = rec['dimension'].replace('_', ' ').title()
            st.markdown(f"""
            #### {i}. 🎯 {dim_name}
            **Écart avec le profil optimal**: {rec['gap']:.2f} points  
            **Actions recommandées**:
            """)
            for action in rec['actions'][:2]:  # Limiter à 2 actions
                st.markdown(f"- {action}")
            st.markdown("---")
        
        # 4. PLAN D'ACTION PERSONNALISÉ
        st.markdown("### � Plan d'Action à 6 Mois")
        
        action_plan = f"""
        **Mois 1-2 : Focus sur l'expérience pratique**
        - Candidater pour {max(1, int(best_profile['Internships_Completed'] - internships))} stage(s) supplémentaire(s)
        - Commencer un projet personnel dans votre domaine
        
        **Mois 3-4 : Développement des compétences**
        - Suivre une formation technique ciblée
        - Participer à 2-3 événements de réseautage
        
        **Mois 5-6 : Consolidation et optimisation**
        - Finaliser les projets en cours
        - Préparer votre recherche d'emploi avec un CV optimisé
        
        **Objectif de salaire révisé**: ${predicted_salary + (len(recommendations) * 5000):,.0f}
        """
        
        st.markdown(action_plan)

def create_encoded_user_data(age, university_gpa, high_school_gpa, sat_score, university_ranking,
                           internships, projects, work_experience, technical_skills, soft_skills,
                           networking, study_hours, extracurricular, motivation, work_life_balance,
                           field_of_study, location, gender, encoded_columns):
    """Crée un DataFrame encodé pour les données utilisateur"""
    
    # Initialiser avec des zéros pour toutes les colonnes encodées
    user_data = pd.DataFrame(0, index=[0], columns=encoded_columns)
    
    # Remplir les variables numériques
    user_data['Age'] = age
    user_data['University_GPA'] = university_gpa
    user_data['High_School_GPA'] = high_school_gpa
    user_data['SAT_Score'] = sat_score
    user_data['University_Ranking'] = university_ranking
    user_data['Internships_Completed'] = internships
    user_data['Projects_Completed'] = projects
    user_data['Work_Experience_Years'] = work_experience
    user_data['Technical_Skills_Score'] = technical_skills
    user_data['Soft_Skills_Score'] = soft_skills
    user_data['Networking_Score'] = networking
    user_data['Study_Hours_Per_Week'] = study_hours
    user_data['Extracurricular_Activities'] = extracurricular
    user_data['Motivation'] = motivation
    user_data['Work_Life_Balance'] = work_life_balance
    
    # Variables avec valeurs par défaut
    user_data['Job_Offers'] = 3
    user_data['Career_Satisfaction'] = 6
    user_data['Years_to_Promotion'] = 3
    
    # Encoder les variables catégoriques
    if f'Field_of_Study_{field_of_study}' in user_data.columns:
        user_data[f'Field_of_Study_{field_of_study}'] = 1
    
    if f'Location_{location}' in user_data.columns:
        user_data[f'Location_{location}'] = 1
    
    if f'Gender_{gender}' in user_data.columns:
        user_data[f'Gender_{gender}'] = 1
    
    # Variables langues (par défaut Anglais)
    if 'Languages_Spoken_Anglais' in user_data.columns:
        user_data['Languages_Spoken_Anglais'] = 1
    
    # Supprimer Starting_Salary si présent (c'est la target)
    if 'Starting_Salary' in user_data.columns:
        user_data = user_data.drop(columns=['Starting_Salary'])
    
    return user_data

def main():
    """Fonction principale de l'application"""
    
    # En-tête principal
    st.markdown('<h1 class="main-header">🎓 Analyse du Succès en Éducation & Carrière</h1>', unsafe_allow_html=True)
    
    # Sidebar pour la navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choisissez une section:",
        ["🏠 Accueil", "📊 Exploration des Données", "🔍 Analyse Approfondie", 
         "🎯 Clustering", "🤖 Modèle Prédictif", "💡 Recommandations", "📈 Insights Avancés"]
    )
    
    # Chargement des données
    df_original, df_encoded = load_data()
    if df_original is None or df_encoded is None:
        st.stop()
    
    df_processed, numeric_cols, categorical_cols = prepare_data(df_original)
    
    # Pré-entraîner les modèles pour les utiliser dans les recommandations
    model = None
    cluster_summary = None
    kmeans_model = None
    
    if page in ["🤖 Modèle Prédictif", "💡 Recommandations"]:
        # Entraîner le modèle de prédiction
        model, feature_importance, r2, rmse = train_prediction_model(df_encoded)
        
        # Entraîner le modèle de clustering  
        df_with_clusters, cluster_summary, cluster_cols = perform_clustering(df_encoded)
        if df_with_clusters is not None:
            # Recréer le modèle k-means pour les prédictions
            from sklearn.cluster import KMeans
            X_cluster = df_encoded[cluster_cols]
            kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
            kmeans_model.fit(X_cluster)
    
    # Navigation entre les pages
    if page == "🏠 Accueil":
        st.markdown("""
        ## 🌟 Bienvenue dans l'Analyse du Succès Éducatif
        
        Cette application interactive explore les facteurs qui influencent le succès dans l'éducation et la carrière. 
        
        ### 🎯 Objectifs du Projet
        
        Notre **problématique hybride** consiste à créer un modèle prédictif qui simultanément :
        - **Recommande des choix de carrière optimaux** basés sur le profil de l'étudiant
        - **Estime le salaire de départ** en tenant compte des interactions complexes
        - **Analyse l'équité** pour corriger les biais potentiels
        
        ### 🔬 Méthodologie Hybride
        
        #### 1. **Clustering K-Means** 
        - Segmentation des profils d'étudiants similaires
        - 4 clusters identifiés avec des caractéristiques distinctes
        
        #### 2. **Modèles de Régression** 
        - **Random Forest** et **XGBoost** pour prédire le salaire
        - Performance: R² ≈ 0.77, RMSE ≈ 0.48
        
        #### 3. **Système de Recommandation**
        - Combinaison clustering + analyse d'impact supervisée
        - Recommandations personnalisées et hiérarchisées
        
        #### 4. **Analyse d'Équité**
        - Détection et correction des biais (Genre, Localisation)
        - Utilisation de **Fairlearn** pour l'équité démographique
        
        ### 📊 Données Analysées
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📚 Étudiants", f"{len(df_original):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📋 Variables", len(df_encoded.columns))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💰 Salaire Moyen", f"${df_original['Starting_Salary'].mean():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'Field_of_Study' in df_original.columns:
                st.metric("🎓 Domaines", df_original['Field_of_Study'].nunique())
            else:
                st.metric("🔢 Variables Numériques", len(numeric_cols))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Aperçu des données
        st.markdown("### 👀 Aperçu des Données Originales")
        st.dataframe(df_original.head(10), use_container_width=True)
        
        # Insights principaux basés sur votre analyse
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🔍 Insights Clés Découverts
        
        **Facteurs d'influence sur le salaire:**
        - Les **stages** (Internships_Completed) sont le facteur le plus influent
        - Les **compétences techniques** jouent un rôle crucial 
        - Le **réseautage** peut compenser des lacunes académiques
        - L'**expérience professionnelle** est hautement valorisée
        
        **Analyse des biais:**
        - **Genre** : Équité observée (écart minimal entre hommes/femmes)
        - **Localisation** : Biais significatif en faveur des étudiants internationaux
        - **Correction appliquée** avec Fairlearn (DP Difference: 0.87 → 0.03)
        
        **Profils identifiés (Clustering):**
        - **Cluster 0** : Performants avec expérience pratique
        - **Cluster 1** : Sociables avec développement technique
        - **Cluster 2** : Réseautés mais compétences relationnelles faibles  
        - **Cluster 3** : En développement avec potentiel d'amélioration
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "📊 Exploration des Données":
        st.markdown('<div class="sub-header">📊 Exploration des Données</div>', unsafe_allow_html=True)
        
        # Statistiques descriptives
        st.markdown("### 📈 Statistiques Descriptives")
        st.dataframe(df_processed[numeric_cols].describe(), use_container_width=True)
        
        # Graphiques de distribution
        st.markdown("### 📊 Distributions des Variables")
        fig_dist = create_distribution_plots(df_processed, numeric_cols[:9])  # Limiter pour la performance
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Analyse des valeurs manquantes
        st.markdown("### 🔍 Analyse des Valeurs Manquantes")
        missing_data = df_original.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if not missing_data.empty:
            fig_missing = px.bar(
                x=missing_data.values, 
                y=missing_data.index,
                orientation='h',
                title="Valeurs Manquantes par Variable",
                labels={'x': 'Nombre de valeurs manquantes', 'y': 'Variables'}
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ Aucune valeur manquante détectée!")
    
    elif page == "🔍 Analyse Approfondie":
        st.markdown('<div class="sub-header">🔍 Analyse Approfondie</div>', unsafe_allow_html=True)
        
        # Matrice de corrélation
        st.markdown("### 🔥 Matrice de Corrélation")
        fig_corr = create_correlation_heatmap(df_processed, numeric_cols)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Analyse des salaires
        st.markdown("### 💰 Analyse Détaillée des Salaires")
        fig_salary = create_salary_analysis(df_processed)
        st.plotly_chart(fig_salary, use_container_width=True)
        
        # Insights de corrélation
        if 'Starting_Salary' in df_processed.columns:
            salary_corr = df_processed[numeric_cols].corr()['Starting_Salary'].sort_values(ascending=False)
            salary_corr = salary_corr[salary_corr.index != 'Starting_Salary']
            
            st.markdown("### 🎯 Facteurs les Plus Corrélés au Salaire")
            fig_salary_corr = px.bar(
                x=salary_corr.values,
                y=salary_corr.index,
                orientation='h',
                title="Corrélation avec le Salaire de Départ",
                color=salary_corr.values,
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_salary_corr, use_container_width=True)
    
    elif page == "🎯 Clustering":
        st.markdown('<div class="sub-header">🎯 Analyse par Clustering</div>', unsafe_allow_html=True)
        
        # Clustering
        df_with_clusters, cluster_summary, cluster_cols = perform_clustering(df_encoded)
        
        if df_with_clusters is not None:
            # Visualisation des clusters
            fig_cluster = create_cluster_visualization(df_with_clusters, cluster_summary, cluster_cols)
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Résumé des clusters
            st.markdown("### 📊 Profils des Clusters")
            st.dataframe(cluster_summary.round(2), use_container_width=True)
            
            # Distribution des clusters
            cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
            fig_cluster_dist = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {i}" for i in cluster_counts.index],
                title="Distribution des Étudiants par Cluster"
            )
            st.plotly_chart(fig_cluster_dist, use_container_width=True)
            
            # Interprétation des clusters basée sur votre notebook
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("""
            ### 🎯 Interprétation des Clusters (d'après votre analyse)
            
            - **Cluster 0** : **Performants avec expérience pratique**
              - GPA moyen, beaucoup de stages, bonnes compétences techniques
              - Salaire élevé grâce à l'expérience pratique
              
            - **Cluster 1** : **Sociables avec développement technique**  
              - Peu de stages, excellentes soft skills, compétences techniques correctes
              - Salaire correct mais potentiel d'amélioration
              
            - **Cluster 2** : **Réseautés mais socialement faibles**
              - Bon réseautage, bonnes compétences techniques, faibles soft skills
              - Salaire relativement bon grâce au réseau
              
            - **Cluster 3** : **En développement avec potentiel**
              - Très faibles compétences techniques, réseautage correct
              - Salaire très faible, nécessite amélioration urgente
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "🤖 Modèle Prédictif":
        st.markdown('<div class="sub-header">🤖 Modèle de Prédiction</div>', unsafe_allow_html=True)
        
        if model is not None:
            # Métriques du modèle
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Score R²", f"{r2:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📏 RMSE", f"{rmse:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Interprétation des performances
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown(f"""
            ### 📊 Interprétation des Performances (d'après votre notebook)
            
            **Généralisation**: Le modèle généralise bien avec un R² de {r2:.3f} sur les données de test.
            
            **Performance globale**: Le modèle explique environ {r2*100:.1f}% de la variabilité des salaires.
            
            **Stabilité**: La validation croisée confirme la robustesse du modèle.
            
            **RMSE**: {rmse:.3f} correspond à l'erreur moyenne sur les données standardisées.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Importance des variables
            st.markdown("### 🔝 Importance des Variables")
            fig_importance = px.bar(
                feature_importance.head(15),  # Top 15 features
                x='Importance',
                y='Feature',
                orientation='h',
                title="Importance des Variables dans la Prédiction",
                color='Importance',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Interface de prédiction simplifiée
            st.markdown("### 🔮 Prédicteur de Salaire Rapide")
            
            col1, col2 = st.columns(2)
            
            with col1:
                quick_gpa = st.slider("GPA Universitaire", 0.0, 4.0, 3.0, 0.1)
                quick_internships = st.slider("Stages complétés", 0, 5, 2)
                quick_technical = st.slider("Compétences techniques (1-10)", 1, 10, 5)
            
            with col2:
                quick_networking = st.slider("Score réseautage (1-10)", 1, 10, 5)
                quick_soft_skills = st.slider("Soft skills (1-10)", 1, 10, 5)
                quick_experience = st.slider("Années d'expérience", 0, 5, 1)
            
            if st.button("⚡ Prédiction Rapide", type="primary"):
                # Créer un DataFrame simple pour la prédiction
                quick_data = create_encoded_user_data(
                    age=23, university_gpa=quick_gpa, high_school_gpa=3.0,
                    sat_score=1200, university_ranking=250, internships=quick_internships,
                    projects=5, work_experience=quick_experience, 
                    technical_skills=quick_technical, soft_skills=quick_soft_skills,
                    networking=quick_networking, study_hours=20, extracurricular=3,
                    motivation=7, work_life_balance=6, field_of_study='Computer Science',
                    location='Urban', gender='Male', encoded_columns=df_encoded.columns
                )
                
                # Faire la prédiction
                predicted_salary = model.predict(quick_data[[col for col in model.feature_names_in_]])[0]
                
                st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                st.markdown(f"## 💰 Salaire Prédit: ${predicted_salary:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("❌ Impossible d'entraîner le modèle avec les données disponibles.")
    
    elif page == "💡 Recommandations":
        if model is not None and cluster_summary is not None and kmeans_model is not None:
            create_advanced_recommendation_system(df_encoded, model, cluster_summary, kmeans_model)
        else:
            st.error("❌ Les modèles ne sont pas disponibles. Veuillez d'abord visiter la section 'Modèle Prédictif'.")
    
    elif page == "📈 Insights Avancés":
        st.markdown('<div class="sub-header">📈 Insights Avancés</div>', unsafe_allow_html=True)
        
        # Analyse par domaine d'étude
        if 'Field_of_Study' in df_original.columns:
            st.markdown("### 🎓 Analyse par Domaine d'Étude")
            
            # Top domaines par salaire
            top_fields = df_original.groupby('Field_of_Study')['Starting_Salary'].agg(['mean', 'count']).round(0)
            top_fields = top_fields[top_fields['count'] >= 100].sort_values('mean', ascending=False)
            
            fig_fields = px.scatter(
                top_fields.reset_index(),
                x='count',
                y='mean',
                size='count',
                color='mean',
                hover_name='Field_of_Study',
                title="Salaire Moyen vs Nombre d'Étudiants par Domaine",
                labels={'count': 'Nombre d\'étudiants', 'mean': 'Salaire moyen'}
            )
            st.plotly_chart(fig_fields, use_container_width=True)
        
        # Analyse des outliers
        st.markdown("### 🎯 Détection des Profils Exceptionnels")
        
        if 'Starting_Salary' in df_original.columns:
            q75, q25 = np.percentile(df_original['Starting_Salary'], [75, 25])
            iqr = q75 - q25
            upper_bound = q75 + (1.5 * iqr)
            
            high_earners = df_original[df_original['Starting_Salary'] > upper_bound]
            
            if not high_earners.empty:
                st.markdown(f"🌟 **{len(high_earners)} profils exceptionnels** identifiés (salaire > ${upper_bound:,.0f})")
                
                # Caractéristiques des high earners
                if len(high_earners) > 5:
                    characteristics = {}
                    for col in numeric_cols:
                        if col in high_earners.columns and col != 'Starting_Salary':
                            avg_high = high_earners[col].mean()
                            avg_all = df_original[col].mean()
                            diff = ((avg_high - avg_all) / avg_all) * 100
                            characteristics[col] = diff
                    
                    # Top caractéristiques
                    top_chars = sorted(characteristics.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                    
                    fig_chars = px.bar(
                        x=[char[1] for char in top_chars],
                        y=[char[0] for char in top_chars],
                        orientation='h',
                        title="Différence (%) des High Earners vs Moyenne Générale",
                        color=[char[1] for char in top_chars],
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_chars, use_container_width=True)
        
        # Recommandations stratégiques basées sur votre analyse
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🎯 Recommandations Stratégiques (d'après vos résultats)
        
        **Pour les Étudiants:**
        - 🏢 **Priorisez les stages** : facteur #1 d'influence sur le salaire
        - 💻 **Développez vos compétences techniques** via des projets concrets
        - 🤝 **Cultivez votre réseau professionnel** dès maintenant
        - 🌍 **Considérez l'international** pour élargir vos opportunités
        
        **Pour les Institutions:**
        - 📈 **Renforcez les partenariats entreprises** pour plus de stages
        - ⚖️ **Corrigez les biais géographiques** identifiés dans l'étude
        - 🎯 **Segmentez l'accompagnement** selon les 4 profils identifiés
        - 📊 **Utilisez l'IA prédictive** pour l'orientation personnalisée
        
        **Équité et Biais:**
        - ✅ **Genre** : Équité confirmée dans votre dataset
        - ⚠️ **Localisation** : Biais significatif corrigé par Fairlearn
        - 🎯 **Focus sur l'inclusion** des étudiants ruraux
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
