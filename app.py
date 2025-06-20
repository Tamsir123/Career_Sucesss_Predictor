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
        df = pd.read_csv('education_career_success_g.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Fichier de données non trouvé. Assurez-vous que 'education_career_success_g.csv' est dans le répertoire.")
        return None

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

def perform_clustering(df, numeric_cols):
    """Effectue le clustering K-Means"""
    # Sélection des variables pour le clustering (comme dans votre notebook)
    cluster_cols = ['University_GPA', 'Internships_Completed', 'Technical_Skills_Score', 
                   'Soft_Skills_Score', 'Networking_Score', 'Starting_Salary']
    
    # Vérifier que toutes les colonnes existent
    available_cols = [col for col in cluster_cols if col in df.columns]
    
    if len(available_cols) < 3:
        st.warning("⚠️ Pas assez de colonnes disponibles pour le clustering")
        return None, None, None
    
    X_cluster = df[available_cols].copy()
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # K-Means avec 4 clusters comme dans votre notebook
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Ajout des clusters au DataFrame
    df_with_clusters = df.copy()
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

def train_prediction_model(df):
    """Entraîne un modèle de prédiction Random Forest"""
    # Préparation des données pour la prédiction (comme dans votre notebook)
    # Exclure la variable cible et les variables non pertinentes
    exclude_cols = ['Starting_Salary', 'Student_ID', 'Field_of_Study', 'Location', 'Gender', 
                   'Current_Job_Level', 'Languages_Spoken', 'Certifications', 'Entrepreneurship', 'Remote_Work']
    
    # Sélectionner toutes les colonnes sauf celles à exclure
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Vérifier les colonnes disponibles
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < 5 or 'Starting_Salary' not in df.columns:
        st.warning("⚠️ Pas assez de variables pour entraîner le modèle de prédiction")
        return None, None, None, None
    
    X = df[available_features].copy()
    y = df['Starting_Salary'].copy()
    
    # Vérifier qu'il n'y a pas de valeurs manquantes
    if X.isnull().sum().sum() > 0:
        st.warning("⚠️ Valeurs manquantes détectées dans les features")
        return None, None, None, None
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement du modèle
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Prédictions et métriques
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Importance des variables
    feature_importance = pd.DataFrame({
        'Feature': available_features,
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

def create_recommendation_system():
    """Système de recommandations personnalisées"""
    st.markdown('<div class="sub-header">🎯 Système de Recommandations Personnalisées</div>', unsafe_allow_html=True)
    
    # Interface utilisateur pour saisir les informations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Informations Académiques")
        gpa = st.slider("GPA Universitaire", 0.0, 4.0, 3.0, 0.1)
        sat_score = st.slider("Score SAT", 400, 1600, 1200, 50)
        internships = st.slider("Nombre de stages", 0, 10, 2)
        projects = st.slider("Projets complétés", 0, 20, 5)
    
    with col2:
        st.markdown("### 🎯 Compétences")
        technical_skills = st.slider("Compétences techniques (1-10)", 1, 10, 5)
        soft_skills = st.slider("Compétences relationnelles (1-10)", 1, 10, 5)
        networking = st.slider("Score de réseautage (1-10)", 1, 10, 5)
        study_hours = st.slider("Heures d'étude/semaine", 0, 50, 20)
    
    if st.button("🚀 Générer des Recommandations", type="primary"):
        # Calcul du profil et recommandations
        profile_score = (gpa/4 + sat_score/1600 + internships/10 + 
                        technical_skills/10 + soft_skills/10 + networking/10) / 6
        
        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
        st.markdown(f"## 📊 Votre Score de Profil: {profile_score:.1%}")
        
        # Recommandations basées sur le profil
        recommendations = []
        
        if gpa < 3.5:
            recommendations.append({
                "domaine": "📚 Académique",
                "action": "Améliorer le GPA",
                "suggestion": "Participez à des groupes d'étude, consultez vos professeurs, et organisez mieux votre temps",
                "impact": "Élevé"
            })
        
        if internships < 3:
            recommendations.append({
                "domaine": "💼 Expérience",
                "action": "Augmenter les stages",
                "suggestion": "Recherchez activement des stages, utilisez LinkedIn, et contactez votre réseau",
                "impact": "Très Élevé"
            })
        
        if technical_skills < 7:
            recommendations.append({
                "domaine": "⚙️ Compétences Techniques",
                "action": "Développer les compétences techniques",
                "suggestion": "Suivez des cours en ligne, participez à des projets open source, obtenez des certifications",
                "impact": "Élevé"
            })
        
        if soft_skills < 7:
            recommendations.append({
                "domaine": "🤝 Compétences Relationnelles",
                "action": "Améliorer les soft skills",
                "suggestion": "Rejoignez des clubs, pratiquez la prise de parole en public, développez votre intelligence émotionnelle",
                "impact": "Moyen"
            })
        
        if networking < 6:
            recommendations.append({
                "domaine": "🌐 Réseautage",
                "action": "Étendre votre réseau",
                "suggestion": "Assistez à des événements professionnels, utilisez LinkedIn activement, rejoignez des associations",
                "impact": "Moyen"
            })
        
        # Affichage des recommandations
        if recommendations:
            st.markdown("### 🎯 Recommandations Prioritaires:")
            for i, rec in enumerate(recommendations[:3], 1):
                st.markdown(f"""
                **{i}. {rec['domaine']} - {rec['action']}**
                - 💡 **Suggestion**: {rec['suggestion']}
                - 📈 **Impact estimé**: {rec['impact']}
                """)
        else:
            st.markdown("🎉 **Excellent profil!** Continuez sur cette voie et explorez des opportunités de leadership.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prédiction de salaire estimée
        estimated_salary = 45000 + (profile_score * 40000) + (internships * 5000) + (technical_skills * 2000)
        st.markdown(f"### 💰 Salaire de départ estimé: ${estimated_salary:,.0f}")

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
    df = load_data()
    if df is None:
        st.stop()
    
    df_processed, numeric_cols, categorical_cols = prepare_data(df)
    
    # Navigation entre les pages
    if page == "🏠 Accueil":
        st.markdown("""
        ## 🌟 Bienvenue dans l'Analyse du Succès Éducatif
        
        Cette application interactive explore les facteurs qui influencent le succès dans l'éducation et la carrière. 
        
        ### 🎯 Objectifs du Projet
        - **Identifier** les facteurs clés de succès professionnel
        - **Analyser** les relations entre éducation et salaire de départ
        - **Proposer** des recommandations personnalisées
        - **Prédire** les résultats de carrière
        
        ### 📊 Données Analysées
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📚 Étudiants", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📋 Variables", len(df.columns))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💰 Salaire Moyen", f"${df['Starting_Salary'].mean():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'Field_of_Study' in df.columns:
                st.metric("🎓 Domaines", df['Field_of_Study'].nunique())
            else:
                st.metric("🔢 Variables Numériques", len(numeric_cols))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Aperçu des données
        st.markdown("### 👀 Aperçu des Données")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Insights principaux
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🔍 Insights Clés Découverts
        - Les **stages** sont le facteur le plus influent sur le salaire de départ
        - Les **compétences techniques** jouent un rôle crucial dans certains domaines
        - Le **réseautage** peut compenser des lacunes académiques
        - Les **étudiants internationaux** obtiennent en moyenne des salaires plus élevés
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
        missing_data = df.isnull().sum()
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
        df_with_clusters, cluster_summary, cluster_cols = perform_clustering(df_processed, numeric_cols)
        
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
            
            # Interprétation des clusters
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("""
            ### 🎯 Interprétation des Clusters
            - **Cluster 0**: Profils équilibrés avec performances moyennes
            - **Cluster 1**: Excellence académique et forte rémunération
            - **Cluster 2**: Compétences techniques élevées, réseautage variable  
            - **Cluster 3**: Profils en développement avec potentiel d'amélioration
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "🤖 Modèle Prédictif":
        st.markdown('<div class="sub-header">🤖 Modèle de Prédiction</div>', unsafe_allow_html=True)
        
        # Entraînement du modèle
        model, feature_importance, r2, rmse = train_prediction_model(df_processed)
        
        if model is not None:
            # Métriques du modèle
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Score R²", f"{r2:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📏 RMSE", f"${rmse:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Importance des variables
            st.markdown("### 🔝 Importance des Variables")
            fig_importance = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Importance des Variables dans la Prédiction",
                color='Importance',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Interface de prédiction
            st.markdown("### 🔮 Prédicteur de Salaire Interactif")
            
            # Créer des sliders pour chaque variable importante
            col1, col2 = st.columns(2)
            prediction_values = {}
            
            top_features = feature_importance.head(6)['Feature'].tolist()
            
            for i, feature in enumerate(top_features):
                col = col1 if i % 2 == 0 else col2
                with col:
                    if feature in df_processed.columns:
                        min_val = float(df_processed[feature].min())
                        max_val = float(df_processed[feature].max())
                        default_val = float(df_processed[feature].median())
                        
                        prediction_values[feature] = st.slider(
                            f"{feature}",
                            min_val, max_val, default_val
                        )
            
            # Compléter avec les valeurs médianes pour les autres features
            for feature in model.feature_names_in_:
                if feature not in prediction_values:
                    prediction_values[feature] = df_processed[feature].median()
            
            # Bouton de prédiction
            if st.button("🚀 Prédire le Salaire", type="primary"):
                # Préparer les données pour la prédiction
                input_data = pd.DataFrame([prediction_values])
                input_data = input_data[model.feature_names_in_]
                
                # Faire la prédiction
                predicted_salary = model.predict(input_data)[0]
                
                st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                st.markdown(f"## 💰 Salaire Prédit: ${predicted_salary:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("❌ Impossible d'entraîner le modèle avec les données disponibles.")
    
    elif page == "💡 Recommandations":
        create_recommendation_system()
    
    elif page == "📈 Insights Avancés":
        st.markdown('<div class="sub-header">📈 Insights Avancés</div>', unsafe_allow_html=True)
        
        # Analyse par domaine d'étude
        if 'Field_of_Study' in df.columns:
            st.markdown("### 🎓 Analyse par Domaine d'Étude")
            
            # Top domaines par salaire
            top_fields = df.groupby('Field_of_Study')['Starting_Salary'].agg(['mean', 'count']).round(0)
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
        
        if 'Starting_Salary' in df.columns:
            q75, q25 = np.percentile(df['Starting_Salary'], [75, 25])
            iqr = q75 - q25
            upper_bound = q75 + (1.5 * iqr)
            
            high_earners = df[df['Starting_Salary'] > upper_bound]
            
            if not high_earners.empty:
                st.markdown(f"🌟 **{len(high_earners)} profils exceptionnels** identifiés (salaire > ${upper_bound:,.0f})")
                
                # Caractéristiques des high earners
                if len(high_earners) > 5:
                    characteristics = {}
                    for col in numeric_cols:
                        if col in high_earners.columns and col != 'Starting_Salary':
                            avg_high = high_earners[col].mean()
                            avg_all = df[col].mean()
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
        
        # Recommandations stratégiques
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🎯 Recommandations Stratégiques pour les Institutions
        
        **Pour améliorer l'employabilité des étudiants:**
        - 🏢 **Renforcer les partenariats** avec l'industrie pour plus de stages
        - 💻 **Développer les compétences techniques** via des projets pratiques
        - 🤝 **Encourager le réseautage** professionnel dès la première année
        - 🌍 **Promouvoir les échanges internationaux** pour élargir les perspectives
        - 📊 **Mettre en place un suivi personnalisé** basé sur les profils étudiants
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
