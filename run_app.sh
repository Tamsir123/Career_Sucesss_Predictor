#!/bin/bash

echo "🎓 Lancement de l'Application d'Analyse Éducation & Carrière"
echo "=============================================="

# Vérifier si Streamlit est installé
if ! command -v streamlit &> /dev/null
then
    echo "⚠️  Streamlit n'est pas installé. Installation en cours..."
    pip install streamlit
fi

# Vérifier si le fichier de données existe
if [ ! -f "education_career_success_g.csv" ]; then
    echo "❌ Fichier de données manquant: education_career_success_g.csv"
    echo "Assurez-vous que le fichier se trouve dans le répertoire courant."
    exit 1
fi

echo "✅ Vérifications terminées"
echo "🚀 Lancement de l'application..."
echo ""
echo "💡 L'application s'ouvrira automatiquement dans votre navigateur"
echo "📍 URL: http://localhost:8501"
echo ""
echo "Pour arrêter l'application, appuyez sur Ctrl+C"
echo ""

# Lancer l'application Streamlit
streamlit run app.py
