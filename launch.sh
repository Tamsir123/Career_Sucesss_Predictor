#!/bin/bash

echo "🎓 ===== LANCEMENT DES APPLICATIONS D'ANALYSE ÉDUCATION & CARRIÈRE ====="
echo ""

# Couleurs pour l'affichage
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Vérifier si Streamlit est installé
if ! command -v streamlit &> /dev/null; then
    echo -e "${RED}❌ Streamlit n'est pas installé. Installation en cours...${NC}"
    pip install streamlit
fi

# Vérifier les fichiers de données
if [ ! -f "education_career_success_g.csv" ]; then
    echo -e "${RED}❌ Fichier education_career_success_g.csv manquant${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Vérifications terminées${NC}"
echo ""

# Menu de sélection
echo "Choisissez l'application à lancer :"
echo ""
echo "1) 📊 Application Principale (app.py) - Port 8502"
echo "2) 🚀 Application Avancée (app_advanced.py) - Port 8503" 
echo "3) 🔧 Test des Données (test_data.py) - Port 8504"
echo "4) 🎪 Toutes les Applications"
echo ""

read -p "Votre choix (1-4) : " choice

case $choice in
    1)
        echo -e "${BLUE}🚀 Lancement de l'Application Principale...${NC}"
        streamlit run app.py --server.port 8502
        ;;
    2)
        echo -e "${BLUE}🚀 Lancement de l'Application Avancée...${NC}"
        streamlit run app_advanced.py --server.port 8503
        ;;
    3)
        echo -e "${BLUE}🚀 Lancement du Test des Données...${NC}"
        streamlit run test_data.py --server.port 8504
        ;;
    4)
        echo -e "${BLUE}🚀 Lancement de toutes les applications...${NC}"
        echo ""
        echo "📊 Application Principale : http://localhost:8502"
        echo "🚀 Application Avancée : http://localhost:8503"
        echo "🔧 Test des Données : http://localhost:8504"
        echo ""
        
        # Lancer toutes les applications en arrière-plan
        streamlit run app.py --server.port 8502 &
        sleep 2
        streamlit run app_advanced.py --server.port 8503 &
        sleep 2
        streamlit run test_data.py --server.port 8504 &
        
        echo -e "${GREEN}✅ Toutes les applications sont lancées !${NC}"
        echo ""
        echo "Pour arrêter toutes les applications, appuyez sur Ctrl+C"
        wait
        ;;
    *)
        echo -e "${RED}❌ Choix invalide${NC}"
        exit 1
        ;;
esac
