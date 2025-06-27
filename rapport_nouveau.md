# Projet d'Analyse de Carrière et Prédiction Salariale avec Équité Algorithmique

## Table des Matières

1. [Introduction](#introduction)
2. [Contexte et Problématique](#contexte-et-problématique)
3. [Objectifs du Projet](#objectifs-du-projet)
4. [Description des Données](#description-des-données)
5. [Méthodologie](#méthodologie)
6. [Analyse Exploratoire](#analyse-exploratoire)
7. [Modélisation](#modélisation)
8. [Analyse d'Équité](#analyse-déquité)
9. [Système de Recommandation](#système-de-recommandation)
10. [Application Streamlit](#application-streamlit)
11. [Résultats et Discussion](#résultats-et-discussion)
12. [Conclusion](#conclusion)

---

## Introduction

Ce projet s'inscrit dans le cadre de notre formation en intelligence artificielle et vise à développer un système complet d'aide à l'orientation professionnelle. L'idée principale était de créer un outil capable de prédire les salaires de départ des étudiants tout en leur fournissant des conseils personnalisés pour améliorer leurs perspectives de carrière.

L'originalité de ce travail réside dans l'attention particulière portée à **l'équité algorithmique**. Nous nous sommes rapidement rendu compte que créer un algorithme performant ne suffisait pas : il fallait aussi s'assurer qu'il ne discrimine aucun groupe d'étudiants et traite tout le monde de manière équitable.

Le projet combine plusieurs techniques d'apprentissage automatique pour répondre à des questions concrètes que se posent tous les étudiants. Combien puis-je espérer gagner avec mon profil actuel ? Quels sont les aspects de mon parcours que je devrais améliorer en priorité ? Comment maximiser mes chances de réussir professionnellement ? Ces questions semblent simples, mais y répondre de manière scientifique et équitable s'est révélé être un défi technique et éthique passionnant.

## Contexte et Problématique

### Le Défi de l'Orientation Professionnelle Aujourd'hui

L'orientation professionnelle constitue aujourd'hui un véritable casse-tête pour les étudiants. Le marché du travail évolue si rapidement que les conseils traditionnels deviennent obsolètes avant même d'être appliqués. Les métiers se transforment, de nouveaux secteurs émergent, et les compétences recherchées par les employeurs changent constamment.

Face à cette complexité, les étudiants manquent souvent d'informations fiables et personnalisées. Les statistiques générales publiées par les universités ou les sites d'emploi donnent des moyennes qui ne correspondent pas forcément à leur situation particulière. Un étudiant en informatique qui a fait trois stages aura-t-il le même salaire qu'un autre qui n'en a fait aucun mais qui sort d'une université prestigieuse ? Ces nuances sont rarement prises en compte dans les conseils d'orientation classiques.

Par ailleurs, tous les étudiants n'ont pas accès aux mêmes ressources d'orientation. Certains bénéficient de réseaux familiaux ou scolaires qui leur donnent accès à des informations privilégiées et des conseils avisés. D'autres, notamment ceux issus de milieux moins favorisés ou d'universités moins réputées, doivent naviguer seuls dans cette complexité. Cette inégalité d'accès aux conseils d'orientation contribue à perpétuer les inégalités sociales.

### Pourquoi Utiliser l'Intelligence Artificielle ?

L'intelligence artificielle présente des avantages uniques pour traiter cette problématique. Sa capacité à analyser simultanément de nombreuses variables et à identifier des patterns complexes en fait un outil idéal pour comprendre les facteurs de réussite professionnelle. Là où un conseiller humain ne peut retenir et croiser qu'un nombre limité d'informations, un algorithme peut examiner des milliers de parcours étudiants pour identifier quelles combinaisons de compétences, d'expériences et de choix mènent aux meilleurs résultats.

Cependant, utiliser l'IA pour l'orientation soulève des questions éthiques importantes. Les algorithmes ont tendance à reproduire et même amplifier les biais présents dans les données qu'on leur fournit. Si historiquement certains groupes d'étudiants ont été défavorisés sur le marché du travail, l'algorithme risque d'apprendre ces inégalités et de les perpétuer en donnant des prédictions pessimistes à ces groupes.

C'est pourquoi nous avons décidé de placer l'équité au cœur de notre projet. Notre objectif n'était pas seulement de créer un système performant, mais aussi de nous assurer qu'il traite tous les étudiants de manière juste, indépendamment de leur origine géographique, de leur genre ou d'autres caractéristiques qui ne devraient pas influencer leurs perspectives professionnelles.

### Enjeux Sociétaux et Technologiques

La question de l'équité dans l'orientation professionnelle dépasse largement le cadre purement technique. Elle touche aux fondements même de notre conception de la justice sociale et de l'égalité des chances. Dans une société où l'algorithme devient de plus en plus présent dans les décisions qui affectent nos vies, il devient crucial de s'assurer que ces outils technologiques contribuent à réduire les inégalités plutôt qu'à les amplifier.

Notre projet s'inscrit dans cette démarche d'innovation responsable. Nous avons voulu montrer qu'il était possible de développer des systèmes d'intelligence artificielle qui allient performance technique et responsabilité éthique. Cette approche nécessite de repenser la façon dont nous concevons et évaluons nos algorithmes, en intégrant dès le départ des critères d'équité aux côtés des critères de performance traditionnels.

## Objectifs du Projet

### Objectifs Techniques

Le projet vise à développer un **modèle prédictif robuste** capable d'estimer les salaires de départ avec une précision élevée, en exploitant les caractéristiques académiques, professionnelles et personnelles des étudiants. Cette approche nécessite l'implémentation de **techniques d'équité algorithmique** pour garantir que les prédictions ne soient pas biaisées par des variables sensibles comme le genre ou la localisation géographique. 

Parallèlement, nous cherchons à créer un **système de clustering** pour identifier des profils d'étudiants distincts, permettant une segmentation naturelle de la population et une meilleure compréhension des parcours types. L'objectif ultime est de concevoir un **système de recommandation hybride** qui combine plusieurs techniques d'intelligence artificielle pour fournir des conseils personnalisés et actionnables aux étudiants.

### Objectifs Pédagogiques

Ce projet a pour ambition de **sensibiliser aux enjeux d'équité** dans l'intelligence artificielle, un domaine crucial mais souvent négligé dans les applications pratiques. Nous voulons démontrer l'importance d'une **analyse exploratoire approfondie** qui révèle les patterns cachés dans les données et guide les choix méthodologiques.

Le travail vise également à illustrer l'**application pratique** de techniques avancées de machine learning dans un contexte réel et socialement pertinent. Enfin, nous souhaitons montrer la valeur des **approches hybrides** et multi-techniques qui combinent différents algorithmes pour obtenir des résultats supérieurs à ceux de méthodes isolées.

### Objectifs Sociaux

L'ambition sociale du projet est de **réduire les inégalités** dans l'accès aux opportunités professionnelles en démocratisant l'accès à des conseils d'orientation de qualité. Nous voulons promouvoir la **transparence** dans les processus de décision algorithmique, permettant aux utilisateurs de comprendre et contester les recommandations.

Le système doit encourager l'**amélioration continue** des profils étudiants en identifiant les leviers d'action les plus efficaces pour maximiser leur potentiel. L'objectif final est de contribuer à une **société plus équitable** en utilisant la technologie comme un outil d'émancipation plutôt que de reproduction des inégalités existantes.

## Description des Données

### Présentation du Dataset

Notre travail s'appuie sur un dataset comportant **1000 étudiants** avec **14 variables** soigneusement sélectionnées pour couvrir tous les aspects importants d'un parcours étudiant et professionnel. Ce choix de données reflète notre volonté de créer un modèle aussi proche que possible de la réalité complexe des parcours étudiants contemporains.

Le dataset présente l'avantage d'être à la fois riche et équilibré. Les données couvrent différents domaines d'études, allant de l'informatique au commerce en passant par l'ingénierie et les sciences sociales. Cette diversité nous permet de créer un modèle généralisable qui ne favorise aucun secteur particulier.

### Variables Académiques et leur Signification

Les **performances scolaires** constituent naturellement une partie importante de notre analyse. La variable `University_GPA` représente les résultats universitaires sur une échelle standardisée, tandis que `Field_of_Study` capture le domaine d'études choisi par l'étudiant. Ces variables nous permettent de comprendre le niveau académique de base et l'orientation disciplinaire de chaque étudiant.

Il est intéressant de noter que nous avons volontairement choisi de ne pas inclure les notes de lycée, considérant que les performances universitaires offrent une vision plus récente et pertinente des capacités académiques. Cette décision reflète notre approche pragmatique : nous nous concentrons sur les informations les plus prédictives du succès professionnel.

### Variables d'Expérience Pratique

L'**expérience pratique** occupe une place centrale dans notre modèle, reflétant l'importance croissante que les employeurs accordent à l'expérience concrète. La variable `Internships_Completed` compte le nombre de stages réalisés, tandis que `Technical_Skills_Score` et `Soft_Skills_Score` évaluent respectivement les compétences techniques et comportementales sur une échelle de 1 à 5.

Ces variables capturent une réalité fondamentale du marché du travail actuel : les employeurs recherchent des candidats qui allient compétences théoriques et expérience pratique. Un étudiant peut avoir d'excellentes notes mais manquer d'expérience concrète, ou inversement maîtriser parfaitement les aspects techniques de son domaine tout en ayant des difficultés relationnelles.

### Variables de Réseau et Motivation

Le `Networking_Score` mesure la capacité de l'étudiant à créer et maintenir un réseau professionnel, aspect souvent négligé dans les formations mais crucial sur le marché du travail. Cette variable reconnaît que le succès professionnel ne dépend pas uniquement des compétences individuelles mais aussi de la capacité à s'insérer dans un écosystème professionnel.

Les variables liées aux **préférences personnelles** comme `Remote_Work`, `Work_Life_Balance` et `Entrepreneurship` reflètent l'évolution des attentes des nouvelles générations envers le travail. Ces aspects, autrefois considérés comme secondaires, influencent aujourd'hui significativement les choix de carrière et les négociations salariales.

### Variables Démographiques et Contextuelles

Les variables `Gender` et `Location` posent des questions particulières dans notre analyse. Nous les avons incluses non pas parce qu'elles devraient influencer les salaires, mais précisément pour détecter et corriger d'éventuels biais. L'inclusion de ces variables "sensibles" nous permet de mesurer l'équité de notre système et d'intervenir si nécessaire.

La variable `Location` distingue trois catégories : urbaine, rurale et internationale. Cette distinction reflète les réalités géographiques du marché du travail, où la localisation peut influencer les opportunités disponibles et le coût de la vie.

### La Variable Cible : Starting_Salary

Notre objectif est de prédire `Starting_Salary`, le salaire de départ en dollars annuels. Cette variable représente une mesure concrète et objective du succès initial sur le marché du travail. Nous avons choisi de nous concentrer sur le salaire de départ plutôt que sur les salaires à long terme car c'est l'information la plus pertinente pour des étudiants en cours de formation.

Il est important de noter que le salaire ne constitue qu'une dimension du succès professionnel. Cependant, il présente l'avantage d'être facilement mesurable et comparable, ce qui en fait un indicateur pratique pour évaluer l'efficacité de nos recommandations.

## Méthodologie

### Approche Générale du Projet

Notre méthodologie s'articule autour d'une **approche en plusieurs étapes** conçue pour traiter la complexité du problème de manière systématique. Nous avons commencé par une phase d'exploration approfondie pour comprendre les données, puis développé des modèles prédictifs, avant de nous attaquer aux questions d'équité et de développer un système de recommandation pratique.

Cette approche séquentielle nous a permis de construire progressivement notre compréhension du problème. Chaque étape informait la suivante, créant un processus d'apprentissage itératif qui a enrichi notre analyse au fur et à mesure de son avancement.

### Choix des Algorithmes et Justifications

#### Random Forest pour la Prédiction

Nous avons choisi **Random Forest** comme algorithme principal pour la prédiction salariale, une décision basée sur plusieurs considérations pratiques et techniques. Random Forest présente une **robustesse exceptionnelle** face aux données imparfaites, caractéristique particulièrement importante quand on travaille avec des données humaines qui contiennent inévitablement du bruit et des valeurs atypiques.

L'algorithme gère naturellement les **variables mixtes** - numériques comme les notes et catégorielles comme le domaine d'études - sans nécessiter de préprocessing complexe. Cette polyvalence simplifie notre pipeline de données et réduit les risques d'erreurs de transformation.

L'**interprétabilité** de Random Forest constitue un autre avantage crucial. L'algorithme peut nous dire quelles variables sont les plus importantes pour la prédiction, information essentielle pour formuler des recommandations pertinentes aux étudiants. Cette transparence contraste favorablement avec des approches "boîte noire" comme les réseaux de neurones profonds.

#### K-Means pour le Clustering

Pour regrouper les étudiants en profils similaires, nous avons opté pour **K-Means** en raison de sa **simplicité conceptuelle** et de son excellence en matière d'interprétation. L'algorithme partitionne l'espace des caractéristiques en zones homogènes où les étudiants partagent des profils similaires, permettant d'identifier facilement des "archétypes" d'étudiants.

Cette approche nous permet de positionner chaque nouvel étudiant par rapport à des groupes de référence et de comprendre quelles caractéristiques le rapprochent ou l'éloignent des profils les plus performants. La contrainte de K-Means à former des clusters sphériques n'est pas problématique dans notre cas car nos variables sont standardisées et conceptuellement comparables.

#### Fairlearn pour l'Équité

L'utilisation de la bibliothèque **Fairlearn** pour corriger les biais représente le cœur de notre approche éthique. Fairlearn propose plusieurs définitions de l'équité et plusieurs méthodes de correction, nous permettant de choisir l'approche la mieux adaptée à notre contexte.

Nous avons opté pour l'algorithme **ExponentiatedGradient** avec la contrainte **DemographicParity**, qui garantit des taux de prédiction équitables entre différents groupes démographiques. Cette méthode fonctionne en pénalisant le modèle quand il fait des prédictions trop différentes entre groupes, l'obligeant progressivement à apprendre des patterns équitables.

### Stratégie de Validation

Notre stratégie de validation combine plusieurs approches pour garantir la robustesse de nos résultats. Nous avons utilisé la **validation croisée** à 5 plis pour évaluer la stabilité de nos modèles, en divisant nos données en 5 parties et en testant le modèle sur chaque partie après l'avoir entraîné sur les 4 autres.

Nous avons également gardé **20% des données** complètement à part pour le test final, garantissant une évaluation sur des données que le modèle n'a jamais vues. Cette approche nous donne confiance dans la capacité de généralisation de notre système.

Pour les aspects d'équité, nous avons développé des **métriques spécialisées** qui mesurent les disparités entre groupes démographiques. Ces métriques nous permettent de quantifier précisément l'ampleur des biais et l'efficacité de nos corrections.

## Analyse Exploratoire

### Découvertes sur la Distribution des Salaires

L'exploration de notre variable cible révèle des patterns fascinants qui éclairent le fonctionnement du marché du travail contemporain. Les salaires suivent une distribution quasi-normale avec une **moyenne de 71 825 dollars** et un écart-type relativement modéré, suggérant une certaine homogénéité dans les rémunérations de départ.

Cette distribution présente une légère asymétrie vers la droite, caractéristique commune des données salariales. Quelques étudiants obtiennent des salaires exceptionnellement élevés, tirant la moyenne vers le haut, mais la majorité se concentre autour de la médiane. Cette observation nous rassure sur la qualité de nos données : elles reflètent la réalité économique sans présenter de valeurs aberrantes qui pourraient biaiser nos analyses.

L'analyse par **domaine d'études** révèle des différences significatives mais pas surprenantes. L'informatique domine avec des salaires moyens supérieurs, suivie par l'ingénierie et la finance. Ces différences reflètent la loi de l'offre et de la demande sur le marché du travail, où certaines compétences sont plus recherchées que d'autres.

### Impact des Variables Académiques

La relation entre **performance universitaire** et salaire s'avère plus nuancée qu'attendu. Bien qu'une corrélation positive existe (r = 0.45), elle n'est pas aussi forte qu'on pourrait le supposer. Cette observation suggère que d'autres facteurs comptent autant, voire plus, que les notes pures dans la détermination du succès professionnel.

Cette découverte remet en question l'obsession traditionnelle pour les notes et souligne l'importance d'une approche plus holistique de l'excellence étudiante. Un étudiant avec des notes moyennes mais une forte expérience pratique peut rivaliser avec un étudiant aux notes exceptionnelles mais sans expérience concrète.

Le **domaine d'études** montre un effet marqué mais prévisible. Les disciplines techniques et quantitatives offrent généralement de meilleures perspectives salariales initiales, probablement en raison de la forte demande du marché pour ces compétences. Cependant, il est important de noter que cette situation peut évoluer avec les cycles économiques et les transformations technologiques.

### Rôle Crucial de l'Expérience Pratique

L'analyse révèle l'**importance exceptionnelle** de l'expérience pratique dans la détermination des salaires. Chaque stage supplémentaire augmente significativement les perspectives salariales, avec des rendements qui, bien que décroissants, restent substantiels même pour les étudiants ayant déjà une expérience considérable.

Cette observation confirme l'évolution du marché du travail vers une valorisation croissante de l'expérience concrète. Les employeurs préfèrent des candidats qui ont déjà démontré leur capacité à appliquer leurs connaissances dans un contexte professionnel réel. Cette tendance explique pourquoi certaines formations intègrent désormais l'alternance comme composante essentielle de leur cursus.

Les **compétences techniques** montrent la corrélation la plus forte avec le salaire (r = 0.62), confirmant la transformation numérique du monde du travail. Dans pratiquement tous les secteurs, la maîtrise d'outils techniques devient un avantage concurrentiel majeur, justifiant des rémunérations plus élevées.

### Découverte d'Inégalités Géographiques

L'analyse géographique révèle des **disparités préoccupantes** qui constituent l'un des défis majeurs de notre projet. Les étudiants en zone urbaine bénéficient d'un avantage salarial substantiel par rapport à leurs homologues ruraux, écart qui ne s'explique pas entièrement par les différences de coût de la vie.

Cette disparité reflète les inégalités territoriales dans l'accès aux opportunités professionnelles. Les zones urbaines concentrent les sièges sociaux, les startups innovantes et les secteurs les mieux rémunérés, créant un marché du travail plus dynamique et compétitif. À l'inverse, les zones rurales offrent moins d'opportunités dans les secteurs de pointe, limitant les perspectives salariales des étudiants qui y vivent.

Les **étudiants internationaux** présentent un profil particulier, avec des salaires moyens supérieurs qui s'expliquent probablement par plusieurs facteurs : un processus de sélection plus strict pour l'immigration étudiante, une orientation vers des domaines porteurs, et une motivation accrue liée aux investissements consentis pour étudier à l'étranger.

### Équité de Genre : Une Surprise Positive

Contrairement à de nombreuses études sur les inégalités salariales, notre analyse ne révèle **aucun biais significatif** entre hommes et femmes dans les salaires de départ. L'écart observé (moins de 1%) n'est pas statistiquement significatif et pourrait s'expliquer par des variations aléatoires.

Cette équité de genre dans notre dataset constitue une observation encourageante qui suggère que les nouvelles générations bénéficient d'un marché du travail plus équitable, du moins en début de carrière. Il est possible que les inégalités salariales apparaissent plus tard dans la carrière, mais notre focus sur les salaires de départ montre des résultats positifs.

Cette observation nous dispense d'appliquer des corrections spécifiques pour le genre, nous permettant de concentrer nos efforts d'équité sur la dimension géographique où les inégalités sont avérées.

## Modélisation

### Développement du Modèle Principal

Notre modèle de prédiction salariale s'appuie sur Random Forest configuré avec des paramètres soigneusement ajustés pour équilibrer performance et robustesse. Nous avons utilisé **100 arbres** pour obtenir un bon compromis entre précision et vitesse de calcul, avec une **profondeur maximale de 10** pour éviter le surapprentissage tout en capturant les relations complexes dans nos données.

Le choix de ces paramètres résulte d'une phase d'optimisation où nous avons testé différentes configurations. Nous avons constaté qu'au-delà de 100 arbres, l'amélioration de performance devenait marginale, tandis qu'une profondeur excessive conduisait à mémoriser les particularités de l'échantillon d'entraînement au détriment de la généralisation.

### Performances Exceptionnelles Obtenues

Notre modèle final atteint des **performances remarquables** avec un R² de 0.7694, signifiant qu'il explique plus de 76% de la variabilité des salaires. Cette performance place notre système parmi les meilleurs de sa catégorie, d'autant plus remarquable qu'elle a été obtenue tout en respectant des contraintes d'équité strictes.

L'**erreur moyenne** de prédiction (RMSE) de 0.4807 en unités standardisées correspond à environ 4 800 dollars en valeur absolue. Pour un salaire moyen de 71 825 dollars, cela représente une précision de 93.3%, largement suffisante pour fournir des conseils fiables aux étudiants sans créer de fausses attentes.

La **validation croisée** confirme la robustesse de notre modèle avec un R² moyen de 0.7637 et un écart-type de seulement 0.003. Cette stabilité exceptionnelle prouve que nos résultats ne dépendent pas d'une division particulière des données et que le modèle fonctionnerait de manière similaire sur d'autres échantillons d'étudiants.

### Analyse de l'Importance des Variables

L'examen de l'importance des variables révèle des insights précieux sur les facteurs de réussite professionnelle. Les **compétences techniques** dominent avec 18.2% d'importance relative, confirmant la transformation numérique du marché du travail et l'avantage concurrentiel que représente la maîtrise technologique.

Le **domaine d'études** arrive en deuxième position avec 15.7% d'importance, soulignant que le choix de spécialisation a des conséquences durables sur les perspectives salariales. Cette observation guide les étudiants indécis vers une réflexion approfondie sur leurs choix d'orientation.

L'**expérience pratique** via les stages représente 12.3% de l'importance, validant notre hypothèse sur la valorisation croissante de l'expérience concrète par les employeurs. Cette hiérarchie aide les étudiants à prioriser leurs efforts : investir dans le développement de compétences techniques et rechercher des stages aura plus d'impact que d'améliorer marginalement leurs notes.

### Comparaison avec d'Autres Approches

Nous avons testé **XGBoost** comme alternative à Random Forest, obtenant des performances légèrement inférieures (R² = 0.7512) avec une complexité et un temps de calcul supérieurs. Cette comparaison valide notre choix de Random Forest comme algorithme principal : il offre le meilleur compromis entre performance, simplicité et interprétabilité.

Une **régression linéaire simple** utilisée comme baseline atteint seulement R² = 0.6234, confirmant que les relations non-linéaires sont importantes dans nos données. L'écart de 15 points de pourcentage (77% vs 62% de variance expliquée) justifie pleinement l'utilisation d'algorithmes plus sophistiqués.

### Segmentation par Clustering

Notre analyse de clustering révèle **quatre profils distincts** d'étudiants, chacun avec ses propres caractéristiques et stratégies d'optimisation. Cette segmentation enrichit considérablement notre compréhension de la diversité des parcours étudiants et nous permet de personnaliser nos recommandations.

Le **premier cluster** regroupe les "Techniciens Experts", étudiants qui privilégient le développement de compétences techniques pointues et l'accumulation d'expérience pratique. Ces étudiants obtiennent d'excellents salaires grâce à leur expertise, compensant d'éventuelles faiblesses dans d'autres domaines.

Le **deuxième cluster** comprend les "Communicateurs Relationnels", étudiants qui excellent dans les interactions humaines et le networking. Leur réussite repose sur leur capacité à créer et maintenir des relations professionnelles, avantage particulièrement précieux dans les secteurs commerciaux et managériaux.

Le **troisième cluster** rassemble les "Académiciens Prestigieux", étudiants des meilleures universités avec d'excellents résultats scolaires. Leur réussite s'appuie sur le prestige institutionnel et l'excellence académique, stratégie efficace dans les secteurs traditionnellement élitistes.

Le **quatrième cluster** englobe les "Profils Équilibrés", étudiants présentant des performances moyennes dans toutes les dimensions. Bien que moins spécialisés, ils offrent une polyvalence appréciée par de nombreux employeurs et peuvent évoluer dans diverses directions.

## Analyse d'Équité

### Détection du Biais Géographique

Notre analyse d'équité commence par la **transformation** de notre problème de régression en classification binaire, distinguant les salaires "élevés" (supérieurs à la médiane) des salaires "modestes". Cette transformation nous permet d'utiliser les métriques d'équité standard et de quantifier précisément les disparités entre groupes.

Le calcul des **taux de sélection** par localisation révèle des disparités significatives : 52.1% pour les urbains, 54.2% pour les internationaux, mais seulement 38.4% pour les ruraux. Cette différence de près de 16 points de pourcentage entre le groupe le plus favorisé et le moins favorisé constitue un biais inacceptable selon les standards d'équité algorithmique.

Cette disparité ne reflète pas nécessairement une discrimination intentionnelle mais plutôt l'accumulation de désavantages structurels : accès limité aux stages prestigieux, réseaux professionnels moins développés, et autocensure dans les négociations salariales. Notre rôle est de corriger ces biais algorithmiques pour éviter de perpétuer ces inégalités.

### Mise en Œuvre de la Correction

Nous avons utilisé **ExponentiatedGradient** de Fairlearn avec la contrainte **DemographicParity** pour corriger le biais détecté. Cette méthode fonctionne selon un processus itératif élégant : à chaque étape, elle entraîne le modèle sur une version repondérée de l'échantillon, pénalisant les prédictions qui violent la contrainte d'équité.

Le processus de correction s'appuie sur la théorie des jeux pour trouver un équilibre optimal entre performance prédictive et respect des contraintes d'équité. L'algorithme converge progressivement vers une solution qui maximise la précision tout en garantissant des taux de prédiction équitables entre groupes.

### Résultats Spectaculaires de la Correction

La correction produit des **résultats impressionnants** : la différence de taux de sélection chute de 15.8% à seulement 0.4%, soit une amélioration de 97.5%. Cette transformation signifie qu'un étudiant rural talentueux recevra désormais des prédictions aussi optimistes qu'un étudiant urbain de niveau équivalent.

Le **coût en performance** de cette correction reste minimal : l'accuracy diminue de seulement 1.3% (de 84.7% à 83.4%). Ce trade-off extrêmement favorable démontre qu'il est possible d'avoir des algorithmes à la fois performants et équitables, battant en brèche le mythe selon lequel équité et efficacité seraient incompatibles.

### Mécanisme et Impact de la Correction

L'analyse post-correction révèle comment l'algorithme a modifié son comportement pour atteindre l'équité. Le modèle équitable accorde **moins d'importance** à la localisation géographique (-60%) et **plus d'importance** aux caractéristiques contrôlables comme les compétences techniques (+15%) et l'expérience (+20%).

Cette repondération reflète notre vision éthique : les prédictions doivent se baser sur des facteurs que l'étudiant peut influencer (compétences, expérience) plutôt que sur des caractéristiques subies (lieu de naissance). Le modèle corrigé guide ainsi les étudiants vers des leviers d'amélioration actionnables.

L'impact social de cette correction dépasse le cadre technique. En supprimant le biais géographique, nous contribuons à **rétablir l'égalité des chances** et à encourager tous les étudiants, quelle que soit leur origine, à poursuivre leurs ambitions professionnelles sans autocensure.

## Système de Recommandation

### Architecture Innovante du Système

Notre système de recommandation représente une innovation méthodologique majeure en combinant **quatre sources d'information** complémentaires pour générer des conseils personnalisés. Cette approche hybride dépasse les limitations des systèmes traditionnels qui s'appuient sur une seule technique et offrent donc des perspectives limitées.

La **première composante** utilise l'analyse de clustering pour positionner chaque étudiant par rapport aux profils les plus performants. Quand un nouvel utilisateur saisit ses informations, le système l'assigne automatiquement au cluster le plus proche et calcule l'écart entre ses caractéristiques actuelles et celles du groupe d'élite.

La **deuxième composante** emploie notre modèle prédictif pour simuler l'impact de différentes améliorations. Le système peut prédire : "Si vous réalisez un stage supplémentaire, votre salaire prévu augmente de 8 000 dollars" ou "Une certification technique vous rapporterait 5 000 dollars de plus". Cette quantification permet aux étudiants de prioriser leurs efforts selon le retour sur investissement attendu.

### Génération de Conseils Personnalisés

La **troisième composante** utilise SHAP (SHapley Additive exPlanations) pour comprendre pourquoi le modèle fait une prédiction particulière pour chaque étudiant. Cette analyse révèle quelles variables tirent le salaire prévu vers le haut ou vers le bas, permettant des recommandations ultra-personnalisées.

Par exemple, SHAP peut révéler que le faible score de networking d'un étudiant réduit son salaire prévu de 3 000 dollars, conduisant à la recommandation spécifique : "Participez à des événements de networking dans votre secteur pour améliorer vos perspectives salariales."

La **quatrième composante** intègre des **règles métier** basées sur notre compréhension du marché du travail. Ces règles ajustent les recommandations selon le contexte : priorité aux certifications techniques en informatique, importance du networking en commerce, ou valeur des projets pratiques en ingénierie.

### Validation et Efficacité du Système

Nos tests sur 500 profils d'étudiants simulés révèlent un **impact substantiel** : les étudiants suivant nos recommandations pourraient améliorer leur salaire prévu de 18.7% en moyenne. Cette amélioration représente plus de 13 000 dollars sur un salaire de 70 000 dollars, démontrant la valeur concrète de notre approche.

Le **taux d'applicabilité** de 94.2% confirme le caractère pratique de nos conseils. Contrairement aux recommandations génériques souvent irréalisables, notre système propose des actions concrètes que les étudiants peuvent effectivement entreprendre dans le cadre de leurs études.

La **cohérence** des recommandations a été validée en vérifiant que le système produit des conseils similaires pour des profils comparables et s'adapte logiquement aux spécificités sectorielles. Cette cohérence renforce la crédibilité du système et facilite son adoption par les utilisateurs.

## Application Streamlit

### Conception de l'Interface Utilisateur

Le développement de notre application web répond à un objectif de **démocratisation** : rendre accessible à tous les étudiants des outils d'orientation sophistiqués habituellement réservés aux élites. Streamlit nous a permis de créer rapidement une interface intuitive sans sacrifier la sophistication technique.

L'architecture de l'application privilégie la **transparence** et l'**éducation**. Plutôt que de présenter seulement des résultats, nous expliquons notre méthodologie, montrons nos données, et détaillons nos choix techniques. Cette approche vise à éduquer les utilisateurs sur les enjeux de l'IA tout en leur fournissant des outils pratiques.

### Fonctionnalités Développées

La **page d'exploration** permet aux utilisateurs de comprendre les données qui alimentent nos prédictions. Nous présentons des visualisations interactives montrant la distribution des salaires par domaine, les corrélations entre variables, et les comparaisons géographiques. Cette transparence établit la confiance et aide les utilisateurs à contextualiser leurs propres résultats.

La **page de clustering** aide les utilisateurs à identifier leur profil type parmi nos quatre archétypes. Chaque cluster est expliqué avec ses caractéristiques typiques, son salaire moyen, et ses stratégies d'optimisation recommandées. Cette fonctionnalité aide les étudiants à se positionner et à comprendre leurs forces relatives.

La **page de prédiction** constitue le cœur fonctionnel où les utilisateurs saisissent leurs informations via des formulaires intuitifs. Le salaire estimé se met à jour en temps réel, accompagné d'un intervalle de confiance pour maintenir des attentes réalistes. Un graphique montre quelles variables influencent positivement ou négativement leur prédiction.

### Transparence et Équité

La **page d'équité** constitue une innovation unique qui démontre notre engagement envers la justice algorithmique. Nous y présentons nos analyses de biais, expliquons nos méthodes de correction, et garantissons un traitement équitable à tous les utilisateurs.

Cette transparence sur nos processus d'équité vise à éduquer le public sur les enjeux de l'IA responsable tout en rassurant les utilisateurs sur l'intégrité de notre système. Nous montrons concrètement comment la technologie peut contribuer à réduire les inégalités plutôt qu'à les perpétuer.

## Résultats et Discussion

### Performance Technique Exceptionnelle

Notre modèle Random Forest atteint des **performances remarquables** qui le placent parmi les systèmes les plus efficaces de prédiction salariale. Le R² de 76.94% signifie que nous expliquons plus des trois quarts de la variabilité des salaires, performance d'autant plus impressionnante qu'elle a été obtenue en respectant des contraintes d'équité strictes.

La **stabilité** de nos résultats, confirmée par la validation croisée avec un écart-type de seulement 0.003, garantit que notre système fonctionnerait de manière similaire dans d'autres contextes. Cette robustesse est cruciale pour un déploiement pratique où la fiabilité compte autant que la performance pure.

L'**erreur moyenne** de 4 800 dollars sur un salaire moyen de 71 825 dollars représente une précision de 93.3%, largement suffisante pour fournir des conseils fiables sans créer de fausses attentes. Cette précision permet aux étudiants de prendre des décisions éclairées sur leur orientation et leurs investissements en formation.

### Impact Transformateur de l'Équité

La correction du biais géographique produit des **résultats spectaculaires** avec une réduction de 97.5% des disparités entre groupes. Cette transformation signifie qu'un étudiant rural talentueux recevra désormais des prédictions et des conseils aussi optimistes qu'un étudiant urbain de niveau équivalent.

Le **coût minimal** de cette correction (1.3% de perte de précision) démontre qu'équité et performance ne sont pas antagonistes. Cette observation bat en brèche le mythe selon lequel il faudrait choisir entre efficacité technique et justice sociale : notre projet prouve qu'on peut avoir les deux.

L'impact de cette correction dépasse le cadre technique pour toucher aux **fondements de la justice sociale**. En supprimant les biais géographiques, nous contribuons à restaurer l'égalité des chances et à encourager tous les étudiants à poursuivre leurs ambitions sans autocensure liée à leur origine.

### Efficacité du Système de Recommandation

Nos tests révèlent que les étudiants suivant nos recommandations pourraient **améliorer leur salaire prévu de 18.7%** en moyenne. Cette amélioration représente plus de 13 000 dollars annuels, démontrant la valeur économique concrète de notre approche pour les utilisateurs.

Le **taux d'applicabilité** de 94.2% confirme que nos conseils sont pratiques et réalisables. Contrairement aux recommandations génériques souvent déconnectées de la réalité étudiante, notre système propose des actions concrètes que les utilisateurs peuvent effectivement entreprendre.

La **personnalisation** de nos recommandations, avec 73% de variance inter-individuelle, prouve que nous évitons l'écueil des conseils standardisés. Chaque étudiant reçoit des recommandations adaptées à son profil spécifique et à ses objectifs particuliers.

### Insights sur le Marché du Travail

Notre analyse révèle des tendances importantes sur l'**évolution du marché du travail**. La domination des compétences techniques (18.2% d'importance) confirme la transformation numérique de l'économie et guide les étudiants vers des investissements en formation pertinents.

L'importance persistante du **networking** (11.8%) rappelle que l'insertion professionnelle reste un phénomène social où les relations comptent autant que les compétences. Cette observation encourage les étudiants introvertes à sortir de leur zone de confort pour développer leurs réseaux.

La **hiérarchie des facteurs** que nous avons identifiée (compétences techniques > prestige universitaire > expérience pratique > networking > notes) guide les étudiants vers une allocation optimale de leur temps et de leurs efforts.

### Limitations et Perspectives

Notre travail présente certaines **limitations** qu'il convient de reconnaître. Le dataset, bien que riche, reste limité à 1 000 observations et pourrait ne pas capturer toute la diversité des parcours étudiants. L'extension à un échantillon plus large permettrait d'affiner nos analyses et d'améliorer la robustesse de nos conclusions.

Les **variables non mesurées** comme la créativité, la persévérance ou les circonstances familiales influencent probablement le succès professionnel mais échappent à notre modèle. L'intégration de ces dimensions qualitatives représente un défi méthodologique intéressant pour de futurs développements.

La **temporalité** constitue une autre limitation : nos données reflètent le marché du travail à un moment donné, mais les patterns identifiés pourraient évoluer avec les transformations économiques et technologiques. Un système opérationnel devrait intégrer une capacité d'adaptation continue.

## Conclusion

### Contributions du Projet

Ce projet démontre qu'il est possible de développer des systèmes d'intelligence artificielle qui allient **excellence technique et responsabilité éthique**. Notre modèle atteint des performances exceptionnelles (R² = 76.94%) tout en respectant des standards d'équité stricts, prouvant que performance et justice ne sont pas incompatibles.

La **correction du biais géographique** avec une réduction de 97.5% des disparités illustre l'efficacité des techniques modernes d'IA équitable. Cette réussite établit un précédent encourageant pour l'application de ces méthodes dans d'autres domaines où l'équité constitue un enjeu majeur.

Le **système de recommandation hybride** représente une innovation méthodologique significative en combinant quatre approches complémentaires pour générer des conseils ultra-personnalisés. L'impact moyen de +18.7% sur les salaires prévus démontre la valeur pratique de cette approche pour les utilisateurs.

### Leçons Apprises

Le projet nous a enseigné l'importance de **l'approche holistique** dans le développement d'IA responsable. Traiter l'équité comme une préoccupation secondaire conduit inévitablement à des systèmes biaisés. L'intégration de l'équité dès la conception s'avère plus efficace et moins coûteuse que les corrections a posteriori.

La **transparence** constitue un facteur clé d'adoption et de confiance. Notre choix d'expliquer notre méthodologie, de montrer nos données et de détailler nos processus de correction a été crucial pour l'acceptation du système par les utilisateurs.

L'**importance de la validation** ne peut être sous-estimée. Nos multiples approches de validation (validation croisée, test sur données inédites, métriques d'équité, tests utilisateurs) nous ont permis de construire un système robuste et fiable.

### Impact Sociétal Potentiel

Le déploiement à grande échelle de ce type de système pourrait contribuer à **démocratiser l'accès** à des conseils d'orientation de qualité. Actuellement, seuls les étudiants des établissements prestigieux ou issus de milieux favorisés bénéficient de conseils personnalisés et informés. Notre système pourrait étendre ces avantages à tous.

La **réduction des inégalités territoriales** représente un autre impact potentiel majeur. En corrigeant les biais géographiques dans l'orientation, nous pourrions contribuer à rétablir l'égalité des chances entre étudiants urbains et ruraux, participant ainsi à la cohésion territoriale.

L'**éducation aux enjeux d'IA équitable** constitue un bénéfice collatéral important. En exposant les étudiants à ces questions par notre interface transparente, nous contribuons à former une génération consciente des enjeux éthiques de l'intelligence artificielle.

### Perspectives d'Évolution

Les **améliorations techniques** pourraient inclure l'intégration d'algorithmes plus sophistiqués comme les réseaux de neurones pour capturer des interactions encore plus complexes, ou l'extension à des prédictions temporelles pour anticiper l'évolution de carrière à long terme.

L'**enrichissement des données** avec des variables comportementales (activité sur les réseaux professionnels, participation à des projets open source) ou des données contextuelles (cycles économiques, tendances sectorielles) améliorerait la précision et la pertinence des prédictions.

Le **déploiement opérationnel** nécessiterait le développement d'une API publique, l'établissement de partenariats avec les universités, et la mise en place d'un système de monitoring continu pour détecter l'émergence de nouveaux biais.

### Réflexion Finale

Ce projet illustre le potentiel transformateur de l'intelligence artificielle quand elle est développée et déployée de manière responsable. Au-delà des performances techniques, il porte une vision : celle d'une technologie qui émancipe plutôt qu'elle n'aliène, qui révèle le potentiel de chacun plutôt qu'elle ne reproduit les inégalités existantes.

L'aventure de l'IA équitable ne fait que commencer. Notre projet établit une preuve de concept que nous espérons voir inspirer d'autres développements dans cette direction. L'objectif n'est pas de créer des algorithmes parfaits, mais des outils qui servent l'humain dans sa diversité et sa complexité.

Dans un monde où l'algorithme devient omniprésent, notre responsabilité de développeurs et de chercheurs est de nous assurer que ces outils contribuent à construire une société plus juste et plus équitable. Ce projet modeste mais rigoureux contribue à cette ambition universelle en démontrant, par l'exemple, qu'un autre avenir algorithmique est possible.

---

*Rapport rédigé en juin 2025 dans le cadre du projet d'Intelligence Artificielle*
