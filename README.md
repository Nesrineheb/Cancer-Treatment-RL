# RL Cancer Treatment


## Description

Le projet RL Cancer Treatment vise à développer un système d'apprentissage par renforcement (RL) pour aider à la prise de décisions de traitement dans le contexte de la croissance tumorale. Il utilise l'apprentissage automatique pour optimiser les décisions de traitement en modélisant la croissance tumorale et en prenant des décisions basées sur cette modélisation.

## Objectifs

Les principaux objectifs du projet sont les suivants :

1. Créer un environnement de simulation, CancerGrowthEnv, pour modéliser la croissance tumorale.
2. Implémenter un agent RandomAgent pour prendre des décisions de traitement.
3. Implémenter un agent DQN (Deep Q-Network) pour prendre des décisions de traitement.
4. Personnaliser l'agent DQN pour améliorer ses performances dans l'environnement.
5. Étendre le projet à un contexte multi-agent pour mieux modéliser la situation médicale réelle.
6. Personnaliser l'agent DQN vers une architecture double dans le contexte multi-agent pour améliorer les performances.

## Table des matières

- [Installation](#installation)
- [Structure du Projet](#structure-du-projet)
- [Contribuer](#contribuer)
- [Licence](#licence)

## Installation

Pour utiliser ce projet, suivez les étapes ci-dessous :


1. Créer un environement python :
# Mon Environnement Python Personnalisé (mon_env)

Ceci est un environnement Python personnalisé que j'ai créé pour effectuer des tâches spécifiques. Vous pouvez suivre ces instructions pour configurer l'environnement sur votre système.

#Prérequis

Avant de configurer et d'utiliser l'environnement `mon_env`, assurez-vous d'avoir les éléments suivants installés sur votre système :

- [Python](https://www.python.org/downloads/) (version 3.6 ou supérieure)
- [pip](https://pip.pypa.io/en/stable/installing/) (gestionnaire de packages Python)
  
2. Clonez ce dépôt :

bash
git clone https://github.com/votre_utilisateur/rl-cancer-treatment.git
cd rl-cancer-treatment

3. Installez les dépendances requises à l'aide de `pip` en utilisant le fichier `requirements.txt`:
   pip install -r requirements.txt
   
 ## Structure du Projet
 # Cancer Growth Simulation

## Classes du Projet

### 1. CancerGrowthEnv

La classe `CancerGrowthEnv` est une implémentation d'un environnement Gym personnalisé. Elle modélise la croissance d'une tumeur cancéreuse et permet aux agents d'agir en choisissant de traiter ou de ne pas traiter la tumeur. Voici quelques détails importants sur cette classe :

- **`__init__(self, num_agents=2)`** : Le constructeur de la classe permet de spécifier le nombre d'agents (par défaut à 2). Il définit également les espaces d'observation et d'action pour les agents.

- **`step(self, actions)`** : Cette méthode permet aux agents de prendre des actions en fonction des actions fournies en entrée. Elle calcule les récompenses pour chaque agent et définit si l'épisode est terminé.

- **`reset(self, verbose=True)`** : Cette méthode réinitialise l'état de l'environnement pour chaque agent. Si `verbose` est défini sur True, elle affiche un message de réinitialisation.

### 2. DQNAgent

La classe `DQNAgent` est un agent d'apprentissage profond par renforcement qui utilise un réseau de neurones pour prendre des décisions dans l'environnement `CancerGrowthEnv`. Voici quelques détails importants sur cette classe :

- **`__init__(self, state_size, action_size, verbose=0)`** : Le constructeur de la classe permet de spécifier la taille de l'état et de l'action, ainsi qu'un paramètre optionnel `verbose` pour le niveau de verbosité.

- **`_build_model(self)`** : Cette méthode construit un modèle de réseau de neurones utilisé par l'agent pour prendre des décisions.

- **`remember(self, state, action, reward, next_state, done)`** : Cette méthode permet à l'agent de mémoriser l'expérience passée dans son buffer de mémoire.

- **`act(self, state)`** : Cette méthode permet à l'agent de choisir une action en fonction de l'état actuel. Elle peut prendre en compte l'exploration en utilisant epsilon-greedy.

- **`replay(self)`** : Cette méthode permet à l'agent d'apprendre à partir de ses expériences passées en effectuant une mise à jour du modèle.

### 3. Train

La classe `Train` est responsable de l'entraînement des agents dans l'environnement `CancerGrowthEnv`. Elle coordonne les interactions entre les agents et l'environnement, gère les récompenses et les mises à jour des modèles. Voici quelques détails importants sur cette classe :

- **`__init__(self, env, agents, num_episodes)`** : Le constructeur de la classe prend l'environnement, les agents et le nombre d'épisodes comme paramètres.

- **`train(self)`** : Cette méthode exécute l'entraînement des agents sur un certain nombre d'épisodes. Elle gère également la sauvegarde du modèle pré-entraîné et des récompenses.

### 4. main

Le script principal `main.py` entraîne deux agents DQN dans l'environnement `CancerGrowthEnv` pour gérer la croissance tumorale.


## Contribuer

Les contributions à ce projet sont les bienvenues ! Pour contribuer, suivez ces étapes :

1. Forkez ce référentiel (repository) sur GitHub.
2. Créez une branche (branch) pour votre contribution : `git checkout -b ma-contribution`
3. Faites vos modifications et committez-les : `git commit -m 'Ajout de nouvelles fonctionnalités'`
4. Poussez vos modifications sur votre fork : `git push origin ma-contribution`
5. Créez une Pull Request sur GitHub à partir de votre fork.

Nous examinerons vos contributions avec attention et les intégrerons au projet si elles sont pertinentes.


## Licence

Ce projet est sous licence MIT. Vous pouvez consulter le fichier [LICENSE](LICENSE) pour plus de détails sur les conditions de licence.






