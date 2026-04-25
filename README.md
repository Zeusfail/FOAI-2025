# Prediction de toxicite des champignons extraterrestres

Projet IA multimodal (images + donnees tabulaires) realise pour l'Olympiade Francaise IA 2025.

## Objectif

Predire la probabilite de toxicite d'un champignon a partir de:
- son image
- ses caracteristiques physico-chimiques

Le pipeline combine:
- une branche CNN (EfficientNet-B0) pour les images
- une branche MLP pour les donnees tabulaires
- une fusion avec mecanisme d'attention

## Nouvelle entree du projet

Le point d'entree officiel est maintenant:

```bash
python -m app
```

L'ancien lancement `python main.py` est desactive et affiche un message de redirection.

## Structure du projet

```text
.
|-- app/
|   |-- __main__.py
|   |-- cli.py
|   |-- config.py
|   |-- data/
|   |   |-- dataset.py
|   |   `-- preprocessing.py
|   |-- eval/
|   |   |-- inference.py
|   |   `-- plots.py
|   |-- features/
|   |   `-- engineering.py
|   |-- model/
|   |   |-- losses.py
|   |   `-- multimodal.py
|   |-- pipeline/
|   |   `-- run.py
|   |-- training/
|   |   |-- engine.py
|   |   `-- transforms.py
|   `-- utils/
|       |-- reproducibility.py
|       `-- validation.py
|-- data/
|   |-- X_train.csv
|   |-- y_train.csv
|   `-- X_test.csv
|-- images/
|   |-- train/
|   `-- test/
|-- output/                 # cree automatiquement
`-- main.py                 # script de redirection
```

## Installation

1. Creer un environnement Python (recommande: Python 3.10):

```bash
python -m venv venv
```

2. Activer l'environnement:

- Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

- Linux/macOS:

```bash
source venv/bin/activate
```

3. Installer les dependances:

```bash
pip install -r requirements.txt
```

## Utilisation

Execution standard:

```bash
python -m app
```

Mode test rapide (echantillon reduit):

```bash
python -m app --test_mode
```

Avec chemins personnalises:

```bash
python -m app --train_data data/X_train.csv --train_labels data/y_train.csv --test_data data/X_test.csv --train_images images/train --test_images images/test
```

Options CLI disponibles:
- `--train_data` (defaut: `data/X_train.csv`)
- `--train_labels` (defaut: `data/y_train.csv`)
- `--test_data` (defaut: `data/X_test.csv`)
- `--train_images` (defaut: `images/train`)
- `--test_images` (defaut: `images/test`)
- `--test_mode` (active un run rapide)

## Sorties generees

Le pipeline ecrit ses artefacts dans `output/`:
- `output/submission.csv` : predictions finales
- `output/models/best_model.pt` : meilleur checkpoint
- `output/models/preprocessor.pkl` : preprocesseur tabulaire
- `output/logs/enhanced_training_history.png` : courbes d'entrainement

## Notes techniques

- Reproductibilite geree via seed globale.
- Entrainement avec mixed precision quand GPU disponible.
- Arret anticipe (early stopping) et scheduler OneCycleLR.
