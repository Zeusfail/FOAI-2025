import os
import time
import warnings

import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from app.config import (
    BATCH_SIZE,
    DEVICE,
    MODELS_DIR,
    NUM_EPOCHS,
    OUTPUT_DIR,
    SEED,
    TEST_SAMPLE_SIZE,
    VAL_SIZE,
    ensure_output_dirs,
)
from app.data.dataset import ChampignonDataset
from app.data.preprocessing import ChampignonPreprocessor
from app.eval.inference import evaluate_model
from app.eval.plots import plot_training_history
from app.features.engineering import create_ratio_features
from app.model.losses import FocalLoss
from app.model.multimodal import MultimodalToxicityModel
from app.training.engine import train_model
from app.training.transforms import create_data_augmentation
from app.utils.reproducibility import set_seed
from app.utils.validation import check_data_availability

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def run_pipeline(args):
    ensure_output_dirs()

    print(f"Execution sur: {DEVICE}")

    check_data_availability(args)

    set_seed(SEED)
    print(f"Seed fixe a {SEED} pour reproductibilite")

    start_time = time.time()

    print("\nChargement des donnees...")
    X_train = pd.read_csv(args.train_data)
    y_train = pd.read_csv(args.train_labels)
    X_test = pd.read_csv(args.test_data)

    if args.test_mode:
        print(f"Mode test active: utilisation de {TEST_SAMPLE_SIZE} echantillons")
        np.random.seed(SEED)
        test_indices = np.random.choice(len(X_test), TEST_SAMPLE_SIZE, replace=False)
        X_test = X_test.iloc[test_indices].reset_index(drop=True)

    print("Creation de features derivees...")
    X_train = create_ratio_features(X_train)
    X_test = create_ratio_features(X_test)

    print("Pretraitement des donnees tabulaires...")
    preprocessor = ChampignonPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train.drop("id", axis=1))
    _ = preprocessor.transform(X_test.drop("id", axis=1))

    preprocessor.save(os.path.join(MODELS_DIR, "preprocessor.pkl"))

    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y_train["est_toxique"],
    )

    _ = preprocessor.transform(X_val_df.drop("id", axis=1))

    train_transform, test_transform = create_data_augmentation()

    train_dataset = ChampignonDataset(
        X_train_df,
        args.train_images,
        y_train_df,
        transform=train_transform,
        preprocessor=preprocessor,
    )

    val_dataset = ChampignonDataset(
        X_val_df,
        args.train_images,
        y_val_df,
        transform=test_transform,
        preprocessor=preprocessor,
    )

    test_dataset = ChampignonDataset(
        X_test,
        args.test_images,
        transform=test_transform,
        is_test=True,
        preprocessor=preprocessor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    tabular_size = X_train_processed.shape[1]

    print("\nCreation du modele multimodal...")
    model = MultimodalToxicityModel(tabular_input_size=tabular_size).to(DEVICE)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    total_steps = NUM_EPOCHS * len(train_loader)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    print("\nDebut de l'entrainement...")
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        NUM_EPOCHS,
        DEVICE,
        scheduler,
        early_stopping_patience=7,
    )

    plot_training_history(history)

    print("\nGeneration des predictions finales...")
    predictions = evaluate_model(trained_model, test_loader, DEVICE)

    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    predictions.to_csv(submission_path, index=False)
    print(f"Predictions sauvegardees dans {submission_path}")

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\nTemps d'execution total: {int(minutes)}m {int(seconds)}s")

    print("\nApercu des predictions:")
    print(predictions.head(10))

    return predictions
