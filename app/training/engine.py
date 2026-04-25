import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from app.config import MODELS_DIR


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    scheduler=None,
    early_stopping_patience=5,
    early_stopping_delta=0.001,
):
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}
    best_val_auc = 0.0
    early_stopping_counter = 0
    early_stop = False

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        print(f"\nEpoque {epoch + 1}/{num_epochs}")

        model.train()
        train_loss = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        print(f"Taux d'apprentissage: {current_lr:.7f}")

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_idx, batch in progress_bar:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images, tabular)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            train_loss += loss.item() * images.size(0)
            progress_bar.set_description(f"Batch {batch_idx + 1}/{len(train_loader)}")
            progress_bar.set_postfix(loss=loss.item())

        train_loss = train_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                tabular = batch["tabular"].to(device)
                labels = batch["label"].to(device)

                tta_preds = []

                outputs = model(images, tabular)
                tta_preds.append(outputs)

                flipped_images = torch.flip(images, [3])
                outputs_flip = model(flipped_images, tabular)
                tta_preds.append(outputs_flip)

                outputs_center = model(images[:, :, 16:-16, 16:-16], tabular)
                tta_preds.append(outputs_center)

                outputs = torch.mean(torch.stack(tta_preds), dim=0)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                all_preds.extend(outputs.sigmoid().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(np.array(all_labels), np.array(all_preds))

        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc + early_stopping_delta:
            best_val_auc = val_auc
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pt"))
            print(f"Meilleur modele sauvegarde (AUC: {val_auc:.4f})")
        else:
            early_stopping_counter += 1
            print(f"EarlyStopping: {early_stopping_counter}/{early_stopping_patience}")

            if early_stopping_counter >= early_stopping_patience:
                print(
                    "Early stopping active! "
                    f"Aucune amelioration depuis {early_stopping_patience} epoques."
                )
                early_stop = True
                break

    if early_stop:
        print(f"Chargement du meilleur modele (epoque {epoch + 1 - early_stopping_patience})")

    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pt")))
    return model, history
