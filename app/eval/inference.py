import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            ids = batch["id"].numpy()

            outputs = torch.sigmoid(model(images, tabular))

            all_preds.extend(outputs.cpu().numpy())
            all_ids.extend(ids)

    return pd.DataFrame(
        {
            "id": all_ids,
            "probabilite_toxique": np.concatenate(all_preds).flatten(),
        }
    )
