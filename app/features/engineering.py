import numpy as np
import pandas as pd


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    epsilon = 1e-10

    features["ratio_ph"] = df["ph_du_jus"] / (df["ph_du_sol"] + epsilon)
    features["poids_porosite"] = df["poids"] / (df["porosite"] + epsilon)
    features["distance_origine"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)

    return pd.concat([df, features], axis=1)
