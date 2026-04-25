import argparse

from app.pipeline.run import run_pipeline


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline de prediction de toxicite des champignons"
    )
    parser.add_argument(
        "--train_data", type=str, default="data/X_train.csv", help="Chemin vers X_train.csv"
    )
    parser.add_argument(
        "--train_labels", type=str, default="data/y_train.csv", help="Chemin vers y_train.csv"
    )
    parser.add_argument(
        "--test_data", type=str, default="data/X_test.csv", help="Chemin vers X_test.csv"
    )
    parser.add_argument(
        "--train_images",
        type=str,
        default="images/train",
        help="Dossier des images d'entrainement",
    )
    parser.add_argument(
        "--test_images",
        type=str,
        default="images/test",
        help="Dossier des images de test",
    )
    parser.add_argument("--test_mode", action="store_true", help="Mode test sur 10 echantillons")
    return parser


def main():
    args = build_parser().parse_args()
    run_pipeline(args)
