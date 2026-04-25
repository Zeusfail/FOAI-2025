import os


def check_data_availability(args) -> bool:
    missing_elements = []

    required_files = [
        (args.train_data, "Donnees d'entrainement (X_train.csv)"),
        (args.train_labels, "Etiquettes d'entrainement (y_train.csv)"),
        (args.test_data, "Donnees de test (X_test.csv)"),
    ]

    for file_path, description in required_files:
        if not os.path.isfile(file_path):
            missing_elements.append(f"{description} : {file_path}")

    required_dirs = [
        (args.train_images, "Dossier d'images d'entrainement"),
        (args.test_images, "Dossier d'images de test"),
    ]

    for dir_path, description in required_dirs:
        if not os.path.isdir(dir_path):
            missing_elements.append(f"{description} : {dir_path}")
        else:
            image_files = [
                f for f in os.listdir(dir_path) if f.endswith((".png", ".jpg", ".jpeg"))
            ]
            if not image_files:
                missing_elements.append(f"{description} ne contient aucune image")

    if missing_elements:
        error_message = "Elements manquants detectes:\n" + "\n".join(missing_elements)
        error_message += "\n\nVeuillez verifier la structure des dossiers et les chemins specifies."
        raise FileNotFoundError(error_message)

    print("Tous les fichiers et dossiers necessaires sont presents")
    return True
