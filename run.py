import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog, local_binary_pattern

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = "dataset/Training"
TEST_DIR = "dataset/Test"

MODEL_FILE = "fruit_model.pkl"
LABEL_FILE = "label_encoder.pkl"

IMAGE_SIZE = (100, 100)

# Fruits that we want our model to recognize
TARGET_FRUITS = [
    "Apple",
    "Banana",
    "Orange",
    "Pineapple",
    "Strawberry"
]

# Number of images used from each class.
# Use None to use every image.
MAX_IMAGES_PER_CLASS = 500


# ============================================================
# FIND FRUIT FOLDER
# ============================================================

def find_fruit_folders(dataset_path):

    all_folders = sorted([
        folder
        for folder in os.listdir(dataset_path)
        if os.path.isdir(
            os.path.join(dataset_path, folder)
        )
    ])

    selected_folders = []

    print("\nSearching for requested fruits...")

    for fruit in TARGET_FRUITS:

        matches = [
            folder
            for folder in all_folders
            if folder.lower().startswith(
                fruit.lower()
            )
        ]

        if len(matches) == 0:

            print(
                f"WARNING: {fruit} was not found."
            )

        else:

            # Use the first matching folder
            selected_folder = matches[0]

            selected_folders.append(
                selected_folder
            )

            print(
                f"{fruit} -> {selected_folder}"
            )

    return selected_folders


# ============================================================
# HOG FEATURE EXTRACTION
# ============================================================

def extract_hog(image):

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# ============================================================
# LBP FEATURE EXTRACTION
# ============================================================

def extract_lbp(image):

    radius = 1
    points = 8 * radius

    lbp = local_binary_pattern(
        image,
        points,
        radius,
        method="uniform"
    )

    n_bins = points + 2

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    histogram = histogram.astype(
        "float32"
    )

    histogram /= (
        histogram.sum() + 1e-7
    )

    return histogram


# ============================================================
# COMBINE HOG + LBP
# ============================================================

def extract_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        IMAGE_SIZE
    )

    hog_features = extract_hog(
        gray
    )

    lbp_features = extract_lbp(
        gray
    )

    combined_features = np.concatenate(
        [
            hog_features,
            lbp_features
        ]
    )

    return combined_features


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    dataset_path,
    selected_folders
):

    X = []
    y = []

    print("\n================================")
    print(
        f"Loading: {dataset_path}"
    )
    print("================================")

    for folder in selected_folders:

        class_path = os.path.join(
            dataset_path,
            folder
        )

        image_files = [
            file
            for file in os.listdir(
                class_path
            )
            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            )
        ]

        image_files = sorted(
            image_files
        )

        if MAX_IMAGES_PER_CLASS is not None:

            image_files = image_files[
                :MAX_IMAGES_PER_CLASS
            ]

        print(
            f"Loading {folder}: "
            f"{len(image_files)} images"
        )

        for image_file in image_files:

            image_path = os.path.join(
                class_path,
                image_file
            )

            image = cv2.imread(
                image_path
            )

            if image is None:

                print(
                    "Could not read:",
                    image_path
                )

                continue

            try:

                features = extract_features(
                    image
                )

                X.append(
                    features
                )

                # Use the fruit name rather than
                # the dataset's exact variety name
                fruit_name = get_fruit_name(
                    folder
                )

                y.append(
                    fruit_name
                )

            except Exception as error:

                print(
                    f"Error processing "
                    f"{image_path}: {error}"
                )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# CONVERT DATASET FOLDER TO FRUIT NAME
# ============================================================

def get_fruit_name(folder_name):

    for fruit in TARGET_FRUITS:

        if folder_name.lower().startswith(
            fruit.lower()
        ):

            return fruit

    return folder_name


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\n================================")
    print("TRAINING MODEL")
    print("================================")

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Number of features:",
        X_train.shape[1]
    )

    svm = LinearSVC(
        max_iter=10000,
        random_state=42
    )

    parameter_grid = {
        "C": [
            0.01,
            0.1,
            1,
            10
        ]
    }

    print(
        "\nRunning GridSearchCV..."
    )

    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=parameter_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(
        X_train,
        y_train
    )

    print(
        "\nBest parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        "\nBest CV score:"
    )

    print(
        f"{grid_search.best_score_ * 100:.2f}%"
    )

    return (
        grid_search.best_estimator_
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    label_encoder
):

    print("\n================================")
    print("MODEL EVALUATION")
    print("================================")

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nTest Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    display.plot(
        ax=ax,
        xticks_rotation=45
    )

    plt.title(
        "Fruit Identifier Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()

    return accuracy


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    label_encoder
):

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        label_encoder,
        LABEL_FILE
    )

    print("\n================================")
    print("MODEL SAVED")
    print("================================")

    print(
        f"Model: {MODEL_FILE}"
    )

    print(
        f"Labels: {LABEL_FILE}"
    )


# ============================================================
# PREDICT A NEW IMAGE
# ============================================================

def predict_image(
    image_path
):

    print("\n================================")
    print("FRUIT PREDICTION")
    print("================================")

    if not os.path.exists(
        image_path
    ):

        print(
            "Image not found:"
        )

        print(
            image_path
        )

        return

    if not os.path.exists(
        MODEL_FILE
    ):

        print(
            "Model file does not exist."
        )

        print(
            "Train the model first."
        )

        return

    model = joblib.load(
        MODEL_FILE
    )

    label_encoder = joblib.load(
        LABEL_FILE
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "Could not read image."
        )

        return

    features = extract_features(
        image
    )

    features = features.reshape(
        1,
        -1
    )

    prediction = model.predict(
        features
    )

    predicted_number = prediction[0]

    predicted_fruit = (
        label_encoder.inverse_transform(
            [predicted_number]
        )[0]
    )

    print(
        "\n🍎 Predicted Fruit:"
    )

    print(
        predicted_fruit
    )

    # --------------------------------------------------------
    # Display image
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        image_rgb
    )

    plt.title(
        f"Prediction: {predicted_fruit}"
    )

    plt.axis(
        "off"
    )

    plt.show()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n================================")
    print("       FRUIT IDENTIFIER")
    print("================================")

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not os.path.exists(
        TRAIN_DIR
    ):

        print(
            "\nTraining directory not found:"
        )

        print(
            TRAIN_DIR
        )

        return

    if not os.path.exists(
        TEST_DIR
    ):

        print(
            "\nTest directory not found:"
        )

        print(
            TEST_DIR
        )

        return

    # --------------------------------------------------------
    # Find requested fruit folders
    # --------------------------------------------------------

    selected_folders = find_fruit_folders(
        TRAIN_DIR
    )

    if len(selected_folders) < 2:

        print(
            "\nNot enough fruit classes found."
        )

        return

    print(
        "\nSelected dataset folders:"
    )

    for folder in selected_folders:

        print(
            "  ",
            folder
        )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    X_train, y_train = load_dataset(
        TRAIN_DIR,
        selected_folders
    )

    if len(X_train) == 0:

        print(
            "\nNo training images found."
        )

        return

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test, y_test = load_dataset(
        TEST_DIR,
        selected_folders
    )

    if len(X_test) == 0:

        print(
            "\nNo test images found."
        )

        return

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    print("\n================================")
    print("ENCODING LABELS")
    print("================================")

    label_encoder = LabelEncoder()

    label_encoder.fit(
        TARGET_FRUITS
    )

    y_train_encoded = (
        label_encoder.transform(
            y_train
        )
    )

    y_test_encoded = (
        label_encoder.transform(
            y_test
        )
    )

    print(
        "\nFruit classes:"
    )

    for number, fruit in enumerate(
        label_encoder.classes_
    ):

        print(
            f"{number} -> {fruit}"
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train_encoded
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_test,
        y_test_encoded,
        label_encoder
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model,
        label_encoder
    )

    # --------------------------------------------------------
    # Predict custom image
    # --------------------------------------------------------

    print("\n================================")
    print("TEST YOUR OWN FRUIT IMAGE")
    print("================================")

    image_path = input(
        "\nEnter the full path of a fruit image "
        "(or press Enter to finish): "
    ).strip()

    if image_path:

        predict_image(
            image_path
        )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()