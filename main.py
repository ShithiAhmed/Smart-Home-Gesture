


## import the handfeature extractor class

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
import os
import cv2
import numpy as np
from frameextractor import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor

# Define paths
video_dir = "traindata/Videos_[HusnaAhmedShithi]/"
frames_dir = "traindata/frames"

# Initialize counter
count = 0

# Make sure the output folder exists
for filename in os.listdir(video_dir):
    if filename.startswith('.'):  # skip hidden/trash files
        continue

    videopath = os.path.join(video_dir, filename)
    print("Processing", videopath)

    try:
        frameExtractor(videopath, frames_dir, count)
    except Exception as e:
        print(f"Skipping {filename}, error: {e}")
    count += 1
print("\n✅ Frame extraction complete. Now extracting features...\n")

# --- STEP 2: Extract features using CNN model ---
feature_extractor = HandShapeFeatureExtractor.get_instance()

features = []
labels = []

for filename in os.listdir(frames_dir):
    if not filename.endswith(".png"):
        continue

    img_path = os.path.join(frames_dir, filename)
    label = filename.split("_")[0]  # e.g. LightOn_1.png → "LightOn"

    try:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feature = feature_extractor.extract_feature(image)
        features.append(feature)
        labels.append(label)
    except Exception as e:
        print(f"Skipping {filename}, error: {e}")

# --- STEP 3: Save features and labels as .npy files ---
features = np.array(features).squeeze()
labels = np.array(labels)

np.save("train_data_features.npy", features)
np.save("train_data_labels.npy", labels)

print("✅ Feature extraction complete.")
print("Saved: train_data_features.npy and train_data_labels.npy")


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

print("▶️ Starting SVM training...")

# Load features and labels
features = np.load("train_data_features.npy")
labels = np.load("train_data_labels.npy")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train SVM
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ SVM training complete. Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(classifier, "svm_model.pkl")
print("📦 Saved model: svm_model.pkl")

