import os
import cv2
import numpy as np
import csv
from frameextractor import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# -------------------------------
# Step 1: Frame Extraction (Train)
# -------------------------------
video_dir = "traindata/Videos_[HusnaAhmedShithi]/"
frames_dir = "traindata/frames"
os.makedirs(frames_dir, exist_ok=True)

count = 0

print("\n✅ Frame extraction complete. Now extracting features...\n")

# -------------------------------
# Step 2: Feature Extraction (Train)
# -------------------------------
feature_extractor = HandShapeFeatureExtractor.get_instance()

features = []
labels = []

# label_map = {
#     "Num0": 0,
#     "Num1": 1,
#     "Num2": 2,
#     "Num3": 3,
#     "Num4": 4,
#     "Num5": 5,
#     "Num6": 6,
#     "Num7": 7,
#     "Num8": 8,
#     "Num9": 9,
#     "FanDown": 10,
#     "FanOff": 11,
#     "FanOn": 12,
#     "FanUp": 13,
#     "LightOff": 14,
#     "LightOn": 15,
#     "SetThermo": 16
    
# }

label_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "DecreaseFanSpeed": 10,
    "FanOff": 11,
    "FanOn": 12,
    "IncreaseFanSpeed": 13,
    "LightOff": 14,
    "LightOn": 15,
    "SetThermo": 16
    
}


for filename in os.listdir(video_dir):
    if filename.startswith('.') or not filename.endswith('.mp4'):
        continue

    videopath = os.path.join(video_dir, filename)
    print("Processing", videopath)

    try:
        frameExtractor(videopath, frames_dir, count)
        img_path = os.path.join(frames_dir, "%#05d.png" % (count+1))
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feature = feature_extractor.extract_feature(image)
        features.append(feature)
        
        label = filename.split("-")[-1] 
        label = label.replace(".mp4","")
        label_index = label_map[label]
        labels.append(label_index)
    except Exception as e:
        print(f"Skipping {filename}, error: {e}")
    count += 1



features = np.array(features).squeeze()
labels = np.array(labels)


np.save("train_data_features.npy", features)
np.save("train_data_labels.npy", labels)

print("✅ Feature extraction complete.")
print("Saved: train_data_features.npy and train_data_labels.npy")


def predict_test_set():
    test_video_dir = "test"
    test_frames_dir = "testdata_frames"
    os.makedirs(test_frames_dir, exist_ok=True)

    train_features = np.load("train_data_features.npy")
    train_labels = np.load("train_data_labels.npy")

    feature_extractor = HandShapeFeatureExtractor.get_instance()
    results = []

    count = 0
    for filename in sorted(os.listdir(test_video_dir)):
        if filename.startswith('.') or not filename.endswith('.mp4'):
            continue

        videopath = os.path.join(test_video_dir, filename)
        print("🔍 Processing test video:", videopath)

        try:
            frameExtractor(videopath, test_frames_dir, count)
            frame_path = os.path.join(test_frames_dir, "%#05d.png" % (count+1))

            if not os.path.exists(frame_path):
                print(f"⚠️ No frame found for {filename}, skipping.")
                count += 1
                continue

            image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            test_feature = feature_extractor.extract_feature(image)

            similarities = cosine_similarity(test_feature, train_features)[0]
            most_similar_index = np.argmax(similarities)
            predicted_label = train_labels[most_similar_index]

            print(f"✅ Predicted label for {filename}: {predicted_label}")
            results.append([filename, predicted_label])

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

        count += 1

    with open("Results.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["VideoName", "PredictedLabel"])
        writer.writerows(results)

    print("🎉 Prediction complete. Results saved in Results.csv")

# Call the prediction function
predict_test_set()


