import pandas as pd
import numpy as np
import warnings
import os

def postprocess_yolo(output_labels_path, test_meta_path, sample_submission_path):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    my_predictions = pd.DataFrame(columns=["image_id", "MyPredictionString"])
    sample_submission = pd.read_csv(sample_submission_path)
    test_meta = pd.read_csv(test_meta_path)
    for file in os.listdir(output_labels_path):
        if file.endswith(".txt"):
            with open(output_labels_path + file, "r") as f:
                lines = f.readlines()
                prediction_string = ""
                for line in lines:
                    image_id = file[:-4]
                    line = line.split(" ")
                    abnormality_id = int(line[0])
                    x_mid_n = float(line[1])
                    y_mid_n = float(line[2])
                    width_n = float(line[3])
                    height_n = float(line[4])
                    confidence = float(line[5])
                    # Get the original image size:
                    dim0 = test_meta[test_meta["image_id"] == image_id]["dim0"].item()
                    dim1 = test_meta[test_meta["image_id"] == image_id]["dim1"].item()
                    # Get the new x_mid, y_mid, width and height:
                    x_mid = x_mid_n * dim1
                    y_mid = y_mid_n * dim0
                    width = width_n * dim1
                    height = height_n * dim0
                    # Get the new x_min, y_min, x_max and y_max:
                    x_min = int(np.round(x_mid - width / 2))
                    y_min = int(np.round(y_mid - height / 2))
                    x_max = int(np.round(x_mid + width / 2))
                    y_max = int(np.round(y_mid + height / 2))
                    # Add the abnorality to the prediction string:
                    prediction_string += str(abnormality_id) + " " + str(confidence) + " " + str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max) + " "
                # Add the prediction string to the dataframe:
                my_predictions = my_predictions.append({"image_id": image_id, "MyPredictionString": prediction_string}, ignore_index=True)
    # Join the processed_labels dataframe with the submission_sample dataframe on the image_id column:
    submission = sample_submission.join(my_predictions.set_index("image_id"), on="image_id", how="left").fillna("14 1 0 0 1 1").drop(columns=["PredictionString"]).rename(columns={"MyPredictionString": "PredictionString"})
    # Save the dataframe to a csv file:
    submission.to_csv("src/data/output/submission.csv", index=False)
    return submission