# IMPORTS=
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        # Set the main directory, the csv file and the transformation pipeline:
        self.transform = transform
        self.data_dir = data_dir
        self.csv_file = csv_file

        # Read the csv file:
        self.df = pd.read_csv(self.data_dir + self.csv_file)

        # Set the labels and the images:
        self.images = []
        self.labels = []
        self.bboxes = []
        for i, row in self.df.iterrows():
            image_path = self.data_dir + "train/" + row["image_id"] + ".png"
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            self.images.append(img)
            self.labels.append(row["class_id"])
            if row["class_id"] != 14:
                self.bboxes.append(torch.tensor([row["new_x_min"], row["new_y_min"], row["width"], row["height"]]))
            else:
                self.bboxes.append(torch.tensor([0, 0, 0, 0]))

    def split(self, ratio):
        # Split the dataset into train and validation sets:
        train_size = int(ratio * len(self))
        val_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, val_size])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.bboxes[idx]


def main():
    pass
if __name__ == "__main__":
    main()