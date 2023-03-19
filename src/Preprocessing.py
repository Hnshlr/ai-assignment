import pandas as pd

def preprocess_data(data_dir, train_df, train_df_sizes):
    # Join rows with on image_id, and keep the class_id that is the most frequent:
    df_ = train_df.groupby("image_id")["class_id"].apply(lambda x: x.value_counts().index[0]).reset_index()
    # Remove the class_id column from the original dataframe:
    train_df.drop("class_id", axis=1, inplace=True)
    # In the train_df, merge the rows with the same image_id, and keep the first row:
    train_df = train_df.groupby("image_id").first().reset_index()
    # Merge the original dataframe with first jointure:
    train_df = pd.merge(train_df, df_, on=["image_id"])
    # Join the clean dataframe with the train_df_sizes dataframe containing the original image size:
    train_df = pd.merge(train_df, train_df_sizes, on=["image_id"])
    # Save in a .csv file
    train_df.to_csv(data_dir + "train_single.csv")
    return train_df

def class_ids_and_names(train_df):
    class_ids, class_names = [], []
    for row in train_df["class_id"]:
        # Add class ids that aren't already in the list:
        if row not in class_ids:
            class_ids.append(row)
            class_names.append(train_df[train_df["class_id"] == row]["class_name"].values[0])
    # Zip them and sort them by class id:
    class_ids, class_names = zip(*sorted(zip(class_ids, class_names)))
    return class_ids, class_names