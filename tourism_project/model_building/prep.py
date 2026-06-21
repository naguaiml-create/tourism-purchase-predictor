# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/kottanagu/smart-tourism-customer-prediction-platform/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

#Corrected Gender values by merging "Fe Male" into "Female".
df['Gender'] = df['Gender'].str.replace(' ', '').str.capitalize()

#Creating categories from Age and Monthly Income
df['age_group'] = pd.cut(
    x=df['Age'],
    bins=[18, 30, 40, 50, 61],
    labels=["18-30", "31-40", "41-50", "51-60", "60+"],
)

# Standardize marital status values by replacing 'Unmarried' with 'Single'
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')


target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="kottanagu/smart-tourism-customer-prediction-platform",
        repo_type="dataset",
    )
