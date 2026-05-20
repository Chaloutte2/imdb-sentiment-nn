import os
import json
import joblib
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


#Create model directory


os.makedirs("model", exist_ok=True)


#Load dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "data", "imdb_balanced_10k.csv")

df = pd.read_csv(data_path)

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

#Convert labels
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

#Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN],
    df[LABEL_COLUMN],
    test_size=0.2,
    random_state=42
)


#TF-IDF Vectorization


vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()


#Convert to tensors


X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

X_test_tensor = torch.tensor(X_test_vec, dtype=torch.float32)


#Neural Network


class SentimentNN(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = SentimentNN(X_train_vec.shape[1])

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


#Training


epochs = 5

for epoch in range(epochs):

    model.train()

    outputs = model(X_train_tensor)

    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")


#Evaluation


model.eval()

with torch.no_grad():

    predictions = model(X_test_tensor)

    predictions = (
        predictions >= 0.5
    ).int().numpy().flatten()

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.4f}")

#Save model artifacts

torch.save(
    model.state_dict(),
    "model/model.pt"
)

joblib.dump(
    vectorizer,
    "model/vectorizer.pkl"
)

config = {
    "input_dim": X_train_vec.shape[1]
}

with open("model/config.json", "w") as f:
    json.dump(config, f)

metrics = {
    "accuracy": float(accuracy)
}

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Artifacts saved successfully.")
