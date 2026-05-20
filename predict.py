import json
import joblib
import torch
import torch.nn as nn


#Load config


with open("model/config.json") as f:
    config = json.load(f)

#Load vectorizer


vectorizer = joblib.load(
    "model/vectorizer.pkl"
)


#Neural Network


class SentimentNN(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = SentimentNN(config["input_dim"])

model.load_state_dict(
    torch.load("model/model.pt")
)

model.eval()


#Prediction


text = input("Enter a movie review: ")

x = vectorizer.transform([text]).toarray()

x_tensor = torch.tensor(
    x,
    dtype=torch.float32
)

with torch.no_grad():

    prediction = model(x_tensor).item()

if prediction >= 0.5:
    print("Positive review")
else:
    print("Negative review")
