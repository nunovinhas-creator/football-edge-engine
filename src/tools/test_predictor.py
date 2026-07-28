from src.api.client import BzzoiroClient
from src.model.predictor import predict_probability


client = BzzoiroClient()


data = client.get(
    "events/?limit=1"
)


event = data["results"][0]


probability = predict_probability(event)


print("----------------")
print(event["home_team"], "vs", event["away_team"])
print("H2H:")
print(event["head_to_head"])

print("----------------")
print("Probabilidade modelo:")
print(probability, "%")
