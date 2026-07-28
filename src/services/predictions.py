from src.api.client import BzzoiroClient

client = BzzoiroClient()

def get_predictions(limit=10):
    return client.get(f"predictions/?limit={limit}")
