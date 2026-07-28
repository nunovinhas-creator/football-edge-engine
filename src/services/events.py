from src.api.client import BzzoiroClient


class EventsService:

    def __init__(self):
        self.client = BzzoiroClient()

    def list(self, limit=20):
        return self.client.get(f"events/?limit={limit}")

    def live(self):
        return self.client.get("events/?status=inprogress")

    def upcoming(self):
        return self.client.get("events/?status=notstarted")
