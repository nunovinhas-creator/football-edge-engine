from bs4 import BeautifulSoup
from .client import SofaClient

class SofaLiveFetcher:

    def __init__(self):
        self.client = SofaClient()

    def fetch(self, event_id):

        url = f"https://www.sofascore.com/event/{event_id}"

        html = self.client.get(url)

        soup = BeautifulSoup(html,"lxml")

        return soup
