import cloudscraper

class SofaClient:

    def __init__(self):
        self.scraper = cloudscraper.create_scraper()

    def get(self, url):
        r = self.scraper.get(url, timeout=20)
        r.raise_for_status()
        return r.text
