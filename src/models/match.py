class Match:

    def __init__(
        self,
        home,
        away,
        odds,
        probability,
        league=None,
        xg_home=None,
        xg_away=None,
        confidence=None,
        h2h_matches=0
    ):

        self.home = home
        self.away = away
        self.odds = odds
        self.odd = odds
        self.probability = probability
        self.league = league
        self.xg_home = xg_home
        self.xg_away = xg_away
        self.confidence = confidence
        self.h2h_matches = h2h_matches


    def to_dict(self):

        return {
            "match": f"{self.home} vs {self.away}",
            "league": self.league,
            "odds": self.odds,
            "probability": self.probability,
            "xg_home": self.xg_home,
            "xg_away": self.xg_away,
            "confidence": self.confidence
        }
