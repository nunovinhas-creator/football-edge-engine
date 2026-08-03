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
        h2h_matches=0,
        dixon_coles_probabilities=None
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

        # Probabilidades 1X2 (fração 0.0-1.0) do modelo Dixon-Coles existente
        # (src/engine/dixon_coles.py, via src/engine/value.py), calculadas a
        # partir de xg_home/xg_away como lambda_home/mu_away. dict com chaves
        # "home"/"draw"/"away", ou None quando não foi possível calcular
        # (ver src/engine/pregame_lambda.py).
        self.dixon_coles_probabilities = dixon_coles_probabilities


    def to_dict(self):

        return {
            "match": f"{self.home} vs {self.away}",
            "league": self.league,
            "odds": self.odds,
            "probability": self.probability,
            "xg_home": self.xg_home,
            "xg_away": self.xg_away,
            "confidence": self.confidence,
            "dixon_coles_probabilities": self.dixon_coles_probabilities
        }
