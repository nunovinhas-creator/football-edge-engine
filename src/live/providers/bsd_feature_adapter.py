
class BSDFeatureAdapter:

    def incidents_to_features(
        self,
        incidents,
        current_minute
    ):

        features = {
            "home_goals": 0,
            "away_goals": 0,
            "goals_last_15": 0,
            "last_goal_minute": None,
            "red_cards": 0,
            "game_state": "unknown"
        }


        for incident in incidents:

            event_type = incident.get("type")


            if event_type == "goal":

                if incident.get("is_home"):
                    features["home_goals"] += 1
                else:
                    features["away_goals"] += 1


                features["last_goal_minute"] = (
                    incident.get("minute")
                )


            if event_type == "card":

                if incident.get("card") == "red":
                    features["red_cards"] += 1



            if event_type == "period":

                if incident.get("is_live"):

                    features["game_state"] = (
                        incident.get("text")
                        or "live"
                    )


        last_goal = features["last_goal_minute"]


        if last_goal is not None:

            if current_minute - last_goal <= 15:
                features["goals_last_15"] = 1


        return features

