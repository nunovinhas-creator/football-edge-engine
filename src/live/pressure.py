class PressureEngine:

    @staticmethod
    def score(minute, home_score, away_score, last_goal_minute, odds_over=None, home_xg=1.5, away_xga=1.2, red_cards=0):
        pressure = 0
        pressure += min(minute, 90) * 0.6
        diff = abs(home_score-away_score)
        pressure += 20 if diff==0 else 15 if diff==1 else 5
        if last_goal_minute is not None:
            pressure += min(max(minute-last_goal_minute,0),35)
        if odds_over:
            pressure += max(0,(2.0-odds_over)*30)
        pressure += home_xg*6
        pressure += away_xga*5
        pressure += red_cards*5
        return round(min(pressure,100),1)
