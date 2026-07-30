from abc import ABC, abstractmethod


class OddsProvider(ABC):
    @abstractmethod
    def get_live_odds(self, match_id):
        """
        Devolve as odds live do jogo.
        """
        pass
