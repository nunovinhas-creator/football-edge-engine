from abc import ABC, abstractmethod


class MatchProvider(ABC):
    @abstractmethod
    def get_live_match(self, match_id):
        """
        Devolve o estado live do jogo.
        """
        pass
