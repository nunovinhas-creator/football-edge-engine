"""
Estruturas de dados do Backtesting Framework histórico.

Este módulo define o contrato de entrada (`HistoricalBet`) e de saída
(`EvaluatedBet`) do framework, sem tocar em nenhuma fórmula matemática do
motor (Poisson, Dixon-Coles, Monte Carlo, Goal Engine, Kelly, Edge, EV).
Toda a matemática é reutilizada de `src.engine.*`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class HistoricalBet:
    """
    Um registo histórico de aposta, tal como fornecido pelo utilizador.

    Campos obrigatórios:
        match:            identificador do jogo ("jogo"), ex. "Benfica vs Porto".
        date:             data do jogo ("data"), string ISO ou objeto date/datetime.
        market:           mercado da aposta ("mercado"), ex. "HOME", "OVER_2.5".
        odd:              odd decimal disponível no mercado (> 1.0).
        model_prob:       probabilidade prevista pelo modelo, em fração (0.0-1.0].
        engine_decision:  decisão histórica do motor (ex. "BET", "PASS", "WAIT").
        result:           resultado real da aposta — aceita bool, "WIN"/"LOSS"/
                          "WON"/"LOST", ou 1/0.

    Campos opcionais (usados nas análises por segmento):
        competition:      nome da competição/liga.
        home_or_away:     "HOME" ou "AWAY" (perspetiva da equipa apostada).
        is_favorite:      True se a seleção apostada era a favorita do mercado
                          (odd mais baixa entre as opções). Se omitido, é
                          inferido a partir da odd (odd <= 2.0 => favorito) só
                          para fins de segmentação — não é usado em nenhum
                          cálculo de edge/EV/kelly.
        extra:            quaisquer outros metadados livres.
    """

    match: str
    date: Any
    market: str
    odd: float
    model_prob: float
    engine_decision: str
    result: Any
    competition: Optional[str] = None
    home_or_away: Optional[str] = None
    is_favorite: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _coerce_result_won(result: Any) -> bool:
        """Normaliza os formatos aceites de resultado para um booleano `won`."""
        if isinstance(result, bool):
            return result
        if isinstance(result, (int, float)):
            return bool(result)
        if isinstance(result, str):
            normalized = result.strip().upper()
            if normalized in {"WIN", "WON", "W", "GREEN", "1"}:
                return True
            if normalized in {"LOSS", "LOST", "LOSE", "L", "RED", "0"}:
                return False
        raise ValueError(f"Resultado inválido: {result!r}")

    @property
    def won(self) -> bool:
        return self._coerce_result_won(self.result)

    @staticmethod
    def is_bet_decision(decision: Any) -> bool:
        """
        Interpreta a decisão histórica do motor como "aposta colocada".

        Aceita strings como "BET", "BET 🔥", "bet" (case-insensitive) e
        rejeita "PASS", "WAIT", None, etc. Isto acompanha o vocabulário já
        usado por `src.engine.decision` (BET 🔥 / PASS ❄️ / WAIT ⚠️).
        """
        if decision is None:
            return False
        return "bet" in str(decision).strip().lower()

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "HistoricalBet":
        """
        Cria um HistoricalBet a partir de um dicionário, aceitando tanto
        chaves em português (conforme os requisitos) como em inglês.
        """
        aliases = {
            "match": ("match", "jogo", "game"),
            "date": ("date", "data"),
            "market": ("market", "mercado"),
            "odd": ("odd", "odd_disponivel", "available_odd", "bookie_odd"),
            "model_prob": ("model_prob", "probabilidade", "probabilidade_prevista", "prob_model"),
            "engine_decision": ("engine_decision", "decisao", "decisao_motor", "decision"),
            "result": ("result", "resultado", "resultado_real"),
            "competition": ("competition", "competicao", "liga", "league"),
            "home_or_away": ("home_or_away", "casa_fora", "venue"),
            "is_favorite": ("is_favorite", "favorito"),
        }

        def pick(field_name: str, required: bool = True) -> Any:
            for key in aliases[field_name]:
                if key in row and row[key] is not None:
                    return row[key]
            if required:
                raise KeyError(
                    f"Campo obrigatório em falta: {field_name} "
                    f"(aceite qualquer de {aliases[field_name]})"
                )
            return None

        known_keys = {alias for group in aliases.values() for alias in group}
        extra = {k: v for k, v in row.items() if k not in known_keys}

        return cls(
            match=pick("match"),
            date=pick("date"),
            market=pick("market"),
            odd=float(pick("odd")),
            model_prob=float(pick("model_prob")),
            engine_decision=pick("engine_decision"),
            result=pick("result"),
            competition=pick("competition", required=False),
            home_or_away=pick("home_or_away", required=False),
            is_favorite=pick("is_favorite", required=False),
            extra=extra,
        )


def load_historical_bets(rows: Iterable[Dict[str, Any]]) -> List[HistoricalBet]:
    """Converte um iterável de dicts (ex. linhas de CSV/DataFrame) em HistoricalBet."""
    return [HistoricalBet.from_dict(row) for row in rows]


@dataclass
class EvaluatedBet:
    """
    Resultado do cálculo por aposta (probability, market probability, edge,
    ev, kelly, stake, resultado, lucro líquido), preservando os campos
    originais para permitir segmentação e relatórios.
    """

    match: str
    date: Any
    market: str
    competition: Optional[str]
    home_or_away: Optional[str]
    is_favorite: Optional[bool]

    odd: float
    probability: float
    market_probability: float
    edge: float
    ev: float
    kelly: float
    stake: float

    engine_decision: str
    placed: bool
    won: bool
    profit: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match": self.match,
            "date": self.date,
            "market": self.market,
            "competition": self.competition,
            "home_or_away": self.home_or_away,
            "is_favorite": self.is_favorite,
            "odd": self.odd,
            "probability": self.probability,
            "market_probability": self.market_probability,
            "edge": self.edge,
            "ev": self.ev,
            "kelly": self.kelly,
            "stake": self.stake,
            "engine_decision": self.engine_decision,
            "placed": self.placed,
            "won": self.won,
            "result": "WIN" if self.won else "LOSS",
            "profit": self.profit,
        }
