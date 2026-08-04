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

    Campos opcionais de confiança do modelo (Melhoria #8 da auditoria
    matemática — propagação de `src.engine.lambda_estimator.LambdaEstimate`
    até ao Evaluation Framework, sobretudo para efeitos de avaliação/
    segmentação; nunca usados no cálculo de `probability`/edge/EV/`kelly`
    (Kelly completo, sem fração) nem em nenhuma decisão do motor
    (`engine_decision`/`placed`). Desde a Melhoria #6, `evaluate_bet`
    passa-os à estratégia de staking (`src.backtest.historical.staking`),
    que os pode usar para escalar a FRAÇÃO de Kelly pela confiança do
    modelo — afeta apenas o `stake` de uma aposta já decidida, nunca a
    decisão em si):
        model_confidence: rótulo "HIGH"/"MEDIUM"/"LOW" (ver
                          `src.engine.lambda_estimator.classify_model_confidence`).
        lambda_tier:      `LambdaEstimate.tier` — proveniência da estimativa
                          de lambda ("recent_matches" | "h2h_goal_totals" |
                          "avg_total_goals_or_prior").
        effective_sample_size: `LambdaEstimate.effective_sample_size`.
        Todos opcionais e retrocompatíveis: ficheiros/registos antigos que
        não os tragam continuam válidos (ficam a `None`, sem erro).
    """

    match: str
    date: Any
    market: str
    odd: float
    model_prob: float
    engine_decision: str
    result: Any
    competition: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    home_or_away: Optional[str] = None
    is_favorite: Optional[bool] = None
    model_confidence: Optional[str] = None
    lambda_tier: Optional[str] = None
    effective_sample_size: Optional[float] = None
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
            "home_team": ("home_team", "equipa_casa", "casa", "home"),
            "away_team": ("away_team", "equipa_visitante", "visitante", "fora", "away"),
            "home_or_away": ("home_or_away", "casa_fora", "venue"),
            "is_favorite": ("is_favorite", "favorito"),
            "model_confidence": ("model_confidence", "confianca_modelo"),
            "lambda_tier": ("lambda_tier", "nivel_confianca_lambda"),
            "effective_sample_size": ("effective_sample_size", "amostra_efetiva"),
        }

        def _is_missing(value: Any) -> bool:
            """None, ou NaN (colunas opcionais ausentes num DataFrame/CSV)."""
            if value is None:
                return True
            if isinstance(value, float) and value != value:
                return True
            return False

        def pick(field_name: str, required: bool = True) -> Any:
            for key in aliases[field_name]:
                if key in row and not _is_missing(row[key]):
                    return row[key]
            if required:
                raise KeyError(
                    f"Campo obrigatório em falta: {field_name} "
                    f"(aceite qualquer de {aliases[field_name]})"
                )
            return None

        known_keys = {alias for group in aliases.values() for alias in group}
        extra = {k: v for k, v in row.items() if k not in known_keys}

        home_team = pick("home_team", required=False)
        away_team = pick("away_team", required=False)
        match_value = pick("match", required=False)
        if not match_value:
            if home_team and away_team:
                match_value = f"{home_team} vs {away_team}"
            else:
                raise KeyError(
                    "Campo obrigatório em falta: match (aceite 'match'/'jogo'/'game', "
                    "ou em alternativa 'home_team'+'away_team')"
                )

        raw_effective_sample_size = pick("effective_sample_size", required=False)

        return cls(
            match=match_value,
            date=pick("date"),
            market=pick("market"),
            odd=float(pick("odd")),
            model_prob=float(pick("model_prob")),
            engine_decision=pick("engine_decision"),
            result=pick("result"),
            competition=pick("competition", required=False),
            home_team=home_team,
            away_team=away_team,
            home_or_away=pick("home_or_away", required=False),
            is_favorite=pick("is_favorite", required=False),
            model_confidence=pick("model_confidence", required=False),
            lambda_tier=pick("lambda_tier", required=False),
            effective_sample_size=(
                float(raw_effective_sample_size) if raw_effective_sample_size is not None else None
            ),
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

    home_team: Optional[str] = None
    away_team: Optional[str] = None

    # Melhoria #8 (auditoria matemática): metadados opcionais de confiança
    # do modelo, propagados de `HistoricalBet` sem qualquer alteração —
    # nunca usados nos cálculos de probability/edge/ev/kelly (Kelly
    # completo) acima, só para segmentação no Evaluation Framework (ver
    # `src.evaluation.segments`). Desde a Melhoria #6, podem influenciar
    # `stake` (não `kelly`) quando a estratégia de staking usada é
    # `KellyStake` — ver `src.backtest.historical.evaluator.evaluate_bet`.
    model_confidence: Optional[str] = None
    lambda_tier: Optional[str] = None
    effective_sample_size: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match": self.match,
            "date": self.date,
            "competition": self.competition,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "market": self.market,
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
            "model_confidence": self.model_confidence,
            "lambda_tier": self.lambda_tier,
            "effective_sample_size": self.effective_sample_size,
        }
