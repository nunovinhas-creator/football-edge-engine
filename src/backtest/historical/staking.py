"""
Estratégias de gestão de banca (staking) para o Backtesting Framework.

Estas estratégias apenas decidem QUANTO apostar (fração da banca ou stake
fixo). Não recalculam Edge, EV ou Kelly — reutilizam sempre
`src.engine.kelly.kelly_fraction` / `fractional_kelly` para a fração de
Kelly, mantendo uma única fonte de verdade matemática.

Melhoria #6 (auditoria matemática): `KellyStake.stake_for` aceita agora,
opcionalmente, `lambda_tier`/`effective_sample_size`
(`LambdaEstimate.tier`/`.effective_sample_size`, já disponíveis em
`HistoricalBet` desde a Melhoria #8) para escalar a fração de Kelly pela
confiança do modelo, via `src.engine.kelly.fractional_kelly` — sem
recalcular nada localmente. Omissos, o comportamento é exatamente igual
ao de antes desta melhoria.
"""

from dataclasses import dataclass
from typing import Optional

from src.engine.kelly import fractional_kelly, kelly_fraction


class StakingStrategy:
    """Interface comum das estratégias de stake."""

    def stake_for(
        self,
        probability: float,
        odd: float,
        bankroll: float = 1.0,
        lambda_tier: Optional[str] = None,
        effective_sample_size: Optional[float] = None,
    ) -> float:
        raise NotImplementedError


@dataclass
class FlatStake(StakingStrategy):
    """
    Stake fixo (unidade constante), independente do edge/kelly.
    É a estratégia por omissão quando não existe gestão de banca.
    """

    unit: float = 1.0

    def stake_for(
        self,
        probability: float,
        odd: float,
        bankroll: float = 1.0,
        lambda_tier: Optional[str] = None,
        effective_sample_size: Optional[float] = None,
    ) -> float:
        return self.unit


@dataclass
class KellyStake(StakingStrategy):
    """
    Stake baseado em Kelly fracionário, como fração da banca.

    fraction:   fração do Kelly completo a usar (ex. 0.25 = Quarter Kelly).
    cap:        limite máximo de stake, como fração da banca (proteção de risco).
    bankroll:   banca de referência usada para converter a fração em valor
                monetário quando `bankroll` não é passado explicitamente a
                `stake_for`.
    """

    fraction: float = 0.25
    cap: float = 0.05
    bankroll: float = 1.0

    def stake_for(
        self,
        probability: float,
        odd: float,
        bankroll: float = None,
        lambda_tier: Optional[str] = None,
        effective_sample_size: Optional[float] = None,
    ) -> float:
        bankroll = self.bankroll if bankroll is None else bankroll
        kelly_pct = fractional_kelly(
            probability,
            odd,
            fraction=self.fraction,
            lambda_tier=lambda_tier,
            effective_sample_size=effective_sample_size,
        )
        kelly_pct = min(kelly_pct, self.cap)
        return round(kelly_pct * bankroll, 4)
