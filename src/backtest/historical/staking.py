"""
Estratégias de gestão de banca (staking) para o Backtesting Framework.

Estas estratégias apenas decidem QUANTO apostar (fração da banca ou stake
fixo). Não recalculam Edge, EV ou Kelly — reutilizam sempre
`src.engine.kelly.kelly_fraction` / `fractional_kelly` para a fração de
Kelly, mantendo uma única fonte de verdade matemática.
"""

from dataclasses import dataclass

from src.engine.kelly import fractional_kelly, kelly_fraction


class StakingStrategy:
    """Interface comum das estratégias de stake."""

    def stake_for(self, probability: float, odd: float, bankroll: float = 1.0) -> float:
        raise NotImplementedError


@dataclass
class FlatStake(StakingStrategy):
    """
    Stake fixo (unidade constante), independente do edge/kelly.
    É a estratégia por omissão quando não existe gestão de banca.
    """

    unit: float = 1.0

    def stake_for(self, probability: float, odd: float, bankroll: float = 1.0) -> float:
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

    def stake_for(self, probability: float, odd: float, bankroll: float = None) -> float:
        bankroll = self.bankroll if bankroll is None else bankroll
        kelly_pct = fractional_kelly(probability, odd, fraction=self.fraction)
        kelly_pct = min(kelly_pct, self.cap)
        return round(kelly_pct * bankroll, 4)
