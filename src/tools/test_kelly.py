from src.engine.kelly import fractional_kelly


odd = 2.10
probability = 0.55


stake = fractional_kelly(
    probability,
    odd
)


print("Kelly fracionado:")
print(round(stake * 100, 2), "%")
