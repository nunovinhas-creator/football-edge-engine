from src.engine.edge import (
    implied_probability,
    calculate_edge,
    calculate_ev
)


odd = 2.10
model = 55


market = implied_probability(
    odd
)


edge = calculate_edge(
    model,
    market
)


ev = calculate_ev(
    model,
    odd
)


print("----------------")
print("Odd:", odd)
print("Modelo:", model)
print("Mercado:", market)
print("Edge:", edge)
print("EV:", ev)
