from src.engine.edge import (
    implied_probability,
    calculate_edge,
    calculate_ev
)


odd = 2.10
model_pct = 55
model = model_pct / 100.0  # calculate_edge/calculate_ev esperam fração (0.0-1.0)


market = implied_probability(
    odd
)


edge = calculate_edge(
    model,
    odd
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
