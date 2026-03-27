"""
Strategy #2 weekly rebalancing:
  BTC cap:      B=$500, C=$750 (max, Polymarket BTC liquidity)
  DOGE/BNB cap: B=$75,  C=$100

Start capital: $1,000
Weekly rebalancing — bets scale proportionally until each pair hits its cap.
"""

# Monthly P&L contribution per pair at starting bets ($1k capital)
# BTC start: $20/$30 → monthly ~$874
# DOGE start: $7.5/$10 → monthly ~$289
# BNB start:  $7.5/$10 → monthly ~$290

CAPITAL_START = 1_000

# Pair configs: (start_bet_base, cap_bet_base, monthly_pnl_at_start)
PAIRS = {
    "BTC":  {"start": 20,  "cap": 500,  "monthly_pnl_start": 874},
    "DOGE": {"start": 7.5, "cap": 75,   "monthly_pnl_start": 289},
    "BNB":  {"start": 7.5, "cap": 75,   "monthly_pnl_start": 290},
}

# Each pair's cap is hit when capital = (cap/start) × CAPITAL_START
for p, cfg in PAIRS.items():
    cfg["cap_capital"] = (cfg["cap"] / cfg["start"]) * CAPITAL_START

print(f"Strategy #2  |  $1,000 старт  |  еженедельный пересчёт")
print(f"BTC лимит: $500/$750  |  DOGE/BNB лимит: $75/$100")
print()
print(f"  Пары достигают лимита при:")
for p, cfg in PAIRS.items():
    print(f"    {p}: ${cfg['cap_capital']:,.0f} на счёте")

print()
print(f"  {'Мес':>4} | {'На счёте':>12} | {'Доход за мес':>14} | Ставки BTC")
print(f"  {'-'*62}")

capital = CAPITAL_START
results = []

for month in range(1, 13):
    start_cap = capital
    monthly_pnl = 0

    for week in range(4):
        week_pnl = 0
        for p, cfg in PAIRS.items():
            eff_cap = min(capital, cfg["cap_capital"])
            scale = eff_cap / CAPITAL_START
            week_pnl += (cfg["monthly_pnl_start"] / 4.33) * scale
        capital += week_pnl
        monthly_pnl += week_pnl

    # Current BTC bet size
    btc_scale = min(capital, PAIRS["BTC"]["cap_capital"]) / CAPITAL_START
    btc_d = 20 * btc_scale
    btc_dc = 30 * btc_scale

    results.append((month, capital, monthly_pnl))

    mode = ""
    if capital >= PAIRS["BTC"]["cap_capital"]:
        mode = "все лимиты"
    elif capital >= PAIRS["DOGE"]["cap_capital"]:
        mode = "DOGE/BNB на лимите"
    else:
        mode = "масштабирование"

    print(f"  {month:>4} | ${capital:>11,.0f} | ${monthly_pnl:>12,.0f}  | BTC ${btc_d:.0f}/${btc_dc:.0f}  {mode}")

print()
print(f"  {'='*62}")
final = results[-1][1]
print(f"  Итог за 12 месяцев:")
print(f"    Стартовый капитал  : $1,000")
print(f"    Финальный капитал  : ${final:,.0f}")
print(f"    Прибыль            : ${final - CAPITAL_START:,.0f}")
print(f"    Рост               : +{(final/CAPITAL_START - 1)*100:.0f}%")
print()

# Steady state (after all caps hit)
monthly_max = sum(cfg["monthly_pnl_start"] * (cfg["cap"] / cfg["start"]) for cfg in PAIRS.values())
print(f"  При выходе на все лимиты (стабильная фаза):")
print(f"    BTC:      B=$500 / C=$750  → ~${PAIRS['BTC']['monthly_pnl_start'] * 25:,.0f}/мес")
print(f"    DOGE/BNB: B=$75  / C=$100  → ~${(PAIRS['DOGE']['monthly_pnl_start'] + PAIRS['BNB']['monthly_pnl_start']) * 10:,.0f}/мес")
print(f"    Итого:    ~${monthly_max:,.0f}/мес")
