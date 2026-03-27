"""
Weekly rebalancing simulation for Strategy #2 at $1,000 start capital.
Bets scale proportionally to capital each week.
Practical cap: $10,000 (liquidity ceiling on Polymarket).

Weekly P&L = monthly P&L / 4.33
"""

CAPITAL_START = 1_000
MONTHLY_PNL_AT_1K = 1_453   # at $1k capital (fixed bets)
LIQUIDITY_CAP = 10_000       # max capital where bets stop scaling (Polymarket limit)
WEEKS = 52

weekly_base = MONTHLY_PNL_AT_1K / 4.33  # ~$335.5/week at $1k

print(f"Strategy #2  |  $1,000 старт  |  перерасчёт раз в неделю")
print(f"Ликвидность Polymarket ограничивает рост до ~${LIQUIDITY_CAP:,}")
print()
print(f"{'Нед':>4} {'Мес':>4} | {'Капитал':>10} | {'Доход за нед':>14} | {'Ставки BTC':>12}")
print(f"{'-'*60}")

capital = CAPITAL_START
prev_month = 0

for week in range(1, WEEKS + 1):
    month = (week - 1) // 4 + 1

    # Scale bets proportionally, but cap at liquidity ceiling
    effective_capital = min(capital, LIQUIDITY_CAP)
    scale = effective_capital / CAPITAL_START
    weekly_pnl = weekly_base * scale

    capital_before = capital
    capital += weekly_pnl

    # Scale for display
    btc_d = 20 * scale
    btc_dc = 30 * scale

    if week <= 12 or week % 4 == 0:
        cap_note = " ← ліквід. кап" if capital_before >= LIQUIDITY_CAP and capital_before - weekly_pnl < LIQUIDITY_CAP else ""
        print(f"{week:>4} {month:>4} | ${capital:>9,.0f} | ${weekly_pnl:>12,.0f}  | BTC ${btc_d:.0f}/${btc_dc:.0f}{cap_note}")

print()
print(f"{'='*60}")
print(f"  Итог за 12 месяцев (52 недели):")
print(f"  Финальный капитал : ${capital:,.0f}")
print(f"  Прибыль           : ${capital - CAPITAL_START:,.0f}")
print(f"  Рост              : +{(capital / CAPITAL_START - 1)*100:.0f}%")

# Also show monthly summary
print(f"\n  Помесячно:")
print(f"  {'Мес':>4} | {'На счёте':>12} | {'Доход за мес':>14} | {'Режим'}")
print(f"  {'-'*55}")

capital2 = CAPITAL_START
for month in range(1, 13):
    start_cap = capital2
    for week in range(4):
        eff = min(capital2, LIQUIDITY_CAP)
        scale = eff / CAPITAL_START
        capital2 += weekly_base * scale
    monthly_gain = capital2 - start_cap
    mode = "масштабирование" if start_cap < LIQUIDITY_CAP else "фиксированные ставки"
    print(f"  {month:>4} | ${capital2:>11,.0f} | ${monthly_gain:>12,.0f}  | {mode}")
