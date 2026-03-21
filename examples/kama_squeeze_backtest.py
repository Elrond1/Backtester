"""
Backtest: KAMA 21 + Squeeze Momentum (LazyBear)
Balance : $10,000  |  From: 2020-01-01
Timeframes tested: 1h and 4h

Risk Management:
- 2.5% risk per trade (position sizing based on SL distance)
- Stop-Loss: min/max(KAMA, signal-candle low/high) — whichever is tighter
- Trailing Stop: activates at 1:1.5 RR, trails at 2×ATR behind price
- Take-Profit: histogram brightness fades (lime→green or red→maroon)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtester.data import get_ohlcv
from backtester.engine.backtester import BacktestResult
from backtester.engine.metrics import compute_metrics
from backtester.strategy.kama_squeeze import KamaSqueezeStrategy
from backtester.strategy.indicators import atr as calc_atr, ema as calc_ema
from backtester.visualization.charts import plot_backtest

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL     = "BTC/USDT"
SINCE      = "2020-01-01"
CAPITAL    = 10_000.0
COMMISSION = 0.001      # 0.1% taker fee
SLIPPAGE   = 0.0005     # 0.05%
RISK_PCT   = 0.025      # 2.5% risk per trade
ATR_PERIOD = 14
ATR_TRAIL  = 2.0        # trailing stop = 2×ATR behind price
RR_TRAIL   = 1.5        # activate trailing stop after 1:1.5 RR
MAX_LEVER  = 1.0        # NO leverage — cap at 100% of capital
EMA_TREND  = 200        # only long above EMA200, only short below

strategy = KamaSqueezeStrategy(
    kama_period = 21,
    sq_length   = 20,
    sq_mult_bb  = 2.0,
    sq_mult_kc  = 1.5,
    sq_mom      = 12,
    min_squeeze_bars = 5,
)


# ── Custom bar-by-bar simulation ──────────────────────────────────────────────

def _run_kama_squeeze_advanced(df, strategy, initial_capital, commission, slippage, symbol, timeframe):
    """
    Bar-by-bar simulation with:
      - Dynamic SL behind KAMA or signal-candle extremum
      - Position sizing: RISK_PCT / SL_distance
      - Trailing stop after 1:RR_TRAIL (at ATR_TRAIL × ATR)
      - Exit on histogram brightness fade (lime→green, red→maroon)
    """
    close_arr = df["close"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    times     = df.index
    n         = len(times)

    # Indicators
    kama_line, squeeze_on, histogram, color, valid_release, kama_rising, kama_falling = (
        strategy.get_raw_data(df)
    )
    atr_arr  = calc_atr(df, ATR_PERIOD).values
    ema_arr  = calc_ema(df["close"], EMA_TREND).values

    kama_arr     = kama_line.values
    hist_arr     = histogram.values
    color_arr    = color.values
    vrel_arr     = valid_release.values
    krise_arr    = kama_rising.values
    kfall_arr    = kama_falling.values

    # ── State ─────────────────────────────────────────────────────────────────
    capital     = initial_capital
    equity_arr  = np.full(n, initial_capital)
    returns_arr = np.zeros(n)
    pos_arr     = np.zeros(n)
    trades      = []

    in_trade      = False
    side          = 0
    entry_idx     = 0
    entry_price   = 0.0
    sl_price      = 0.0
    sl_dist_pct   = 0.0
    pos_fraction  = 0.0    # fraction of capital risked (can be >1 if leverage)
    trail_active  = False
    trail_price   = 0.0

    for i in range(1, n):
        sig_idx = i - 1         # signal candle index (previous bar)
        c = close_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        bar_return = 0.0

        # ── Entry conditions (signal from bar i-1, execute at bar i) ──────────
        if not np.isnan(atr_arr[i]) and not np.isnan(kama_arr[sig_idx]) and not np.isnan(ema_arr[sig_idx]):
            long_sig  = (
                vrel_arr[sig_idx]
                and color_arr[sig_idx] == "lime"
                and close_arr[sig_idx] > kama_arr[sig_idx]
                and krise_arr[sig_idx]
                and close_arr[sig_idx] > ema_arr[sig_idx]   # above EMA200
            )
            short_sig = (
                vrel_arr[sig_idx]
                and color_arr[sig_idx] == "red"
                and close_arr[sig_idx] < kama_arr[sig_idx]
                and kfall_arr[sig_idx]
                and close_arr[sig_idx] < ema_arr[sig_idx]   # below EMA200
            )
        else:
            long_sig = short_sig = False

        # ── Manage open trade ─────────────────────────────────────────────────
        if in_trade:
            pos_arr[i] = side * pos_fraction

            # Update trailing stop (only move in profit direction)
            if trail_active:
                if side == 1:
                    new_trail = h - ATR_TRAIL * atr_arr[i]
                    if new_trail > trail_price:
                        trail_price = new_trail
                else:
                    new_trail = l + ATR_TRAIL * atr_arr[i]
                    if new_trail < trail_price:
                        trail_price = new_trail

            # --- Check stops (SL before trail) ---
            exit_price = None
            exit_type  = None

            # Hard SL
            if side == 1 and l <= sl_price:
                exit_price = sl_price * (1 - slippage)
                exit_type  = "stop_loss"
            elif side == -1 and h >= sl_price:
                exit_price = sl_price * (1 + slippage)
                exit_type  = "stop_loss"

            # Trailing SL
            if exit_price is None and trail_active:
                if side == 1 and l <= trail_price:
                    exit_price = trail_price * (1 - slippage)
                    exit_type  = "trail_stop"
                elif side == -1 and h >= trail_price:
                    exit_price = trail_price * (1 + slippage)
                    exit_type  = "trail_stop"

            # Exit when histogram crosses zero (momentum exhausted)
            if exit_price is None:
                hist_prev = hist_arr[i - 1]
                hist_curr = hist_arr[i]
                zero_cross_long  = (side == 1  and hist_prev > 0 and hist_curr <= 0)
                zero_cross_short = (side == -1 and hist_prev < 0 and hist_curr >= 0)
                if zero_cross_long or zero_cross_short:
                    exit_price = c * (1 - side * slippage)
                    exit_type  = "zero_cross"

            # Reversal signal
            if exit_price is None:
                reversal = (side == 1 and short_sig) or (side == -1 and long_sig)
                if reversal:
                    exit_price = c * (1 - side * slippage)
                    exit_type  = "reversal"

            # End of data
            if exit_price is None and i == n - 1:
                exit_price = c * (1 - side * slippage)
                exit_type  = "end_of_data"

            if exit_price is not None:
                raw_pnl = side * (exit_price / entry_price - 1)
                cost    = 2 * (commission + slippage)
                net_pnl = raw_pnl - cost
                bar_return += net_pnl * pos_fraction
                capital   *= (1 + net_pnl * pos_fraction)

                trades.append({
                    "entry_time":  times[entry_idx],
                    "exit_time":   times[i],
                    "side":        "long" if side == 1 else "short",
                    "entry_price": round(entry_price, 6),
                    "exit_price":  round(exit_price, 6),
                    "pnl_pct":     round(net_pnl * 100, 4),
                    "duration":    times[i] - times[entry_idx],
                    "exit_type":   exit_type,
                    "pos_frac":    round(pos_fraction, 4),
                })

                in_trade     = False
                trail_active = False

                # Immediate reversal entry
                if exit_type == "reversal":
                    new_side  = -side
                    new_ep    = c * (1 + new_side * slippage)
                    sig_low   = low_arr[sig_idx]
                    sig_high  = high_arr[sig_idx]
                    kama_s    = kama_arr[sig_idx]
                    if new_side == 1:
                        sl_cand = min(kama_s, sig_low)
                        sl_p    = min(sl_cand, new_ep * 0.995)
                    else:
                        sl_cand = max(kama_s, sig_high)
                        sl_p    = max(sl_cand, new_ep * 1.005)
                    sl_d = abs(new_ep - sl_p) / new_ep
                    pf   = min(RISK_PCT / sl_d if sl_d > 0 else 0.1, MAX_LEVER)
                    cost_r = (commission + slippage) * pf
                    bar_return -= cost_r
                    capital   *= (1 - cost_r)
                    in_trade      = True
                    side          = new_side
                    entry_idx     = i
                    entry_price   = new_ep
                    sl_price      = sl_p
                    sl_dist_pct   = sl_d
                    pos_fraction  = pf
                    trail_active  = False
                    pos_arr[i]    = side * pos_fraction

            # Check if trailing stop should now activate
            if in_trade and not trail_active:
                if side == 1:
                    profit_frac = (c - entry_price) / entry_price
                else:
                    profit_frac = (entry_price - c) / entry_price
                if profit_frac >= RR_TRAIL * sl_dist_pct:
                    trail_active = True
                    trail_price  = (c - ATR_TRAIL * atr_arr[i] if side == 1
                                    else c + ATR_TRAIL * atr_arr[i])

        # ── Enter new trade (if flat) ─────────────────────────────────────────
        if not in_trade:
            new_side = 1 if long_sig else (-1 if short_sig else 0)
            if new_side != 0 and not np.isnan(atr_arr[i]) and not np.isnan(kama_arr[sig_idx]):
                ep     = c * (1 + new_side * slippage)
                kama_s = kama_arr[sig_idx]
                sig_lo = low_arr[sig_idx]
                sig_hi = high_arr[sig_idx]

                if new_side == 1:
                    sl_cand = min(kama_s, sig_lo)
                    sl_p    = min(sl_cand, ep * 0.995)
                else:
                    sl_cand = max(kama_s, sig_hi)
                    sl_p    = max(sl_cand, ep * 1.005)

                sl_d = abs(ep - sl_p) / ep
                pf   = min(RISK_PCT / sl_d if sl_d > 0 else 0.1, MAX_LEVER)
                cost = (commission + slippage) * pf
                bar_return -= cost
                capital    *= (1 - cost)

                in_trade     = True
                side         = new_side
                entry_idx    = i
                entry_price  = ep
                sl_price     = sl_p
                sl_dist_pct  = sl_d
                pos_fraction = pf
                trail_active = False
                pos_arr[i]   = side * pos_fraction

        returns_arr[i] = bar_return
        equity_arr[i]  = capital

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "side", "entry_price",
                 "exit_price", "pnl_pct", "duration", "exit_type", "pos_frac"]
    )
    equity   = pd.Series(equity_arr, index=df.index)
    returns  = pd.Series(returns_arr, index=df.index)
    positions = pd.Series(pos_arr, index=df.index)

    metrics = compute_metrics(returns, equity, trades_df, initial_capital)

    return BacktestResult(
        equity=equity,
        returns=returns,
        positions=positions,
        trades=trades_df,
        metrics=metrics,
        params=strategy.get_params(),
        symbol=symbol,
        timeframe=timeframe,
    )


# ── Run tests ─────────────────────────────────────────────────────────────────

for tf in ["1d", "4h", "1h"]:
    print(f"\n{'#'*62}")
    print(f"  {SYMBOL}  {tf}  |  {SINCE} → today  |  Capital: ${CAPITAL:,.0f}")
    print(f"  Strategy : KAMA 21 + Squeeze Momentum (LazyBear)")
    print(f"  Risk     : {RISK_PCT*100:.1f}% per trade  |  Trail: {ATR_TRAIL}×ATR after 1:{RR_TRAIL} RR")
    print(f"{'#'*62}")

    df = get_ohlcv(SYMBOL, tf, since=SINCE)
    print(f"  Loaded {len(df):,} candles  ({df.index[0]} → {df.index[-1]})\n")

    result = _run_kama_squeeze_advanced(
        df, strategy,
        initial_capital=CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        symbol=SYMBOL,
        timeframe=tf,
    )

    print(result.report())

    # ── Exit type breakdown ───────────────────────────────────────────────────
    if not result.trades.empty:
        trades = result.trades.copy()

        if "exit_type" in trades.columns:
            print(f"\n  Exit type breakdown:")
            for etype, grp in trades.groupby("exit_type"):
                wr  = (grp["pnl_pct"] > 0).mean() * 100
                avg = grp["pnl_pct"].mean()
                print(f"    {etype:<15} trades={len(grp):>4}  win={wr:>5.1f}%  avg_pnl={avg:>7.2f}%")

        # ── Trades by year ────────────────────────────────────────────────────
        trades["year"] = trades["entry_time"].dt.year
        print(f"\n  Trades by year:")
        print(f"  {'Year':<6} {'Trades':>7} {'Win%':>7} {'PnL%':>10}")
        print(f"  {'-'*34}")
        for yr, grp in trades.groupby("year"):
            wr  = (grp["pnl_pct"] > 0).mean() * 100
            tot = grp["pnl_pct"].sum()
            print(f"  {yr:<6} {len(grp):>7}  {wr:>6.1f}%  {tot:>9.1f}%")

        print(f"\n  Last 10 trades:")
        cols = ["entry_time", "exit_time", "side", "entry_price",
                "exit_price", "pnl_pct", "exit_type", "pos_frac"]
        print(result.trades[cols].tail(10).to_string(index=False))

    # ── Chart ─────────────────────────────────────────────────────────────────
    inds = strategy.get_indicators(df)
    ema200 = calc_ema(df["close"], EMA_TREND)
    html_name = f"kama_squeeze_{tf.replace('h', 'H')}.html"
    plot_backtest(
        result, df,
        indicators={"KAMA 21": inds["KAMA 21"], f"EMA {EMA_TREND}": ema200},
        title=f"{SYMBOL} {tf} — KAMA 21 + Squeeze + EMA200  (2.5% risk | ATR trail)",
        save_html=html_name,
    )
    print(f"\n  Chart saved → {html_name}")
