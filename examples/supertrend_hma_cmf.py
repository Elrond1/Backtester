"""
Backtest: SuperTrend + HMA 55 + CMF 20 on BTC/USDT
Balance : $10,000  |  From: 2020-01-01
Timeframes tested: 1h and 15m
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester.data import get_ohlcv
from backtester.engine.backtester import run_backtest
from backtester.strategy.supertrend_hma_cmf import SupertrendHmaCmfStrategy
from backtester.visualization.charts import plot_backtest

SYMBOL    = "BTC/USDT"
SINCE     = "2020-01-01"
CAPITAL   = 10_000.0
COMMISSION = 0.001    # 0.1% taker fee
SLIPPAGE   = 0.0005   # 0.05%

strategy = SupertrendHmaCmfStrategy(
    st_period=10,
    st_mult=3.0,
    hma_period=55,
    cmf_period=20,
)

for tf in ["1h", "15m"]:
    print(f"\n{'#'*55}")
    print(f"  {SYMBOL}  {tf}  |  {SINCE} → today  |  Capital: ${CAPITAL:,.0f}")
    print(f"{'#'*55}")

    df = get_ohlcv(SYMBOL, tf, since=SINCE)
    print(f"  Loaded {len(df):,} candles  ({df.index[0]} → {df.index[-1]})\n")

    result = run_backtest(
        df, strategy,
        initial_capital=CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        symbol=SYMBOL,
        timeframe=tf,
    )

    print(result.report())

    if not result.trades.empty:
        print(f"\n  First 10 trades:")
        print(result.trades.head(10).to_string(index=False))

    inds = strategy.get_indicators(df)
    plot_backtest(
        result, df,
        indicators={k: v for k, v in inds.items() if k != "CMF"},
        title=f"{SYMBOL} {tf} — SuperTrend + HMA55 + CMF20",
        save_html=f"supertrend_hma_cmf_{tf.replace('m','min')}.html",
    )
    print(f"\n  Chart saved → supertrend_hma_cmf_{tf.replace('m','min')}.html")
