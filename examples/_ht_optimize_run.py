"""Standalone optimizer — не импортирует backtester верхний уровень."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings; warnings.filterwarnings("ignore")
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

# Импортируем только нужные подмодули (без data — там DB)
from backtester.strategy.breakbar import BreakBar
from backtester.strategy.indicators import halftrend
from backtester.engine.backtester import BacktestResult
from backtester.engine.metrics import compute_metrics
from backtester.visualization.charts import plot_backtest


# ── copy run_mm_backtest без импорта data ──────────────────────────
def run_mm(df, strategy, risk_per_trade=0.06, initial_capital=10_000,
           commission=0.001, slippage=0.0005, symbol="", timeframe=""):
    raw  = strategy.generate_signals(df).reindex(df.index).fillna(0)
    zsl  = strategy.zone_sl_levels(df)
    swl, swh = strategy.swing_levels(df)

    signals = raw.shift(1).fillna(0).values.astype(int)
    zsl_v   = zsl.shift(1).fillna(np.nan).values
    swl_v   = swl.shift(1).fillna(np.nan).values
    swh_v   = swh.shift(1).fillna(np.nan).values

    c_arr = df["close"].values; h_arr = df["high"].values; l_arr = df["low"].values
    times = df.index; n = len(times)

    trades=[]; capital=initial_capital
    eq=np.empty(n); ret=np.zeros(n); pos=np.zeros(n)
    in_trade=False; side=0; eidx=0; ep=sl=tp=pw=0.0; block=False

    for i in range(n):
        sig=signals[i]; br=0.0; c,h,l=c_arr[i],h_arr[i],l_arr[i]
        if in_trade:
            pos[i]=side; xp=None; forced=False; reason=""
            if side==1:
                if l<=sl: xp=sl; forced=True; reason="sl"
                elif h>=tp: xp=tp; forced=True; reason="tp"
            else:
                if h>=sl: xp=sl; forced=True; reason="sl"
                elif l<=tp: xp=tp; forced=True; reason="tp"
            if not forced and (sig==-side or i==n-1):
                xp = c*(1-side*slippage)
            if xp is not None:
                if forced: xp*=(1-side*slippage)
                raw_r = side*(xp/ep-1)
                costs = pw*2*(commission+slippage)
                np_ = pw*raw_r - costs
                br+=np_; capital*=(1+np_)
                trades.append({"entry_time":times[eidx],"exit_time":times[i],
                    "side":"long" if side==1 else "short",
                    "entry_price":round(ep,2),"exit_price":round(xp,2),
                    "sl_price":round(sl,2),"tp_price":round(tp,2),
                    "exit_reason":reason if forced else "signal",
                    "pnl_pct":round(np_*100,4),"duration":times[i]-times[eidx]})
                in_trade=False
                if forced: block=True
        if not in_trade and not block and sig!=0:
            ep_=c*(1+sig*slippage)
            z=zsl_v[i]
            sl_ = (z if not np.isnan(z) else (swl_v[i] if sig==1 else swh_v[i]))
            if np.isnan(sl_): eq[i]=capital; ret[i]=br; block=False; continue
            d=(ep_-sl_) if sig==1 else (sl_-ep_)
            if d<=0 or d/ep_>0.30: eq[i]=capital; ret[i]=br; block=False; continue
            pw_ = min(risk_per_trade/(d/ep_), 5.0)
            tp_ = ep_+sig*strategy.rr_ratio*d
            in_trade=True; side=sig; eidx=i; ep=ep_; sl=sl_; tp=tp_; pw=pw_
            pos[i]=side; cost=pw_*(commission+slippage); br-=cost; capital*=(1-cost)
        block=False; ret[i]=br; eq[i]=capital

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time","exit_time","side","entry_price","exit_price",
                 "sl_price","tp_price","exit_reason","pnl_pct","duration"])
    equity=pd.Series(eq,index=df.index); rets=pd.Series(ret,index=df.index)
    metrics=compute_metrics(rets, equity, tdf, initial_capital)
    return BacktestResult(equity=equity,returns=rets,positions=pd.Series(pos,index=df.index),
        trades=tdf,metrics=metrics,
        params={**strategy.get_params(),"risk_per_trade":risk_per_trade},
        symbol=symbol,timeframe=timeframe)


# ── Стратегия с HalfTrend фильтром ────────────────────────────────
class BreakBarHT(BreakBar):
    def __init__(self, ht_amplitude=2, ht_channel=2.0, ht_atr=100, **kw):
        super().__init__(**kw)
        self.ht_amplitude=ht_amplitude; self.ht_channel=ht_channel; self.ht_atr=ht_atr
    def generate_signals(self, df, aux=None):
        sig = super().generate_signals(df, aux).copy()
        ht_dir, _ = halftrend(df, amplitude=self.ht_amplitude,
                              channel_dev=self.ht_channel, atr_period=self.ht_atr)
        sig[(sig== 1)&(ht_dir!= 1)]=0
        sig[(sig==-1)&(ht_dir!=-1)]=0
        return sig


if __name__ == "__main__":
    df = pd.read_csv("/tmp/btc_4h.csv", index_col=0, parse_dates=True)
    RISK = 0.06

    grid = {
        "min_size_pct": [4.0, 5.0, 6.0, 7.0, 8.0],
        "min_bars_in":  [6, 8, 10, 12],
        "rr_ratio":     [2.0, 2.5, 3.0, 3.5, 4.0],
        "ht_amplitude": [1, 2, 3, 4],
        "ht_channel":   [1.5, 2.0, 2.5, 3.0],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Grid: {len(combos)} combinations, risk={RISK:.0%}")

    rows = []
    for combo in tqdm(combos, desc="Optimizing"):
        p = dict(zip(keys, combo))
        try:
            s   = BreakBarHT(**p)
            res = run_mm(df, s, risk_per_trade=RISK)
            m   = res.metrics
            dd  = m.get("max_drawdown_pct", -999)
            if -30.0 <= dd <= -20.0:
                rows.append({**p,
                    "return": round(m.get("total_return_pct",0),2),
                    "dd":     round(dd,2),
                    "wr":     round(m.get("win_rate_pct",0),1),
                    "pf":     round(m.get("profit_factor",0),3),
                    "sharpe": round(m.get("sharpe_ratio",0),3),
                    "trades": int(m.get("total_trades",0))})
        except Exception:
            pass

    results = pd.DataFrame(rows).sort_values("return", ascending=False).reset_index(drop=True)
    print(f"\nПрошли фильтр DD [-30%,-20%]: {len(results)} конфигов")
    print(f"\nТОП 15:")
    print(results.head(15).to_string(index=False))
    results.to_csv("breakbar_ht_opt.csv", index=False)

    # График лучшего
    best = results.iloc[0]
    best_strat = BreakBarHT(
        min_size_pct=float(best.min_size_pct), min_bars_in=int(best.min_bars_in),
        rr_ratio=float(best.rr_ratio), ht_amplitude=int(best.ht_amplitude),
        ht_channel=float(best.ht_channel),
    )
    res_best = run_mm(df, best_strat, risk_per_trade=RISK, symbol="BTC/USDT", timeframe="4h")
    print(f"\n{res_best.report()}")

    _, ht_line = halftrend(df, amplitude=int(best.ht_amplitude),
                           channel_dev=float(best.ht_channel), atr_period=100)
    html = "breakbar_ht_best.html"
    plot_backtest(res_best, df,
        indicators={"HalfTrend": ht_line},
        title=(f"BTC/USDT 4h — BreakBar+HalfTrend BEST  "
               f"size={best.min_size_pct}%  bars={int(best.min_bars_in)}  "
               f"RR={best.rr_ratio}  amp={int(best.ht_amplitude)}  ch={best.ht_channel}  risk=6%"),
        save_html=html, show=False)
    print(f"\nChart → {html}")

    import subprocess
    subprocess.Popen(["open", "-a", "Google Chrome", html])
