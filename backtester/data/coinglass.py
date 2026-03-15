"""
Coinglass API — hourly aggregated liquidation data.

Free API key: https://coinglass.com → Profile → API
Set via: export COINGLASS_API_KEY=your_key
Or pass directly: CoinglassDownloader(api_key="...")
"""

import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

_BASE = "https://open-api.coinglass.com/public/v2"


class CoinglassDownloader:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("COINGLASS_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Coinglass API key required.\n"
                "Get free key at https://coinglass.com → Profile → API\n"
                "Set: export COINGLASS_API_KEY=your_key"
            )
        self._session = requests.Session()
        self._session.headers.update({
            "coinglassSecret": self.api_key,
            "Content-Type": "application/json",
        })

    def fetch_liquidations(
        self,
        symbol: str,
        since: datetime,
        until: datetime,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """
        Fetch hourly aggregated liquidation volumes.

        Parameters
        ----------
        symbol    : "BTC" or "ETH" (without USDT)
        since     : start datetime (UTC)
        until     : end datetime (UTC)
        timeframe : "1h", "4h", "12h", "1d"

        Returns
        -------
        pd.DataFrame with DatetimeIndex (UTC) and columns:
          liq_long   — liquidated long volume in USD (price dropped → longs squeezed)
          liq_short  — liquidated short volume in USD (price rose → shorts squeezed)
          liq_total  — sum
        """
        sym = symbol.upper().replace("/USDT", "").replace("USDT", "")

        # Coinglass returns full history for the symbol at once
        tf_map = {"1h": "h1", "4h": "h4", "12h": "h12", "1d": "d1"}
        tf = tf_map.get(timeframe, timeframe)

        r = self._session.get(
            f"{_BASE}/liquidation_chart",
            params={"symbol": sym, "time_type": tf},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()

        if data.get("code") != "0" or not data.get("data"):
            raise RuntimeError(f"Coinglass error: {data.get('msg', 'unknown')}")

        d = data["data"]
        df = pd.DataFrame({
            "ts": pd.to_datetime(d["dateList"], unit="ms", utc=True),
            "liq_long":  [float(x) for x in d["longList"]],
            "liq_short": [float(x) for x in d["shortList"]],
        })
        df = df.set_index("ts").sort_index()
        df["liq_total"] = df["liq_long"] + df["liq_short"]

        return df.loc[since:until]
