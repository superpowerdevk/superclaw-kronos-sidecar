#!/usr/bin/env python3
"""SuperClaw Kronos sidecar.

Hosts the Kronos foundation model (https://github.com/shiyu-coder/Kronos) and
serves per-asset probabilistic forecasts to the superclaw-trader skill.

For each tracked asset it pulls recent 1h candles from Hyperliquid (keyless),
runs N single-path Kronos forecasts, and reports P(price touches the next round
level within the horizon) plus an expected range — the numbers the dashboard
renders as "BTC: $X, Y% odds going up to $Z in next 4h".

Endpoints:
  GET /health    -> {"ok": true}
  GET /forecast  -> cached forecasts for all assets (instant; refreshed in bg)

Config (env):
  KRONOS_MODEL=NeoQuasar/Kronos-small     KRONOS_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base
  DEVICE=cpu      INTERVAL=1h    HORIZON_HRS=4    LOOKBACK=512
  SAMPLE_COUNT=30 REFRESH_TTL=300                 MAX_CONTEXT=512
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
import json
import urllib.request
from datetime import datetime, timezone, timedelta

# Kronos model code is vendored at /app/kronos (see Dockerfile: git clone).
sys.path.append(os.environ.get("KRONOS_DIR", "/app/kronos"))

import pandas as pd  # noqa: E402
from fastapi import FastAPI  # noqa: E402

# ---- config -------------------------------------------------------------
MODEL_NAME = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER_NAME = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
DEVICE = os.environ.get("DEVICE", "cpu")
INTERVAL = os.environ.get("INTERVAL", "1h")
HORIZON_HRS = int(os.environ.get("HORIZON_HRS", "4"))
LOOKBACK = int(os.environ.get("LOOKBACK", "512"))
SAMPLE_COUNT = int(os.environ.get("SAMPLE_COUNT", "30"))
REFRESH_TTL = int(os.environ.get("REFRESH_TTL", "300"))
MAX_CONTEXT = int(os.environ.get("MAX_CONTEXT", "512"))

INTERVAL_MS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[INTERVAL] * 1000
PRED_LEN = max(1, round(HORIZON_HRS * 3600_000 / INTERVAL_MS))

HL = "https://api.hyperliquid.xyz/info"
# (coin sent to Hyperliquid, optional builder dex). GOLD trades on the xyz HIP-3 dex.
ASSETS = [
    ("BTC", "BTC", None),
    ("ETH", "ETH", None),
    ("BNB", "BNB", None),
    ("HYPE", "HYPE", None),
    ("SOL", "SOL", None),
    ("GOLD", "xyz:GOLD", "xyz"),
]

app = FastAPI()
CACHE: dict = {"status": "warming", "assets": [], "updated_at": None}
_PREDICTOR = None
_LOCK = threading.Lock()


# ---- data ---------------------------------------------------------------
def _hl_candles(coin: str, dex: str | None) -> pd.DataFrame | None:
    end = int(time.time() * 1000)
    start = end - (LOOKBACK + 8) * INTERVAL_MS
    req = {"type": "candleSnapshot",
           "req": {"coin": coin, "interval": INTERVAL, "startTime": start, "endTime": end}}
    if dex:
        req["req"]["dex"] = dex
    try:
        r = urllib.request.Request(HL, data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=20) as resp:
            rows = json.loads(resp.read().decode())
        if not isinstance(rows, list) or len(rows) < 64:
            return None
        rows = rows[-LOOKBACK:]
        df = pd.DataFrame({
            "timestamps": pd.to_datetime([int(c["t"]) for c in rows], unit="ms", utc=True),
            "open": [float(c["o"]) for c in rows],
            "high": [float(c["h"]) for c in rows],
            "low": [float(c["l"]) for c in rows],
            "close": [float(c["c"]) for c in rows],
            "volume": [float(c["v"]) for c in rows],
        })
        return df
    except Exception as e:
        print(f"[candles] {coin} failed: {e}", flush=True)
        return None


def _next_round(p: float) -> float:
    if p >= 10000: step = 1000.0
    elif p >= 1000: step = 100.0
    elif p >= 100: step = 10.0
    elif p >= 10: step = 1.0
    elif p >= 1: step = 0.1
    else: step = 0.01
    nxt = math.floor(p / step) * step + step
    return nxt if nxt > p else nxt + step


# ---- model --------------------------------------------------------------
def _load_model():
    global _PREDICTOR
    from model import Kronos, KronosTokenizer, KronosPredictor
    tok = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    mdl = Kronos.from_pretrained(MODEL_NAME)
    _PREDICTOR = KronosPredictor(mdl, tok, device=DEVICE, max_context=MAX_CONTEXT)
    print(f"[model] loaded {MODEL_NAME} on {DEVICE}", flush=True)


def _forecast_all() -> list:
    """Run SAMPLE_COUNT single-path forecasts for every asset, aggregate to odds."""
    loaded, meta = [], []
    for label, coin, dex in ASSETS:
        df = _hl_candles(coin, dex)
        if df is None or len(df) < 64:
            meta.append((label, None, None)); continue
        hist = df.iloc[-min(len(df), MAX_CONTEXT):].reset_index(drop=True)
        spot = float(hist["close"].iloc[-1])
        loaded.append((label, hist, spot))

    out_by_label: dict = {}
    if loaded:
        x_list = [h[["open", "high", "low", "close", "volume"]] for _, h, _ in loaded]
        xts_list = [h["timestamps"] for _, h, _ in loaded]
        last_ts = [h["timestamps"].iloc[-1] for _, h, _ in loaded]
        yts_list = [pd.Series([t + timedelta(milliseconds=INTERVAL_MS * (k + 1))
                               for k in range(PRED_LEN)]) for t in last_ts]
        agg = {lab: {"hits": 0, "highs": [], "lows": [], "ends": []} for lab, _, _ in loaded}
        for _ in range(SAMPLE_COUNT):
            try:
                preds = _PREDICTOR.predict_batch(
                    df_list=x_list, x_timestamp_list=xts_list, y_timestamp_list=yts_list,
                    pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False)
            except Exception as e:
                print(f"[predict] batch failed: {e}", flush=True)
                break
            for (lab, _, spot), pdf in zip(loaded, preds):
                tgt = _next_round(spot)
                hi = float(pdf["high"].max()); lo = float(pdf["low"].min())
                end = float(pdf["close"].iloc[-1])
                a = agg[lab]
                a["hits"] += 1 if hi >= tgt else 0
                a["highs"].append(hi); a["lows"].append(lo); a["ends"].append(end)
        for lab, _, spot in loaded:
            a = agg[lab]; n = len(a["ends"])
            if n == 0:
                out_by_label[lab] = None; continue
            tgt = _next_round(spot)
            out_by_label[lab] = {
                "spot": round(spot, 4),
                "target": round(tgt, 4),
                "prob_pct": round(100 * a["hits"] / n),
                "prob_up_pct": round(100 * sum(1 for e in a["ends"] if e > spot) / n),
                "exp_high": round(sum(a["highs"]) / n, 4),
                "exp_low": round(sum(a["lows"]) / n, 4),
                "exp_close": round(sum(a["ends"]) / n, 4),
                "horizon_hrs": HORIZON_HRS,
            }

    result = []
    for label, _, _ in ASSETS:
        d = out_by_label.get(label)
        result.append({"asset": label, **d} if d else {"asset": label, "spot": None})
    return result


def _refresher():
    while True:
        try:
            res = _forecast_all()
            with _LOCK:
                CACHE["assets"] = res
                CACHE["updated_at"] = datetime.now(timezone.utc).isoformat()
                CACHE["status"] = "ok"
            print(f"[refresh] updated {len([a for a in res if a.get('spot')])} assets", flush=True)
        except Exception as e:
            print(f"[refresh] error: {e}", flush=True)
        time.sleep(REFRESH_TTL)


@app.on_event("startup")
def _startup():
    def boot():
        _load_model()
        _refresher()
    threading.Thread(target=boot, daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True, "status": CACHE["status"], "model": MODEL_NAME}


@app.get("/forecast")
def forecast():
    with _LOCK:
        return dict(CACHE) | {"interval": INTERVAL, "horizon_hrs": HORIZON_HRS,
                              "sample_count": SAMPLE_COUNT, "model": MODEL_NAME}
