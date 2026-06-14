#!/usr/bin/env python3
"""SuperClaw Kronos sidecar.

Hosts the Kronos foundation model (https://github.com/shiyu-coder/Kronos) and serves
per-asset probabilistic forecasts to the superclaw-trader skill.

For each asset it pulls recent 1h candles from Hyperliquid (keyless), runs N single-path
Kronos forecasts over the long horizon, and reports — at both a short and long horizon —
the probability of touching the next ~1% round level, plus directional conviction,
expected range, a forecast path (for a sparkline), and Kronos-derived stop/target levels.

Endpoints:
  GET /health    -> {"ok": true, "status": "warming|ok", "model": ...}
  GET /forecast  -> cached forecasts for all assets (instant; refreshed in background)
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

sys.path.append(os.environ.get("KRONOS_DIR", "/app/kronos"))

import pandas as pd  # noqa: E402
from fastapi import FastAPI  # noqa: E402

# ---- config -------------------------------------------------------------
MODEL_NAME = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER_NAME = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
DEVICE = os.environ.get("DEVICE", "cpu")
INTERVAL = os.environ.get("INTERVAL", "1h")
HORIZON_SHORT = int(os.environ.get("HORIZON_SHORT", "4"))
HORIZON_LONG = int(os.environ.get("HORIZON_LONG", "24"))
LOOKBACK = int(os.environ.get("LOOKBACK", "512"))
SAMPLE_COUNT = int(os.environ.get("SAMPLE_COUNT", "20"))
REFRESH_TTL = int(os.environ.get("REFRESH_TTL", "600"))
MAX_CONTEXT = int(os.environ.get("MAX_CONTEXT", "512"))

INTERVAL_MS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[INTERVAL] * 1000
PRED_LEN = max(1, round(HORIZON_LONG * 3600_000 / INTERVAL_MS))
SHORT_LEN = max(1, round(HORIZON_SHORT * 3600_000 / INTERVAL_MS))

HL = "https://api.hyperliquid.xyz/info"
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


# ---- helpers ------------------------------------------------------------
def _clamp(p) -> int:
    """Keep displayed odds in 1-99 so they never read as fake 0%/100%."""
    return max(1, min(99, int(round(p))))


def _downsample(lst, k):
    if len(lst) <= k:
        return lst
    step = (len(lst) - 1) / (k - 1)
    return [lst[round(i * step)] for i in range(k)]


def _next_round(p: float) -> float:
    """Next 'nice' level ~1% above spot, snapped to a clean 1/2/5/10 number — close
    enough to be reachable in the horizon, far enough that odds aren't pinned."""
    if p <= 0:
        return 0.0
    raw = p * 0.01
    mag = 10 ** math.floor(math.log10(raw))
    step = next((mag * m for m in (1, 2, 5, 10) if mag * m >= raw), mag * 10)
    return (math.floor(p / step) + 1) * step


def _hl_candles(coin: str, dex: str | None):
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
        return pd.DataFrame({
            "timestamps": pd.to_datetime([int(c["t"]) for c in rows], unit="ms", utc=True),
            "open": [float(c["o"]) for c in rows],
            "high": [float(c["h"]) for c in rows],
            "low": [float(c["l"]) for c in rows],
            "close": [float(c["c"]) for c in rows],
            "volume": [float(c["v"]) for c in rows],
        })
    except Exception as e:
        print(f"[candles] {coin} failed: {e}", flush=True)
        return None


# ---- model --------------------------------------------------------------
def _load_model():
    global _PREDICTOR
    from model import Kronos, KronosTokenizer, KronosPredictor
    tok = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    mdl = Kronos.from_pretrained(MODEL_NAME)
    _PREDICTOR = KronosPredictor(mdl, tok, device=DEVICE, max_context=MAX_CONTEXT)
    print(f"[model] loaded {MODEL_NAME} on {DEVICE}", flush=True)


def _forecast_all() -> list:
    loaded = []
    for label, coin, dex in ASSETS:
        df = _hl_candles(coin, dex)
        if df is None or len(df) < 64:
            continue
        hist = df.iloc[-min(len(df), MAX_CONTEXT):].reset_index(drop=True)
        loaded.append((label, hist, float(hist["close"].iloc[-1])))

    out: dict = {}
    if loaded:
        x_list = [h[["open", "high", "low", "close", "volume"]] for _, h, _ in loaded]
        xts = [h["timestamps"] for _, h, _ in loaded]
        last = [h["timestamps"].iloc[-1] for _, h, _ in loaded]
        yts = [pd.Series([t + timedelta(milliseconds=INTERVAL_MS * (k + 1)) for k in range(PRED_LEN)])
               for t in last]
        agg = {lab: {"hs": 0, "hl": 0, "up": 0, "hi": [], "lo": [], "end": [], "paths": []}
               for lab, _, _ in loaded}
        runs = 0
        for _ in range(SAMPLE_COUNT):
            try:
                preds = _PREDICTOR.predict_batch(
                    df_list=x_list, x_timestamp_list=xts, y_timestamp_list=yts,
                    pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False)
            except Exception as e:
                print(f"[predict] batch failed: {e}", flush=True)
                break
            runs += 1
            for (lab, _, spot), pdf in zip(loaded, preds):
                tgt = _next_round(spot)
                highs = pdf["high"].tolist(); lows = pdf["low"].tolist(); closes = pdf["close"].tolist()
                a = agg[lab]
                a["hs"] += 1 if max(highs[:SHORT_LEN]) >= tgt else 0
                a["hl"] += 1 if max(highs) >= tgt else 0
                a["up"] += 1 if closes[-1] > spot else 0
                a["hi"].append(max(highs)); a["lo"].append(min(lows)); a["end"].append(closes[-1])
                a["paths"].append(closes)
        for lab, _, spot in loaded:
            a = agg[lab]; n = runs
            if n == 0 or not a["paths"]:
                out[lab] = None; continue
            tgt = _next_round(spot)
            mean_path = [sum(p[i] for p in a["paths"]) / len(a["paths"]) for i in range(PRED_LEN)]
            out[lab] = {
                "spot": round(spot, 4),
                "target": round(tgt, 4),
                "prob_short": _clamp(100 * a["hs"] / n),
                "prob_long": _clamp(100 * a["hl"] / n),
                "prob_up_pct": _clamp(100 * a["up"] / n),
                "exp_high": round(sum(a["hi"]) / n, 4),
                "exp_low": round(sum(a["lo"]) / n, 4),
                "exp_close": round(sum(a["end"]) / n, 4),
                "suggested_stop": round(sum(a["lo"]) / n, 4),
                "suggested_tp": round(sum(a["hi"]) / n, 4),
                "horizon_short": HORIZON_SHORT,
                "horizon_long": HORIZON_LONG,
                "path": [round(x, 4) for x in _downsample(mean_path, 16)],
            }

    result = []
    for label, _, _ in ASSETS:
        d = out.get(label)
        result.append({"asset": label, **d} if d else {"asset": label, "spot": None})
    return result


def _refresher():
    while True:
        try:
            res = _forecast_all()
            with _LOCK:
                CACHE["assets"] = res
                CACHE["updated_at"] = datetime.now(timezone.utc).isoformat()
                if any(a.get("spot") for a in res):
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
        return dict(CACHE) | {"interval": INTERVAL, "horizon_short": HORIZON_SHORT,
                              "horizon_long": HORIZON_LONG, "sample_count": SAMPLE_COUNT,
                              "model": MODEL_NAME}
