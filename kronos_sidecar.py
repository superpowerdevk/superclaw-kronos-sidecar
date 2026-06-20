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

import os
import statistics
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
_MODEL_LOCK = threading.Lock()  # serialize Kronos access (batch refresher + on-demand symbol calls)

# ---- generic on-demand (price-forecast skill) ---------------------------
import math  # noqa: E402
import urllib.parse  # noqa: E402

YF_SEARCH = "https://query1.finance.yahoo.com/v1/finance/search"
YF_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
# Yahoo blocks cloud IPs without a browser UA.
YF_HEADERS = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/124.0 Safari/537.36")}
GEN_INTERVAL = os.environ.get("GEN_INTERVAL", "1d")      # daily candles = uniform across asset classes
GEN_RANGE = os.environ.get("GEN_RANGE", "2y")            # ~512 daily candles of lookback
GEN_HORIZON = int(os.environ.get("GEN_HORIZON", "5"))    # forecast N candles ahead (≈1 trading week)
GEN_SAMPLES = int(os.environ.get("GEN_SAMPLES", "20"))   # forecast paths per symbol
SYMBOL_TTL = int(os.environ.get("SYMBOL_TTL", "600"))    # per-symbol forecast cache (s)
GEN_INTERVAL_S = {"1d": 86400, "1h": 3600, "1wk": 604800}.get(GEN_INTERVAL, 86400)

SYM_CACHE: dict = {}   # lowercased query -> resolved meta
FC_CACHE: dict = {}    # symbol -> (epoch, payload)


# ---- helpers ------------------------------------------------------------
def _clamp(p) -> int:
    """Keep displayed odds in 1-99 so they never read as fake 0%/100%."""
    return max(1, min(99, int(round(p))))


def _downsample(lst, k):
    if len(lst) <= k:
        return lst
    step = (len(lst) - 1) / (k - 1)
    return [lst[round(i * step)] for i in range(k)]



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
        agg = {lab: {"ups": 0, "upl": 0, "hi": [], "lo": [], "end": [], "paths": []}
               for lab, _, _ in loaded}
        runs = 0
        for _ in range(SAMPLE_COUNT):
            try:
                with _MODEL_LOCK:
                    preds = _PREDICTOR.predict_batch(
                        df_list=x_list, x_timestamp_list=xts, y_timestamp_list=yts,
                        pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False)
            except Exception as e:
                print(f"[predict] batch failed: {e}", flush=True)
                break
            runs += 1
            for (lab, _, spot), pdf in zip(loaded, preds):
                highs = pdf["high"].tolist(); lows = pdf["low"].tolist(); closes = pdf["close"].tolist()
                a = agg[lab]
                a["ups"] += 1 if closes[SHORT_LEN - 1] > spot else 0   # up at the short mark
                a["upl"] += 1 if closes[-1] > spot else 0              # up at the long mark
                a["hi"].append(max(highs)); a["lo"].append(min(lows))
                a["end"].append(closes[-1]); a["paths"].append(closes)
        for lab, _, spot in loaded:
            a = agg[lab]; n = runs
            if n == 0 or not a["paths"]:
                out[lab] = None; continue
            # Conviction = how often paths CLOSE up vs down (doesn't saturate like touch-odds).
            prob_up_short = _clamp(100 * a["ups"] / n)
            prob_up_long = _clamp(100 * a["upl"] / n)
            direction = "up" if prob_up_long >= 50 else "down"
            exp_high = sum(a["hi"]) / n; exp_low = sum(a["lo"]) / n
            exp_close = statistics.median(a["end"])  # median agrees with direction even under skew
            stop, tp = (exp_low, exp_high) if direction == "up" else (exp_high, exp_low)
            mean_path = [sum(p[i] for p in a["paths"]) / len(a["paths"]) for i in range(PRED_LEN)]
            out[lab] = {
                "spot": round(spot, 4),
                "direction": direction,
                "prob_up_short": prob_up_short,
                "prob_up_long": prob_up_long,
                "exp_high": round(exp_high, 4),
                "exp_low": round(exp_low, 4),
                "exp_close": round(exp_close, 4),
                "suggested_stop": round(stop, 4),
                "suggested_tp": round(tp, 4),
                "horizon_short": HORIZON_SHORT,
                "horizon_long": HORIZON_LONG,
                "path": [round(x, 4) for x in _downsample(mean_path, 16)],
            }

    result = []
    for label, _, _ in ASSETS:
        d = out.get(label)
        result.append({"asset": label, **d} if d else {"asset": label, "spot": None})
    return result


def _yf_get(url: str, params: dict):
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers=YF_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def resolve_symbol(q: str):
    """Natural-language or ticker -> best Yahoo symbol. Covers stocks, ETFs, indices,
    FX, commodities, and crypto through one endpoint. Cached per query."""
    q = (q or "").strip()
    if not q:
        return None
    key = q.lower()
    if key in SYM_CACHE:
        return SYM_CACHE[key]
    try:
        data = _yf_get(YF_SEARCH, {"q": q, "quotesCount": 6, "newsCount": 0})
        quotes = [x for x in data.get("quotes", []) if x.get("symbol")]
        if not quotes:
            return None
        exact = next((x for x in quotes if x["symbol"].lower() == key), None)
        pick = exact or quotes[0]
        meta = {
            "symbol": pick["symbol"],
            "name": pick.get("shortname") or pick.get("longname") or pick["symbol"],
            "type": pick.get("quoteType", "EQUITY"),
            "exchange": pick.get("exchDisp") or pick.get("exchange", ""),
        }
        SYM_CACHE[key] = meta
        return meta
    except Exception as e:
        print(f"[resolve] {q!r} failed: {e}", flush=True)
        return None


def _yf_candles(symbol: str):
    """Daily OHLCV from Yahoo's keyless chart endpoint. None if too thin."""
    try:
        data = _yf_get(YF_CHART.format(symbol=urllib.parse.quote(symbol, safe="")),
                       {"range": GEN_RANGE, "interval": GEN_INTERVAL})
        res = data["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        rows = []
        for i, t in enumerate(ts):
            o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
            v = (q.get("volume") or [None] * len(ts))[i]
            if None in (o, h, l, c):
                continue
            rows.append((t, o, h, l, c, float(v or 0.0)))
        if len(rows) < 64:
            return None
        rows = rows[-LOOKBACK:]
        return pd.DataFrame({
            "timestamps": pd.to_datetime([r[0] for r in rows], unit="s", utc=True),
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
        })
    except Exception as e:
        print(f"[candles-yf] {symbol} failed: {e}", flush=True)
        return None


def _forecast_symbol(meta: dict):
    """Run Kronos GEN_SAMPLES times over daily candles for one resolved symbol."""
    sym = meta["symbol"]
    df = _yf_candles(sym)
    if df is None or len(df) < 64:
        return None
    hist = df.iloc[-min(len(df), MAX_CONTEXT):].reset_index(drop=True)
    spot = float(hist["close"].iloc[-1])
    x = hist[["open", "high", "low", "close", "volume"]]
    xts = hist["timestamps"]
    last = hist["timestamps"].iloc[-1]
    yts = pd.Series([last + timedelta(seconds=GEN_INTERVAL_S * (k + 1)) for k in range(GEN_HORIZON)])

    ups = 0
    hi: list = []
    lo: list = []
    end: list = []
    paths: list = []
    runs = 0
    for _ in range(GEN_SAMPLES):
        try:
            with _MODEL_LOCK:
                preds = _PREDICTOR.predict_batch(
                    df_list=[x], x_timestamp_list=[xts], y_timestamp_list=[yts],
                    pred_len=GEN_HORIZON, T=1.0, top_p=0.9, sample_count=1, verbose=False)
        except Exception as e:
            print(f"[predict-sym] {sym} failed: {e}", flush=True)
            break
        runs += 1
        pdf = preds[0]
        closes = pdf["close"].tolist()
        highs = pdf["high"].tolist()
        lows = pdf["low"].tolist()
        ups += 1 if closes[-1] > spot else 0
        hi.append(max(highs))
        lo.append(min(lows))
        end.append(closes[-1])
        paths.append(closes)

    if runs == 0 or not paths:
        return None

    prob_up = _clamp(100 * ups / runs)
    exp_close = statistics.median(end)
    exp_high = sum(hi) / runs
    exp_low = sum(lo) / runs
    mean_path = [sum(p[i] for p in paths) / len(paths) for i in range(GEN_HORIZON)]

    # "Meaningful move" target = ~1σ over the horizon, from realized daily vol.
    rets = hist["close"].pct_change().dropna()
    daily_vol = float(rets.std()) if len(rets) > 5 else 0.02
    move_pct = max(0.01, daily_vol * math.sqrt(GEN_HORIZON))
    up_t = spot * (1 + move_pct)
    dn_t = spot * (1 - move_pct)
    odds_up = _clamp(100 * sum(1 for v in hi if v >= up_t) / runs)
    odds_dn = _clamp(100 * sum(1 for v in lo if v <= dn_t) / runs)

    return {
        "symbol": sym,
        "name": meta["name"],
        "type": meta["type"],
        "exchange": meta["exchange"],
        "spot": round(spot, 4),
        "direction": "up" if prob_up >= 50 else "down",
        "prob_up": prob_up,
        "exp_close": round(exp_close, 4),
        "exp_high": round(exp_high, 4),
        "exp_low": round(exp_low, 4),
        "exp_change_pct": round((exp_close / spot - 1) * 100, 2),
        "move_pct": round(move_pct * 100, 2),
        "odds_up_move": odds_up,
        "odds_dn_move": odds_dn,
        "horizon_days": GEN_HORIZON,
        "interval": GEN_INTERVAL,
        "samples": runs,
        "path": [round(v, 4) for v in _downsample(mean_path, 16)],
    }


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


@app.get("/forecast/symbol")
def forecast_symbol(q: str):
    """On-demand forecast for ANY asset (stocks, ETFs, indices, FX, commodities, crypto).
    Resolves q -> Yahoo symbol -> daily candles -> Kronos. Per-symbol TTL cache."""
    if _PREDICTOR is None:
        return {"ok": False, "status": "warming", "query": q}
    meta = resolve_symbol(q)
    if not meta:
        return {"ok": False, "error": "could not resolve a tradable symbol", "query": q}
    sym = meta["symbol"]
    now = time.time()
    hit = FC_CACHE.get(sym)
    if hit and now - hit[0] < SYMBOL_TTL:
        return {"ok": True, "cached": True, "query": q, **hit[1]}
    res = _forecast_symbol(meta)
    if not res:
        return {"ok": False, "error": "no candle data or forecast failed",
                "query": q, "symbol": sym, "name": meta["name"]}
    FC_CACHE[sym] = (now, res)
    return {"ok": True, "cached": False, "query": q, **res}
