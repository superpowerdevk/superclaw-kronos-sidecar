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
import re  # noqa: E402
import urllib.error  # noqa: E402
import urllib.parse  # noqa: E402

HTTP_UA = {"User-Agent": "superclaw-price-forecast/1.0 (+https://superpower.io; admin@superpower.io)"}
BROWSER_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
STOOQ = "https://stooq.com/q/d/l/"
CG = "https://api.coingecko.com/api/v3"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
FMP = "https://financialmodelingprep.com/stable/historical-price-eod/full"

GEN_INTERVAL = os.environ.get("GEN_INTERVAL", "1d")
GEN_HORIZON = int(os.environ.get("GEN_HORIZON", "5"))    # forecast N candles ahead (≈1 trading week)
GEN_SAMPLES = int(os.environ.get("GEN_SAMPLES", "20"))   # forecast paths per symbol
SYMBOL_TTL = int(os.environ.get("SYMBOL_TTL", "600"))    # per-symbol forecast cache (s)
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "512"))
GEN_INTERVAL_S = {"1d": 86400, "1h": 3600, "1wk": 604800}.get(GEN_INTERVAL, 86400)

# crypto name/ticker -> Hyperliquid coin (primary crypto source — proven from this box)
CRYPTO_ALIASES = {
    "btc": "BTC", "bitcoin": "BTC", "xbt": "BTC",
    "eth": "ETH", "ethereum": "ETH", "ether": "ETH",
    "sol": "SOL", "solana": "SOL", "bnb": "BNB", "binance coin": "BNB", "binancecoin": "BNB",
    "xrp": "XRP", "ripple": "XRP", "doge": "DOGE", "dogecoin": "DOGE",
    "ada": "ADA", "cardano": "ADA", "avax": "AVAX", "avalanche": "AVAX",
    "link": "LINK", "chainlink": "LINK", "dot": "DOT", "polkadot": "DOT",
    "matic": "MATIC", "polygon": "MATIC", "pol": "MATIC", "ltc": "LTC", "litecoin": "LTC",
    "trx": "TRX", "tron": "TRX", "shib": "SHIB", "shiba": "SHIB", "shiba inu": "SHIB",
    "uni": "UNI", "uniswap": "UNI", "atom": "ATOM", "cosmos": "ATOM", "near": "NEAR",
    "apt": "APT", "aptos": "APT", "arb": "ARB", "arbitrum": "ARB", "op": "OP", "optimism": "OP",
    "sui": "SUI", "sei": "SEI", "ton": "TON", "toncoin": "TON",
    "hype": "HYPE", "hyperliquid": "HYPE", "pepe": "PEPE", "wif": "WIF", "dogwifhat": "WIF",
    "bonk": "BONK", "jup": "JUP", "jupiter": "JUP", "ena": "ENA", "ethena": "ENA",
    "tia": "TIA", "celestia": "TIA", "inj": "INJ", "injective": "INJ",
    "fil": "FIL", "filecoin": "FIL", "fartcoin": "FARTCOIN", "ldo": "LDO", "lido": "LDO",
}

# tradfi natural language -> (stooq symbol, display name, type)
TRADFI_ALIASES = {
    "gold": ("xauusd", "Gold (XAU/USD)", "COMMODITY"), "xau": ("xauusd", "Gold (XAU/USD)", "COMMODITY"),
    "silver": ("xagusd", "Silver (XAG/USD)", "COMMODITY"), "xag": ("xagusd", "Silver (XAG/USD)", "COMMODITY"),
    "oil": ("cl.f", "WTI Crude Oil", "COMMODITY"), "crude": ("cl.f", "WTI Crude Oil", "COMMODITY"),
    "wti": ("cl.f", "WTI Crude Oil", "COMMODITY"), "crude oil": ("cl.f", "WTI Crude Oil", "COMMODITY"),
    "brent": ("cb.f", "Brent Crude Oil", "COMMODITY"),
    "natural gas": ("ng.f", "Natural Gas", "COMMODITY"), "natgas": ("ng.f", "Natural Gas", "COMMODITY"),
    "copper": ("hg.f", "Copper", "COMMODITY"), "platinum": ("pl.f", "Platinum", "COMMODITY"),
    "palladium": ("pa.f", "Palladium", "COMMODITY"),
    "sp500": ("^spx", "S&P 500", "INDEX"), "s&p 500": ("^spx", "S&P 500", "INDEX"),
    "s&p500": ("^spx", "S&P 500", "INDEX"), "s&p": ("^spx", "S&P 500", "INDEX"),
    "spx": ("^spx", "S&P 500", "INDEX"), "gspc": ("^spx", "S&P 500", "INDEX"),
    "nasdaq": ("^ndq", "Nasdaq Composite", "INDEX"), "nasdaq composite": ("^ndq", "Nasdaq Composite", "INDEX"),
    "ixic": ("^ndq", "Nasdaq Composite", "INDEX"),
    "nasdaq 100": ("^ndx", "Nasdaq 100", "INDEX"), "ndx": ("^ndx", "Nasdaq 100", "INDEX"),
    "dow": ("^dji", "Dow Jones", "INDEX"), "dow jones": ("^dji", "Dow Jones", "INDEX"),
    "djia": ("^dji", "Dow Jones", "INDEX"), "dji": ("^dji", "Dow Jones", "INDEX"),
    "russell": ("^rut", "Russell 2000", "INDEX"), "russell 2000": ("^rut", "Russell 2000", "INDEX"),
    "vix": ("^vix", "Volatility Index (VIX)", "INDEX"),
    "ftse": ("^ukx", "FTSE 100", "INDEX"), "ftse 100": ("^ukx", "FTSE 100", "INDEX"),
    "dax": ("^dax", "DAX", "INDEX"), "nikkei": ("^nkx", "Nikkei 225", "INDEX"),
    "^gspc": ("^spx", "S&P 500", "INDEX"), "^ixic": ("^ndq", "Nasdaq Composite", "INDEX"),
    "^ndx": ("^ndx", "Nasdaq 100", "INDEX"), "^rut": ("^rut", "Russell 2000", "INDEX"),
}
_CCY = {"usd", "eur", "gbp", "jpy", "chf", "cad", "aud", "nzd", "cny", "hkd",
        "sgd", "sek", "nok", "mxn", "zar", "try", "inr", "krw", "brl"}

SYM_CACHE: dict = {}   # lowercased query -> resolved meta
FC_CACHE: dict = {}    # symbol -> (epoch, payload)
SEC_MAP = {"by_ticker": None, "items": None, "ts": 0.0}  # cached SEC name->ticker index


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


def _http(url: str, headers: dict | None = None, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers=headers or HTTP_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _df_from_rows(rows):
    """rows: list of (epoch_seconds, o, h, l, c, v). Returns a candle DataFrame or None."""
    if not rows or len(rows) < 64:
        return None
    rows = rows[-LOOKBACK_DAYS:]
    return pd.DataFrame({
        "timestamps": pd.to_datetime([r[0] for r in rows], unit="s", utc=True),
        "open": [r[1] for r in rows],
        "high": [r[2] for r in rows],
        "low": [r[3] for r in rows],
        "close": [r[4] for r in rows],
        "volume": [r[5] for r in rows],
    })


# ---- candle sources (all keyless, datacenter-friendly) ------------------
def _stooq_candles(sym: str):
    """Daily OHLCV CSV from Stooq. Covers US/global equities, ETFs, indices, FX, commodities.
    Keyless but rate-limited on shared datacenter IPs."""
    try:
        raw = _http(STOOQ + "?" + urllib.parse.urlencode({"s": sym, "i": "d"}),
                    headers=BROWSER_UA).decode("utf-8", "replace")
        lines = raw.strip().splitlines()
        if len(lines) < 2 or not lines[0].lower().startswith("date"):
            print(f"[stooq] {sym} non-CSV body: {raw[:120]!r}", flush=True)
            return None
        rows = []
        for ln in lines[1:]:
            p = ln.split(",")
            if len(p) < 5:
                continue
            try:
                ep = int(pd.Timestamp(p[0], tz="UTC").timestamp())
                o, h, l, c = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                v = float(p[5]) if len(p) > 5 and p[5] not in ("", "N/D") else 0.0
            except Exception:
                continue
            rows.append((ep, o, h, l, c, v))
        return _df_from_rows(rows)
    except Exception as e:
        print(f"[stooq] {sym} failed: {e}", flush=True)
        return None


_FMP_INDEX = {"^SPX": "^GSPC", "^NDQ": "^IXIC", "^DJI": "^DJI", "^RUT": "^RUT",
              "^NDX": "^NDX", "^VIX": "^VIX", "^UKX": "^FTSE", "^DAX": "^GDAXI", "^NKX": "^N225"}
_FMP_CMDTY = {"XAUUSD": "GCUSD", "XAGUSD": "SIUSD", "CL.F": "CLUSD", "CB.F": "BZUSD",
              "NG.F": "NGUSD", "HG.F": "HGUSD", "PL.F": "PLUSD", "PA.F": "PAUSD"}


def _fmp_symbol(meta: dict) -> str:
    t, s = meta.get("type", ""), meta["symbol"]
    if t == "INDEX":
        return _FMP_INDEX.get(s, s)
    if t == "COMMODITY":
        return _FMP_CMDTY.get(s, s)
    return s  # equity/ETF ticker, or 6-letter FX pair (EURUSD)


def _fmp_candles(meta: dict):
    """Daily OHLCV from Financial Modeling Prep stable EOD API (not a scrape -> no IP block).
    Active only when FMP_API_KEY is set."""
    key = os.environ.get("FMP_API_KEY", "").strip()
    if not key:
        return None
    sym = _fmp_symbol(meta)
    try:
        frm = (datetime.now(timezone.utc) - timedelta(days=int(LOOKBACK_DAYS * 1.6))).strftime("%Y-%m-%d")
        url = FMP + "?" + urllib.parse.urlencode({"symbol": sym, "from": frm, "apikey": key})
        data = json.loads(_http(url).decode())
        hist = data.get("historical") if isinstance(data, dict) else data  # stable=list, legacy=dict
        if not isinstance(hist, list) or len(hist) < 64:
            return None
        rows = []
        for r in hist:
            try:
                ep = int(pd.Timestamp(r["date"], tz="UTC").timestamp())
                rows.append((ep, float(r["open"]), float(r["high"]),
                             float(r["low"]), float(r["close"]), float(r.get("volume") or 0)))
            except Exception:
                continue
        rows.sort()  # FMP returns newest-first; sort to oldest-first
        return _df_from_rows(rows)
    except Exception as e:
        print(f"[fmp] {sym} failed: {e}", flush=True)
        return None


def _hl_daily(coin: str):
    """Daily OHLCV from Hyperliquid (keyless; already proven from this box)."""
    end = int(time.time() * 1000)
    start = end - (LOOKBACK_DAYS + 8) * 86400_000
    req = {"type": "candleSnapshot",
           "req": {"coin": coin, "interval": "1d", "startTime": start, "endTime": end}}
    try:
        r = urllib.request.Request(HL, data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=20) as resp:
            arr = json.loads(resp.read().decode())
        if not isinstance(arr, list) or len(arr) < 64:
            return None
        rows = [(int(c["t"]) // 1000, float(c["o"]), float(c["h"]),
                 float(c["l"]), float(c["c"]), float(c["v"])) for c in arr]
        return _df_from_rows(rows)
    except Exception as e:
        print(f"[hl-daily] {coin} failed: {e}", flush=True)
        return None


def _cg_search(q: str):
    """CoinGecko coin search -> {id, symbol, name} of the top market-cap match, or None."""
    try:
        data = json.loads(_http(CG + "/search?" + urllib.parse.urlencode({"query": q})).decode())
        coins = data.get("coins") or []
        if not coins:
            return None
        coins.sort(key=lambda c: (c.get("market_cap_rank") is None, c.get("market_cap_rank") or 1e9))
        c = coins[0]
        return {"id": c["id"], "symbol": (c.get("symbol") or "").upper(), "name": c.get("name") or c["id"]}
    except Exception as e:
        print(f"[cg-search] {q!r} failed: {e}", flush=True)
        return None


def _cg_ohlc(coin_id: str):
    """CoinGecko OHLC fallback for coins not on Hyperliquid (coarse candles; no volume)."""
    try:
        url = CG + f"/coins/{urllib.parse.quote(coin_id)}/ohlc?" + urllib.parse.urlencode(
            {"vs_currency": "usd", "days": "365"})
        arr = json.loads(_http(url).decode())
        if not isinstance(arr, list) or len(arr) < 64:
            return None
        rows = [(int(x[0]) // 1000, float(x[1]), float(x[2]), float(x[3]), float(x[4]), 0.0) for x in arr]
        return _df_from_rows(rows)
    except Exception as e:
        print(f"[cg-ohlc] {coin_id} failed: {e}", flush=True)
        return None


# ---- name -> ticker (SEC, free US equities directory) -------------------
def _sec_index():
    now = time.time()
    if SEC_MAP["items"] is not None and now - SEC_MAP["ts"] < 86400:
        return SEC_MAP["items"]
    try:
        data = json.loads(_http(SEC_TICKERS).decode())
        items, tickers = [], set()
        for v in data.values():
            t = (v.get("ticker") or "").upper()
            title = v.get("title") or ""
            if t and title:
                items.append((t, title, title.lower()))
                tickers.add(t)
        SEC_MAP.update(items=items, by_ticker=tickers, ts=now)
    except Exception as e:
        print(f"[sec] index load failed: {e}", flush=True)
        SEC_MAP.update(items=[], by_ticker=set(), ts=now)
    return SEC_MAP["items"]


def _sec_tickers() -> set:
    _sec_index()
    return SEC_MAP["by_ticker"] or set()


_CORP_SUFFIX = (" inc", " inc.", " incorporated", " corp", " corp.", " corporation",
                " company", " co", " co.", " ltd", " ltd.", " plc", " holdings",
                " group", " the", ",")


def _norm_name(s: str) -> str:
    s = s.lower().strip()
    for suf in _CORP_SUFFIX:
        s = s.replace(suf, " ")
    return " ".join(s.split())


def _sec_ticker(name: str):
    """Best-effort company-name -> US ticker via the SEC directory."""
    items = _sec_index()
    if not items:
        return None
    qn = _norm_name(name)
    if not qn:
        return None
    exact, starts, contains = None, None, None
    for t, title, tl in items:
        n = _norm_name(title)
        if n == qn:
            exact = (t, title)
            break
        if starts is None and n.startswith(qn):
            starts = (t, title)
        if contains is None and qn in n:
            contains = (t, title)
    pick = exact or starts or contains
    return pick  # (ticker, title) or None


# ---- resolver -----------------------------------------------------------
_CRYPTO_PAT = re.compile(r"^([a-z0-9]{2,10})[-/]?(usd|usdt|usdc)$")
_FX_PAT = re.compile(r"^([a-z]{3})([a-z]{3})(=x)?$")
_TICKER_PAT = re.compile(r"^[a-z]{1,5}$")


def resolve_symbol(q: str):
    """q (name or ticker) -> meta {symbol,name,type,source,key}. source in {hl,stooq,cg}."""
    q = (q or "").strip()
    if not q:
        return None
    low = q.lower()
    if low in SYM_CACHE:
        return SYM_CACHE[low]

    meta = None
    # 1) crypto by alias
    if low in CRYPTO_ALIASES:
        coin = CRYPTO_ALIASES[low]
        meta = {"symbol": coin, "name": coin, "type": "CRYPTOCURRENCY", "source": "hl", "key": coin}
    # 2) tradfi by alias (indices, commodities, metals)
    elif low in TRADFI_ALIASES:
        sym, nm, ty = TRADFI_ALIASES[low]
        meta = {"symbol": sym.upper(), "name": nm, "type": ty, "source": "stooq", "key": sym}
    else:
        cm = _CRYPTO_PAT.match(low)
        fm = _FX_PAT.match(low)
        # 3) crypto pair pattern (btc-usd, ethusdt, solusd)
        if cm and cm.group(1).upper() in set(CRYPTO_ALIASES.values()):
            coin = cm.group(1).upper()
            meta = {"symbol": coin, "name": coin, "type": "CRYPTOCURRENCY", "source": "hl", "key": coin}
        elif cm and cm.group(1) in CRYPTO_ALIASES:
            coin = CRYPTO_ALIASES[cm.group(1)]
            meta = {"symbol": coin, "name": coin, "type": "CRYPTOCURRENCY", "source": "hl", "key": coin}
        # 4) FX pair (eurusd, usdjpy, eurusd=x)
        elif fm and fm.group(1) in _CCY and fm.group(2) in _CCY:
            pair = fm.group(1) + fm.group(2)
            meta = {"symbol": pair.upper(), "name": f"{fm.group(1).upper()}/{fm.group(2).upper()}",
                    "type": "CURRENCY", "source": "stooq", "key": pair}
        # 5) explicit index (^spx) or futures (cl.f) typed straight through
        elif q.startswith("^") or low.endswith(".f"):
            meta = {"symbol": q.upper(), "name": q.upper(),
                    "type": "INDEX" if q.startswith("^") else "COMMODITY",
                    "source": "stooq", "key": low}
        # 6) short token: real ticker (known to SEC or typed uppercase) vs a company name
        elif _TICKER_PAT.match(low):
            if low.upper() in _sec_tickers() or q.isupper():
                meta = {"symbol": low.upper(), "name": low.upper(), "type": "EQUITY",
                        "source": "stooq", "key": f"{low}.us"}
            else:
                hit = _sec_ticker(q)
                if hit:
                    meta = {"symbol": hit[0], "name": hit[1], "type": "EQUITY",
                            "source": "stooq", "key": f"{hit[0].lower()}.us"}
                else:
                    cg = _cg_search(q)
                    if cg:
                        on_hl = cg["symbol"] in set(CRYPTO_ALIASES.values())
                        meta = {"symbol": cg["symbol"] or cg["id"].upper(), "name": cg["name"],
                                "type": "CRYPTOCURRENCY", "source": "hl" if on_hl else "cg",
                                "key": cg["symbol"] if on_hl else cg["id"]}
        else:
            # 7) free-text name -> SEC (US equity), else CoinGecko (maybe a coin/project)
            hit = _sec_ticker(q)
            if hit:
                meta = {"symbol": hit[0], "name": hit[1], "type": "EQUITY",
                        "source": "stooq", "key": f"{hit[0].lower()}.us"}
            else:
                cg = _cg_search(q)
                if cg:
                    coin = cg["symbol"]
                    on_hl = coin in set(CRYPTO_ALIASES.values()) or coin in CRYPTO_ALIASES.values()
                    meta = {"symbol": coin or cg["id"].upper(), "name": cg["name"],
                            "type": "CRYPTOCURRENCY", "source": "hl" if on_hl else "cg",
                            "key": coin if on_hl else cg["id"]}
    if meta:
        SYM_CACHE[low] = meta
    return meta


def _candles(meta: dict):
    """Dispatch to the right source. Crypto tries Hyperliquid then CoinGecko OHLC."""
    src = meta.get("source")
    key = meta.get("key")
    if src == "stooq":
        df = _fmp_candles(meta)        # API source (no IP block) when FMP_API_KEY set
        if df is not None:
            meta["source"] = "fmp"
            return df
        return _stooq_candles(key)     # keyless fallback (works on clean IPs / self-host)
    if src == "hl":
        df = _hl_daily(key)
        if df is not None:
            return df
        cg = _cg_search(meta.get("name") or key)  # HL miss -> CoinGecko OHLC fallback
        if cg:
            meta["source"] = "cg"
            return _cg_ohlc(cg["id"])
        return None
    if src == "cg":
        return _cg_ohlc(key)
    return None


def _forecast_symbol(meta: dict):
    """Run Kronos GEN_SAMPLES times over daily candles for one resolved symbol."""
    sym = meta["symbol"]
    df = _candles(meta)
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
        "type": meta.get("type", ""),
        "exchange": meta.get("exchange", ""),
        "source": meta.get("source", ""),
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


@app.get("/src_debug")
def src_debug(q: str = "AAPL"):
    """Diagnostic: resolver decision + per-source reachability from this server's IP."""
    out: dict = {"query": q}
    try:
        meta = resolve_symbol(q)
        out["resolved"] = meta
    except Exception as e:
        out["resolve_err"] = str(e)
        meta = None

    def probe(name, fn):
        try:
            df = fn()
            out[name] = {"ok": df is not None, "rows": (0 if df is None else len(df))}
        except Exception as e:
            out[name] = {"err": str(e)[:200]}

    probe("stooq_aapl_us", lambda: _stooq_candles("aapl.us"))
    probe("hl_BTC", lambda: _hl_daily("BTC"))
    probe("cg_search_bitcoin", lambda: ({"x": _cg_search("bitcoin")}.get("x")
                                        and pd.DataFrame({"a": [1] * 64})))
    try:
        out["sec_items"] = len(_sec_index() or [])
    except Exception as e:
        out["sec_err"] = str(e)[:200]
    # raw stooq body (why is it failing from this IP?)
    try:
        rawb = _http(STOOQ + "?" + urllib.parse.urlencode({"s": "aapl.us", "i": "d"}),
                     headers=BROWSER_UA)
        out["stooq_raw"] = rawb[:160].decode("utf-8", "replace")
    except Exception as e:
        out["stooq_raw_err"] = str(e)[:160]
    out["fmp_key_set"] = bool(os.environ.get("FMP_API_KEY", "").strip())
    if out["fmp_key_set"]:
        probe("fmp_aapl", lambda: _fmp_candles({"type": "EQUITY", "symbol": "AAPL", "key": "aapl.us"}))
        try:
            k = os.environ["FMP_API_KEY"].strip()
            u = FMP + "?" + urllib.parse.urlencode({"symbol": "AAPL", "apikey": k})
            out["fmp_raw"] = _http(u)[:220].decode("utf-8", "replace")
        except Exception as e:
            out["fmp_raw_err"] = str(e)[:160]
    if meta:
        probe("resolved_candles", lambda: _candles(dict(meta)))
    return out


@app.get("/forecast/symbol")
def forecast_symbol(q: str):
    """On-demand forecast for ANY asset (stocks, ETFs, indices, FX, commodities, crypto).
    Resolves q -> symbol -> daily candles (Stooq / Hyperliquid / CoinGecko) -> Kronos."""
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
