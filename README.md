# SuperClaw Kronos Sidecar

Hosts the [Kronos](https://github.com/shiyu-coder/Kronos) foundation model and serves
probabilistic price forecasts to the `superclaw-trader` skill.

For each asset (BTC, ETH, BNB, HYPE, SOL, GOLD) it pulls recent 1h candles from
Hyperliquid (keyless), runs `SAMPLE_COUNT` single-path Kronos forecasts, and reports:

- `prob_pct` — % of paths whose high touches the next round level within the horizon
- `prob_up_pct` — % of paths closing above spot
- `exp_high` / `exp_low` / `exp_close` — averaged forecast range (used by per-asset analytics)

Results are cached and refreshed in the background every `REFRESH_TTL` seconds, so the
skill's `/forecast` call is instant.

## Endpoints
- `GET /health` → `{"ok": true, "status": "...", "model": "..."}`
- `GET /forecast` → cached forecasts for all assets

## Deploy on Render
1. Push this folder to a GitHub repo (e.g. `superpowerdevk/superclaw-kronos-sidecar`).
2. Render → New → Web Service → connect the repo → Runtime: **Docker**.
3. **Instance type: pick one with ≥2 GB RAM** (torch + model weights need headroom; the
   smallest free tier will OOM). Kronos-small on CPU is fine for this workload.
4. Deploy. First boot downloads the model from HuggingFace and warms the cache — `/health`
   returns `"status":"warming"` until the first forecast batch completes, then `"ok"`.
5. Copy the service URL and set it in the skill: `KRONOS_URL=https://<your-service>.onrender.com`.

## Config (env)
| var | default | note |
|---|---|---|
| `KRONOS_MODEL` | `NeoQuasar/Kronos-small` | bump to `Kronos-base` for more capacity |
| `DEVICE` | `cpu` | set `cuda` on a GPU instance for speed |
| `INTERVAL` | `1h` | candle interval |
| `HORIZON_HRS` | `4` | forecast horizon (pred_len = horizon/interval) |
| `LOOKBACK` | `512` | history candles (≤ model max_context) |
| `SAMPLE_COUNT` | `30` | forecast paths → odds resolution (higher = smoother, slower) |
| `REFRESH_TTL` | `300` | seconds between background refreshes |

## Notes
- All data in is keyless (Hyperliquid). No API keys anywhere.
- GOLD uses the `xyz` HIP-3 builder dex; if its candles are unavailable the asset is
  returned with `spot: null` and the skill renders it as price-only (no odds).
- CPU latency: a full refresh is ~`SAMPLE_COUNT` batched forward passes. With caching this
  is hidden from users. Lower `SAMPLE_COUNT` or use a GPU instance if refreshes lag.
