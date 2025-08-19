
import threading
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from typing import Dict, Tuple, Optional
import pandas as pd
import ccxt
import requests
import re

try:
    from app.config import Settings
except Exception:
    from ..config import Settings

from ..indicators import ema, obv, atr as atr_ind, rsi, bollinger_bands, macd, adx
from .telegram import send_telegram
from .fng import get_fng
from .movers import compute_mover_score, dollar_volume, compute_early_mover

SYMBOL_TIMEFRAMES = {
    "BTC": "1d",
    "ETH": "4h",
    "XRP": "1d",
    "SOL": "1h",
    "DOGE": "1h",
}

class SignalEngine:
    def _safe_info(self, msg: str):
        try:
            self.log.info(msg)
        except Exception:
            pass
        try:
            self.log_emit(msg)
        except Exception:
            pass



    def __init__(self, log_emit=None, status_emit=None):
        self.s = Settings()
        self.log = logging.getLogger("signals")
        try:
            self.log.propagate = False
        except Exception:
            pass
        self.log_emit = log_emit or (lambda msg: None)
        self.status_emit = status_emit or (lambda asset, status: None)
        self._stop = threading.Event()
        self._running = False
        # Safety: ensure a compat worker exists
        if not hasattr(self, '_worker_compat'):
            self._worker_compat = lambda asset: self.worker(asset)
        self.threads: Dict[str, threading.Thread] = {}
        self.last_state: Dict[str, str] = {}
        self.last_alert_ts: Dict[str, float] = {}
        self._last_fng = None
        self._last_fng_ts = 0.0

        names = [self.s.EXCHANGE] + self.s.EXCHANGE_FALLBACKS
        self.exchange_name, self.exchange = self._create_working_exchange(names)
        self.quote = self.s.QUOTE_ASSET
        self._safe_info('ENGINE BUILD: FIX3/EARLY_MOVERS')

        self._safe_info(f"Using exchange: {self.exchange_name} | preferred quote: {self.quote}")

    # ---------- infra ----------
    def _create_working_exchange(self, names):
        for name in names:
            try:
                ex_ctor = getattr(ccxt, name)
            except AttributeError:
                self._safe_info(f"Exchange {name} not supported by CCXT, skipping.")
                continue
            try:
                ex = ex_ctor({'enableRateLimit': True})
                ex.load_markets()
                return name, ex
            except Exception as e:
                self._safe_info(f"Exchange {name} not usable: {e}. Trying next...")
                continue
        raise RuntimeError("No usable exchange found. Configure EXCHANGE/EXCHANGE_FALLBACKS in .env")

    def _candidate_symbols(self, asset):
        cands = [f"{asset}/{self.quote}", f"{asset}/USD", f"{asset}/USDC", f"{asset}/USDT"]
        out, seen = [], set()
        for s in cands:
            if s not in seen:
                out.append(s); seen.add(s)
        return out

    def _tf_to_seconds(self, tf: str) -> Optional[int]:
        units = {"m":60, "h":3600, "d":86400, "w":604800}
        m = re.match(r"^(\d+)([mhdw])$", tf)
        if not m:
            return None
        return int(m.group(1)) * units[m.group(2)]

    def _normalize_timeframe(self, requested: str) -> str:
        try:
            tfs = getattr(self.exchange, "timeframes", None)
            if not tfs:
                return requested
            keys = list(tfs.keys())
            if requested in keys:
                return requested
            req_s = self._tf_to_seconds(requested) or 0
            best = None; best_diff = None
            for k in keys:
                ks = self._tf_to_seconds(k)
                if ks is None: 
                    continue
                diff = abs(ks - req_s)
                if best is None or diff < best_diff:
                    best, best_diff = k, diff
            return best or requested
        except Exception:
            return requested

    def _use_closed_df(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.s.ALWAYS_USE_CLOSED_CANDLE and len(df) > 2:
                return df.iloc[:-1].copy()
        except Exception:
            pass
        return df

    def _can_alert(self, key: str, cooldown_minutes: int) -> bool:
        now = time.time()
        last = self.last_alert_ts.get(key, 0.0)
        return (now - last) >= cooldown_minutes*60

    def _stamp_alert(self, key: str):
        self.last_alert_ts[key] = time.time()

    def _get_fng(self):
        if not self.s.ENABLE_FNG_FILTER:
            return get_fng(self.s.FNG_CACHE_MINUTES)  # still used for context
        if (time.time() - self._last_fng_ts) > self.s.FNG_CACHE_MINUTES*60:
            self._last_fng = get_fng(self.s.FNG_CACHE_MINUTES)
            self._last_fng_ts = time.time()
        return self._last_fng

    def is_running(self):
        return self._running

    def start(self):
        if self._running: return
        self._stop.clear()
        self._running = True
        enabled = []
        if self.s.ENABLE_BTC: enabled.append("BTC")
        if self.s.ENABLE_ETH: enabled.append("ETH")
        if self.s.ENABLE_XRP: enabled.append("XRP")
        if self.s.ENABLE_SOL: enabled.append("SOL")
        if self.s.ENABLE_DOGE: enabled.append("DOGE")
        for asset in enabled:
            target = self.worker if hasattr(self, 'worker') else (self._worker_compat if hasattr(self, '_worker_compat') else None)
            if target is None:
                raise AttributeError("SignalEngine has neither 'worker' nor '_worker_compat'")
            t = threading.Thread(target=target, args=(asset,), daemon=True)
            self.threads[asset] = t
            t.start()
            self.status_emit(asset, "running")
            self._safe_info(f"{asset} worker started.")
        if getattr(self.s, 'ENABLE_MOVERS', False):
            target = getattr(self, 'worker_movers', None)
            if target:
                tm = threading.Thread(target=target, daemon=True)
                self.threads['MOVERS'] = tm
                tm.start()
                self.status_emit('MOVERS', 'running')
                self._safe_info('MOVERS worker started.')
            else:
                self._safe_info('MOVERS worker not available; skipping.')


    def worker_early_movers(self):
        while not self._stop.is_set():
            try:
                self.scan_early_movers_once()
            except Exception as e:
                self._safe_info(f"[EARLY] error: {e}")
            time.sleep(self.s.POLL_INTERVAL_SECONDS)

    def stop(self):
        if not self._running: return
        self._stop.set()
        for asset, t in self.threads.items():
            try:
                t.join(timeout=3)
            except Exception:
                pass
            self.status_emit(asset, "stopped")
        self.threads.clear()
        self._running = False
        # Safety: ensure a compat worker exists
        if not hasattr(self, '_worker_compat'):
            self._worker_compat = lambda asset: self.worker(asset)
        self._safe_info("All workers stopped.")

    def log_info(self, msg):
        self.log.info(msg)
        self.log_emit(msg)

    # ---------- data ----------
    def fetch_df(self, asset: str, timeframe: str, limit: int=500):
        markets = self.exchange.markets or {}
        normalized_tf = self._normalize_timeframe(timeframe)
        tfs_to_try = [normalized_tf] if normalized_tf == timeframe else [normalized_tf, timeframe]
        for sym in self._candidate_symbols(asset):
            try:
                if sym not in markets:
                    continue
                df = None
                used_tf = None
                last_err = None
                for tf_try in tfs_to_try:
                    try:
                        ohlcv = self.exchange.fetch_ohlcv(sym, timeframe=tf_try, limit=limit)
                        dft = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
                        if not dft.empty:
                            df = dft; used_tf = tf_try; break
                    except Exception as e:
                        last_err = e
                        continue
                if df is None:
                    raise last_err or RuntimeError("No data returned")
                if self.last_state.get(f"{asset}_symbol") != sym:
                    self._safe_info(f"[{asset}] using symbol {sym} on {self.exchange_name}")
                    self.last_state[f"{asset}_symbol"] = sym
                return df, sym, used_tf or timeframe
            except Exception as e:
                self.log.debug(f"[{asset}] {sym} failed: {e}")
                continue
        raise RuntimeError(f"No working market symbol found for {asset} on {self.exchange_name}")

    def solana_ok(self) -> bool:
        try:
            r = requests.get("https://status.solana.com/api/v2/summary.json", timeout=6)
            if r.status_code != 200:
                return True  # don't block if endpoint fails
            data = r.json()
            degraded = any(comp.get("status","").lower() not in ("operational","operating normally","ok") for comp in data.get("components",[]))
            return not degraded
        except Exception:
            return True

    # ---------- strategies ----------
    def strategy_btc(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        df = self._use_closed_df(df)
        df["ema50"] = ema(df["close"], 50)
        df["ema200"] = ema(df["close"], 200)
        df["obv"] = obv(df["close"], df["volume"])
        golden = df["ema50"].iloc[-1] > df["ema200"].iloc[-1]
        obv_up = df["obv"].iloc[-1] > df["obv"].iloc[-5]
        price = float(df["close"].iloc[-1])
        if golden and obv_up:
            return ("BUY", price, "BTC macro: 50/200 EMA Golden + OBV up")
        death = df["ema50"].iloc[-1] < df["ema200"].iloc[-1]
        if death:
            return ("SELL", price, "BTC macro invalidation: Death Cross")
        return (None, None, None)

    def strategy_eth(self, df: pd.DataFrame):
        df = self._use_closed_df(df)
        df["ema50"] = ema(df["close"], 50)
        df["rsi"] = rsi(df["close"], 14)
        price = float(df["close"].iloc[-1])
        uptrend = df["close"].iloc[-1] > df["ema50"].iloc[-1]
        rsi_prev = df["rsi"].iloc[-2]; rsi_now = df["rsi"].iloc[-1]
        if uptrend and (rsi_prev < 30 and rsi_now >= 30):
            return ("BUY", price, "ETH mean reversion: RSI hook-up from <30 in uptrend (>EMA50)")
        if (rsi_now > 70) or (df["close"].iloc[-1] < df["ema50"].iloc[-1] and df["close"].iloc[-2] >= df["ema50"].iloc[-2]):
            return ("SELL", price, "ETH exit: EMA50 break or RSI>70")
        return (None, None, None)

    def strategy_xrp(self, df: pd.DataFrame):
        df = self._use_closed_df(df)
        upper, mid, lower = bollinger_bands(df["close"], 20, 2.0)
        width = upper - lower
        width_ma = width.rolling(20).mean()
        vol = df["volume"]; vol_ma = vol.rolling(20).mean()
        price = float(df["close"].iloc[-1])
        squeeze = width.iloc[-10:-1].mean() < width_ma.iloc[-10:-1].mean()*0.8  # recent compression
        breakout = df["close"].iloc[-1] > upper.iloc[-2] and vol.iloc[-1] >= 1.5*(vol_ma.iloc[-1] + 1e-9)
        breakdown = df["close"].iloc[-1] < lower.iloc[-2] and vol.iloc[-1] >= 1.5*(vol_ma.iloc[-1] + 1e-9)
        if squeeze and breakout:
            return ("BUY", price, "XRP breakout: BB squeeze + vol ≥1.5×20")
        if squeeze and breakdown:
            return ("SELL", price, "XRP breakdown: BB squeeze + vol ≥1.5×20")
        return (None, None, None)

    def strategy_sol(self, df: pd.DataFrame):
        df = self._use_closed_df(df)
        macd_line, signal, hist = macd(df["close"])
        adxv = adx(df["high"], df["low"], df["close"], 14)
        price = float(df["close"].iloc[-1])
        if not self.solana_ok():
            return (None, None, None)
        bull = macd_line.iloc[-2] <= signal.iloc[-2] and macd_line.iloc[-1] > signal.iloc[-1]
        bear = macd_line.iloc[-2] >= signal.iloc[-2] and macd_line.iloc[-1] < signal.iloc[-1]
        strong = adxv.iloc[-1] >= 25
        if bull and strong:
            return ("BUY", price, "SOL momentum: MACD bull + ADX≥25")
        if bear and strong:
            return ("SELL", price, "SOL exit: MACD bear + ADX≥25")
        return (None, None, None)

    def strategy_doge(self, df: pd.DataFrame):
        df = self._use_closed_df(df)
        ema200v = ema(df["close"], 200)
        ema50v = ema(df["close"], 50)
        macd_line, signal, hist = macd(df["close"])
        adxv = adx(df["high"], df["low"], df["close"], 14)
        upper, mid, lower = bollinger_bands(df["close"], 20, 2.0)
        width = upper - lower; width_ma = width.rolling(20).mean()
        vol = df["volume"]; vol_ma = vol.rolling(20).mean()
        rsiv = rsi(df["close"], 14)
        price = float(df["close"].iloc[-1])

        uptrend = df["close"].iloc[-1] > ema200v.iloc[-1]
        macd_bull = macd_line.iloc[-2] <= signal.iloc[-2] and macd_line.iloc[-1] > signal.iloc[-1]
        macd_bear = macd_line.iloc[-2] >= signal.iloc[-2] and macd_line.iloc[-1] < signal.iloc[-1]
        adx_ok = adxv.iloc[-1] >= 22
        bb_expand = width.iloc[-1] > 1.2*(width_ma.iloc[-1] + 1e-9)
        vol_spike = vol.iloc[-1] >= 1.8*(vol_ma.iloc[-1] + 1e-9)
        rsi_gate = rsiv.iloc[-1] > 50

        if uptrend and macd_bull and adx_ok and bb_expand and vol_spike and rsi_gate:
            return ("BUY", price, "DOGE confluence: MACD bull + ADX≥22 + BB expand + Vol≥1.8×20 + RSI>50 & >EMA200")
        if (macd_bear and adx_ok) or (df['close'].iloc[-1] < ema50v.iloc[-1]) or (rsiv.iloc[-1] > 70):
            return ("SELL", price, "DOGE exit: MACD bear+ADX≥22 or EMA50 break/RSI>70")
        return (None, None, None)

    def _evaluate(self, asset: str, df: pd.DataFrame):
        if asset == "BTC":
            return self.strategy_btc(df)
        if asset == "ETH":
            return self.strategy_eth(df)
        if asset == "XRP":
            return self.strategy_xrp(df)
        if asset == "SOL":
            return self.strategy_sol(df)
        if asset == "DOGE":
            return self.strategy_doge(df)
        return (None, None, None)

    # ---------- workers ----------
    def send_signal(self, asset: str, side: str, price: float, reason: str):
        fng = self._get_fng()
        ctx = f" | F&G: {fng['value']} ({fng['classification']})" if fng else ""
        msg = f"⚡ <b>{asset} {side}</b> @ <b>{price:.4f}</b>\\n{reason}{ctx}"
        self._safe_info(f"{asset} {side} @ {price:.4f} | {reason}{ctx}")
        res = send_telegram(msg)
        self._safe_info(f"telegram: {res}")

    def worker(self, asset: str):
        tf = SYMBOL_TIMEFRAMES.get(asset, "1h")
        self.last_state.setdefault(asset, "NEUTRAL")
        while not self._stop.is_set():
            try:
                df, sym, used_tf = self.fetch_df(asset, tf, limit=500)
                df = self._use_closed_df(df)
                side, price, reason = self._evaluate(asset, df)
                if side:
                    # FNG gating
                    fng = self._get_fng()
                    if self.s.ENABLE_FNG_FILTER and fng:
                        if side == 'BUY' and fng['value'] > self.s.FNG_GREED_THRESHOLD:
                            self._safe_info(f"[{asset}] BUY suppressed by FNG (>{self.s.FNG_GREED_THRESHOLD})")
                            time.sleep(self.s.POLL_INTERVAL_SECONDS); 
                            continue
                        if side == 'SELL' and fng['value'] < self.s.FNG_FEAR_THRESHOLD and "Death Cross" not in (reason or ""):
                            self._safe_info(f"[{asset}] SELL suppressed by FNG (<{self.s.FNG_FEAR_THRESHOLD})")
                            time.sleep(self.s.POLL_INTERVAL_SECONDS); 
                            continue
                    # cooldown
                    key = f"{asset}"
                    if not self._can_alert(key, self.s.MIN_ALERT_GAP_MINUTES):
                        time.sleep(self.s.POLL_INTERVAL_SECONDS); 
                        continue
                    current_state = f"{side}"
                    if current_state != self.last_state[asset]:
                        self.send_signal(asset, side, float(price), reason + f" | symbol={sym} tf={used_tf} ex={self.exchange_name}")
                        self.last_state[asset] = current_state
                        self._stamp_alert(key)
                time.sleep(self.s.POLL_INTERVAL_SECONDS)
            except Exception as e:
                self._safe_info(f"[{asset}] error: {e}")
                time.sleep(self.s.POLL_INTERVAL_SECONDS)


def _worker_compat(self, asset: str):
    # Fallback worker identical to 'worker' to avoid AttributeError in older builds
    tf = SYMBOL_TIMEFRAMES.get(asset, "1h")
    self.last_state.setdefault(asset, "NEUTRAL")
    while not self._stop.is_set():
        try:
            df, sym, used_tf = self.fetch_df(asset, tf, limit=500)
            df = self._use_closed_df(df)
            side, price, reason = self._evaluate(asset, df)
            if side:
                fng = self._get_fng()
                if self.s.ENABLE_FNG_FILTER and fng:
                    if side == 'BUY' and fng['value'] > self.s.FNG_GREED_THRESHOLD:
                        time.sleep(self.s.POLL_INTERVAL_SECONDS); 
                        continue
                    if side == 'SELL' and fng['value'] < self.s.FNG_FEAR_THRESHOLD and "Death Cross" not in (reason or ""):
                        time.sleep(self.s.POLL_INTERVAL_SECONDS); 
                        continue
                key = f"{asset}_SIDE"
                last_ts_ok = self._can_alert(key, self.s.MIN_ALERT_GAP_MINUTES)
                current_state = f"{asset}_{side}"
                if last_ts_ok and current_state != self.last_state.get(asset):
                    self.send_signal(asset, side, float(price), reason + f" | symbol={sym} tf={used_tf} ex={self.exchange_name}")
                    self.last_state[asset] = current_state
                    self._stamp_alert(key)
            time.sleep(self.s.POLL_INTERVAL_SECONDS)
        except Exception as e:
            self._safe_info(f"[{asset}] error: {e}")
            time.sleep(self.s.POLL_INTERVAL_SECONDS)

    def _movers_universe(self, limit: int):
        markets = self.exchange.load_markets()
        quotes = [self.quote, "USD", "USDC", "USDT"]
        cands = []
        for sym, m in markets.items():
            try:
                if m.get("spot") is False:
                    continue
            except Exception:
                pass
            if "/" not in sym:
                continue
            base, quote = sym.split("/")
            if quote not in quotes:
                continue
            cands.append(sym)
        seen = set(); out = []
        for sym in cands:
            base = sym.split("/")[0]
            if base in seen: continue
            seen.add(base); out.append(sym)
            if len(out) >= limit: break
        return out

    def _fetch_two_tfs(self, symbol: str, tf_fast: str, tf_slow: str):
        tff = self._normalize_timeframe(tf_fast)
        tfs = self._normalize_timeframe(tf_slow)
        dff = self.exchange.fetch_ohlcv(symbol, timeframe=tff, limit=400)
        dfs = self.exchange.fetch_ohlcv(symbol, timeframe=tfs, limit=400)
        df1 = pd.DataFrame(dff, columns=["ts","open","high","low","close","volume"])
        df2 = pd.DataFrame(dfs, columns=["ts","open","high","low","close","volume"])
        return df1, df2, tff, tfs

    def scan_movers_once(self):
        if not self.s.ENABLE_MOVERS:
            return []
        symbols = self._movers_universe(self.s.MOVERS_UNIVERSE_LIMIT)
        results = []
        fng = self._get_fng()
        for sym in symbols:
            try:
                df1h, df4h, tfa, tfb = self._fetch_two_tfs(sym, self.s.MOVERS_TF_FAST, self.s.MOVERS_TF_SLOW)
                if df1h.empty or df4h.empty:
                    continue
                # closed-bar
                if self.s.ALWAYS_USE_CLOSED_CANDLE and len(df1h) > 2:
                    df1h = df1h.iloc[:-1]
                if self.s.ALWAYS_USE_CLOSED_CANDLE and len(df4h) > 2:
                    df4h = df4h.iloc[:-1]
                # Liquidity
                usd_turnover = dollar_volume(df1h, bars=24)
                if usd_turnover < self.s.MOVERS_MIN_DOLLAR_VOL:
                    continue
                score, components, reason = compute_mover_score(df1h, df4h)
                # Optional FNG gating for movers BUY-like signals
                if self.s.ENABLE_FNG_FILTER and fng and fng['value'] > self.s.FNG_GREED_THRESHOLD:
                    continue
                if score >= self.s.MOVERS_SCORE_THRESHOLD:
                    asset = sym.split("/")[0]
                    # cooldown
                    key = f"MOVERS_{asset}"
                    if not self._can_alert(key, self.s.MOVERS_COOLDOWN_MINUTES):
                        continue
                    price = float(df1h["close"].iloc[-1])
                    text = f"🚀 Top Mover: <b>{asset}</b> @ <b>{price:.4f}</b>\\nScore={score} | {reason} | symbol={sym} tf={tfa}/{tfb} ex={self.exchange_name}"
                    self._safe_info(text.replace("<b>","").replace("</b>",""))
                    # Append FNG context via send_telegram in send_signal? Movers uses direct telegram
                    fng_ctx = f" | F&G: {fng['value']} ({fng['classification']})" if fng else ""
                    send_telegram(text + fng_ctx)
                    self.last_state[f"MOVERS_{asset}"] = f"{score}"
                    self._stamp_alert(key)
                    results.append({"asset": asset, "score": score, "symbol": sym, "tf": f"{tfa}/{tfb}"})
            except Exception as e:
                self.log.debug(f"[MOVERS] {sym} failed: {e}")
                continue
        return results


def scan_early_movers_once(self):
    if not self.s.ENABLE_EARLY_MOVERS:
        return []
    try:
        markets = self.exchange.load_markets()
    except Exception as e:
        self._safe_info(f"[EARLY] load_markets failed: {e}")
        return []

    # Build universe (same as movers, quote filter)
    universe = [s for s, m in markets.items() if s.endswith(f"/{self.quote}") and m.get("active", True)]
    universe = sorted(universe)[: max(5, self.s.MOVERS_UNIVERSE_LIMIT)]
    results = []
    # Get FNG
    fng = None
    if self.s.ENABLE_FNG_FILTER:
        try:
            fng = self._get_fng_cached()
        except Exception:
            fng = None

    for sym in universe:
        try:
            tfa, tfb = self.s.EARLY_MOVERS_TF_FAST, self.s.EARLY_MOVERS_TF_SLOW
            df_fast = self._fetch_ohlcv_as_df(sym, tfa, 240)  # ~20h for 5m
            df_slow = self._fetch_ohlcv_as_df(sym, tfb, 240)  # ~60h for 15m
            # Liquidity
            usd_turnover = dollar_volume(df_fast, bars=48)
            if usd_turnover < self.s.EARLY_MOVERS_MIN_DOLLAR_VOL:
                continue
            # Optional: closed candle
            if self.s.ALWAYS_USE_CLOSED_CANDLE and len(df_fast) > 2:
                df_fast = df_fast.iloc[:-1]
            if self.s.ALWAYS_USE_CLOSED_CANDLE and len(df_slow) > 2:
                df_slow = df_slow.iloc[:-1]

            ok, reason = compute_early_mover(df_fast, df_slow,
                                             min_ret_fast=self.s.EARLY_MOVERS_MIN_RET_5M,
                                             vol_spike_x=self.s.EARLY_MOVERS_VOL_SPIKE_X)
            if not ok:
                continue
            # Sentiment gate (skip in extreme greed to avoid buying tops)
            if self.s.ENABLE_FNG_FILTER and fng and fng['value'] > self.s.FNG_GREED_THRESHOLD:
                continue

            asset = sym.split("/")[0]
            key = f"EMOVER_{asset}"
            if not self._can_alert(key, self.s.EARLY_MOVERS_COOLDOWN_MINUTES):
                continue

            price = float(df_fast["close"].iloc[-1])
            text = f"⏰ Early Mover: <b>{asset}</b> @ <b>{price:.4f}</b> | {reason} | symbol={sym} tf={tfa}/{tfb} ex={self.exchange_name}"
            self._safe_info(text.replace("<b>","").replace("</b>",""))
            fng_ctx = f" | F&G: {fng['value']} ({fng['classification']})" if fng else ""
            send_telegram(text + fng_ctx)
            self.last_state[key] = "ALERTED"
            self._stamp_alert(key)
            results.append({"asset": asset, "symbol": sym, "tf": f"{tfa}/{tfb}", "price": price, "reason": reason})
        except Exception as e:
            self.log.debug(f"[EARLY] {sym} failed: {e}")
            continue
    return results

    def worker_movers(self):
        while not self._stop.is_set():
            try:
                self.scan_movers_once()
            except Exception as e:
                self._safe_info(f"[MOVERS] error: {e}")
            time.sleep(self.s.POLL_INTERVAL_SECONDS)

    # ---------- control helpers ----------
    def force_send_all(self):
        sent = []
        for asset, tf in SYMBOL_TIMEFRAMES.items():
            try:
                df, sym, used_tf = self.fetch_df(asset, tf, limit=500)
                df = self._use_closed_df(df)
                side, price, reason = self._evaluate(asset, df)
                if side:
                    self.send_signal(asset, side, float(price), (reason or "") + f" | symbol={sym} tf={used_tf} ex={self.exchange_name}")
                    sent.append(asset)
            except Exception as e:
                self._safe_info(f"[{asset}] force_send error: {e}")
        return sent

    def reset_states(self):
        self.last_state.clear()
        self.last_alert_ts.clear()
        self._safe_info("Signal states reset.")


# --- SAFETY PATCH: ensure MOVERS worker is available even if a nested def broke binding ---
def _ensure_movers_worker():
    try:
        _ = getattr(SignalEngine, "worker_movers")
        if callable(_):
            return
    except Exception:
        pass

    def worker_movers(self):
        # Loop calling scan_movers_once
        while not getattr(self, "_stop", None) or not self._stop.is_set():
            try:
                if hasattr(self, "scan_movers_once"):
                    self.scan_movers_once()
                elif hasattr(self, "scan_early_movers_once"):
                    # Fallback if movers scanner is named early movers
                    self.scan_early_movers_once()
                else:
                    # If no scanner exists, sleep and try later
                    time.sleep(getattr(self.s, "POLL_INTERVAL_SECONDS", 5))
                    continue
            except Exception as e:
                try:
                    self._safe_info(f"[MOVERS] error: {e}")
                except Exception:
                    pass
            time.sleep(getattr(self.s, "POLL_INTERVAL_SECONDS", 5))

    try:
        SignalEngine.worker_movers = worker_movers
    except Exception:
        pass

_ensure_movers_worker()
# --- END SAFETY PATCH ---
