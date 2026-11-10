"""Phase 3: long/short volatility signals with confidence, and a 5-year backtest.

The VIX index itself cannot be bought. The tradable stand-ins used here are:
    long volatility  -> VXX  (iPath Series B ETN, 1x short-term VIX futures)
    short volatility -> SVXY (ProShares, -0.5x short-term VIX futures since Feb 2018)
Both lose or gain through futures roll as well as VIX moves, so results differ from the VIX.

Signal pipeline (everything uses only data known at the close of day t):
    1. Phase 2 walk-forward HAR+VIX+FOMC (log) forecast of tomorrow's S&P variance.
    2. Features: VIX vs that forecast (variance risk premium), VIX vs its 1-year mean,
       VIX momentum, VIX / VIX3M term structure, FOMC-tomorrow flag.
    3. Walk-forward logistic regression -> P(volatility product rises over the day held).
       That probability IS the confidence: P(up) for a long, P(down) = 1 - P(up) for a short.
    4. Rules (fixed in advance, not tuned on the test period):
         confidence >= 65%  -> full size,  55-65% -> half size,  below 55% -> flat.

Run:  python -m src.backtest [--refresh] [--lag 1]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data import ROOT, load_data, load_vix
from src.evaluate import walk_forward
from src.features import build_features
from src.models import FULL_FEATURES, HARLogModel

OUT_DIR = ROOT / "outputs"
RAW_DIR = ROOT / "data" / "raw"
FOMC_PATH = ROOT / "data" / "events" / "fomc_decision_dates.csv"

INITIAL_CAPITAL = 10_000.0
HALF_CONF, FULL_CONF = 0.55, 0.65      # fixed a priori; see module docstring
COST_BPS = 5.0                          # per unit of turnover (spread + slippage), each way
FEATURE_COLS = ["vrp", "vix_z", "vix_chg1", "vix_chg5", "term", "fomc_next"]
LABEL_ETF = "UVXY"                      # label source: reaches back to 2011, same sign as a long-vol ETP
TRADE_LONG, TRADE_SHORT = "VXX", "SVXY"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_yahoo_close(ticker: str, refresh: bool = False) -> pd.Series:
    """Dividend/split-adjusted daily close for an ETF or index, cached in data/raw/."""
    import yfinance as yf

    path = RAW_DIR / f"close_{ticker.strip('^').lower()}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)["Close"].rename(ticker)
    raw = yf.download(ticker, start="2000-01-01", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {ticker}; check your connection and retry.")
    raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].dropna()
    s.index = pd.DatetimeIndex(s.index, name="Date")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    s.to_frame("Close").to_csv(path)
    return s.rename(ticker)


# ----------------------------------------------------------------------------
# Signal features and walk-forward classifier
# ----------------------------------------------------------------------------
def har_forecast_vol(df: pd.DataFrame, vix: pd.Series, fomc_dates) -> pd.Series:
    """Walk-forward HAR+VIX+FOMC (log) forecast of next-day vol, annualized %, indexed by origin day."""
    feats = build_features(df, vix, fomc_dates)
    fc = walk_forward(feats, [HARLogModel(FULL_FEATURES, "har")])
    return np.sqrt(252 * fc["har"]) * 100


def build_signal_features(vix: pd.Series, vix3m: pd.Series, har_vol: pd.Series,
                          fomc_next: pd.Series) -> pd.DataFrame:
    """Features at close of day t. Rolling windows look backward only."""
    lv = np.log(vix)
    f = pd.DataFrame(index=vix.index)
    f["vrp"] = lv - np.log(har_vol.reindex(vix.index))        # >0: VIX rich vs forecast realized vol
    f["vix_z"] = (lv - lv.rolling(252).mean()) / lv.rolling(252).std()
    f["vix_chg1"] = lv.diff(1)
    f["vix_chg5"] = lv.diff(5)
    f["term"] = lv - np.log(vix3m.reindex(vix.index))         # >0: backwardation (stress)
    f["fomc_next"] = fomc_next.reindex(vix.index)
    return f


def forward_label(etf_close: pd.Series, index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """1 if the ETF rose on the day that will be held (t + horizon), else 0. NaN if unknown.

    horizon = 1 + lag. This is a LABEL only: models never see it for days not yet complete.
    """
    r = etf_close.reindex(index).pct_change(fill_method=None)
    y = (r.shift(-horizon) > 0).astype(float)
    y[r.shift(-horizon).isna()] = np.nan
    return y


def walk_forward_probability(X: pd.DataFrame, y: pd.Series, horizon: int, start: pd.Timestamp,
                             refit_every: int = 21, min_train: int = 500) -> pd.DataFrame:
    """Expanding-window logistic regression, refit every `refit_every` origins.

    At origin i the training rows are those j <= i - horizon, i.e. rows whose label day is
    already complete. Features are standardized with training-sample statistics only.
    Returns p_up (P(volatility product rises)) and base_rate (training share of up days).
    """
    valid = X.notna().all(axis=1) & y.notna()
    idx = X.index
    origins = [i for i in range(len(idx)) if idx[i] >= start and X.iloc[i].notna().all()]
    p_up, base, params, mu, sd, last_fit = {}, {}, None, None, None, None
    for n, i in enumerate(origins):
        if params is None or n % refit_every == 0:
            tr = valid.iloc[: i - horizon + 1]
            tr = tr[tr]
            if len(tr) >= min_train:
                Xt = X.loc[tr.index]
                mu, sd = Xt.mean(), Xt.std().replace(0, 1)
                try:
                    res = sm.Logit(y.loc[tr.index], sm.add_constant((Xt - mu) / sd)).fit(disp=0, maxiter=200)
                    params, last_fit = res.params, float(y.loc[tr.index].mean())
                except Exception:      # keep the previous fit if this one fails to converge
                    pass
        if params is None:
            continue
        z = ((X.iloc[i] - mu) / sd).values
        eta = params.iloc[0] + float(np.dot(params.iloc[1:].values, z))
        p_up[idx[i]] = 1 / (1 + np.exp(-eta))
        base[idx[i]] = last_fit
    return pd.DataFrame({"p_up": pd.Series(p_up), "base_rate": pd.Series(base)})


# ----------------------------------------------------------------------------
# Trading rules and backtest engine
# ----------------------------------------------------------------------------
def make_signals(p_up: pd.Series, half: float = HALF_CONF, full: float = FULL_CONF,
                 long_half: float | None = None, long_full: float | None = None) -> pd.DataFrame:
    """Turn P(up) into direction, confidence and target size (fraction of equity).

    Shorts need P(down) >= half (full size at >= full). Longs need P(up) >= long_half
    (full size at >= long_full); these default to the same bars as shorts.
    """
    long_half = half if long_half is None else long_half
    long_full = full if long_full is None else long_full
    p_down = 1 - p_up
    direction = np.where(p_up >= long_half, 1, np.where(p_down >= half, -1, 0))
    conf = np.where(direction == 1, p_up, np.where(direction == -1, p_down, np.maximum(p_up, p_down)))
    size = np.where(direction == 0, 0.0,
                    np.where(direction == 1, np.where(p_up >= long_full, 1.0, 0.5),
                             np.where(conf >= full, 1.0, 0.5)))
    label = np.where(direction == 1, "LONG", np.where(direction == -1, "SHORT", "FLAT"))
    return pd.DataFrame({"signal": label, "direction": direction, "confidence": conf,
                         "size": size, "p_up": p_up}, index=p_up.index)


def run_backtest(signals: pd.DataFrame, r_long: pd.Series, r_short: pd.Series, lag: int,
                 initial: float = INITIAL_CAPITAL, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Daily close-to-close simulation.

    Signal at close t is held over day t+1+lag (lag=0: order at the same close; lag=1: one extra
    day to execute). Weights = size * direction in the matching ETF; remainder in cash at 0%.
    Turnover cost is charged on the change in each ETF's weight.
    """
    days = signals.index
    w_long = (signals["direction"].eq(1) * signals["size"]).shift(1 + lag)
    w_short = (signals["direction"].eq(-1) * signals["size"]).shift(1 + lag)
    out = pd.DataFrame({"w_long": w_long, "w_short": w_short}).dropna()
    out["r_long"] = r_long.reindex(out.index)
    out["r_short"] = r_short.reindex(out.index)
    turnover = out["w_long"].diff().abs().fillna(out["w_long"].abs()) + \
        out["w_short"].diff().abs().fillna(out["w_short"].abs())
    out["gross"] = out["w_long"] * out["r_long"] + out["w_short"] * out["r_short"]
    out["net"] = out["gross"] - turnover * cost_bps / 1e4
    out["equity"] = initial * (1 + out["net"]).cumprod()
    out["turnover"] = turnover
    return out


def perf_stats(daily_ret: pd.Series, initial: float = INITIAL_CAPITAL) -> dict:
    eq = initial * (1 + daily_ret).cumprod()
    years = len(daily_ret) / 252
    dd = eq / eq.cummax() - 1
    vol = daily_ret.std() * np.sqrt(252)
    return {"final_value": eq.iloc[-1], "total_return": eq.iloc[-1] / initial - 1,
            "cagr": (eq.iloc[-1] / initial) ** (1 / years) - 1, "ann_vol": vol,
            "sharpe": daily_ret.mean() * 252 / vol if vol > 0 else np.nan,
            "max_drawdown": dd.min(), "worst_day": daily_ret.min()}


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def build_everything(lag: int, refresh: bool = False, as_of: pd.Timestamp | None = None,
                     long_half: float | None = None, long_full: float | None = None):
    df = load_data(refresh=refresh, verbose=False)
    vix = load_vix(refresh=refresh, verbose=False)
    fomc_dates = pd.read_csv(FOMC_PATH, parse_dates=["date"])["date"]
    vix3m = load_yahoo_close("^VIX3M", refresh)
    etf = {t: load_yahoo_close(t, refresh) for t in [TRADE_LONG, TRADE_SHORT, LABEL_ETF, "SPY"]}
    as_of = as_of or min(vix.index.max(), df.index.max())
    start = (as_of - pd.DateOffset(years=5))
    start = vix.index[vix.index.searchsorted(start)]          # first trading day on/after 5y ago

    har_vol = har_forecast_vol(df, vix, fomc_dates)
    fomc_next = build_features(df, vix, fomc_dates)["fomc_next"]
    X = build_signal_features(vix, vix3m, har_vol, fomc_next)
    X = X.loc[:as_of]
    y = forward_label(etf[LABEL_ETF], X.index, horizon=1 + lag)
    probs = walk_forward_probability(X, y, horizon=1 + lag, start=start)
    sig = make_signals(probs["p_up"], long_half=long_half, long_full=long_full)
    rets = {t: s.reindex(X.index).pct_change(fill_method=None) for t, s in etf.items()}
    bt = run_backtest(sig.loc[start:], rets[TRADE_LONG], rets[TRADE_SHORT], lag)
    # Realized outcome of each signal, for calibration: did the held ETF move the predicted way?
    held = (bt["w_long"] > 0) | (bt["w_short"] > 0)
    return dict(bt=bt, sig=sig, rets=rets, start=start, as_of=as_of, probs=probs, held=held,
                lag=lag, y=y)


def summarize(res: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bt, sig, rets, lag = res["bt"], res["sig"], res["rets"], res["lag"]
    idx = bt.index
    rows = {"Strategy (signals)": bt["net"]}
    for name, r in [("Buy & hold SVXY", rets[TRADE_SHORT]), ("Buy & hold VXX", rets[TRADE_LONG]),
                    ("Buy & hold SPY", rets["SPY"])]:
        rows[name] = r.reindex(idx).fillna(0)
    stats = pd.DataFrame({k: perf_stats(v) for k, v in rows.items()}).T
    yearly = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in rows.items()})

    # Calibration: for each confidence bucket, how often was the call right?
    s = sig.shift(1 + lag).reindex(idx)
    outcome = np.where(s["direction"] == 1, res["rets"][TRADE_LONG].reindex(idx) > 0,
                       res["rets"][TRADE_SHORT].reindex(idx) > 0)
    cal = pd.DataFrame({"signal": s["signal"], "conf": s["confidence"], "right": outcome})
    cal = cal[cal["signal"] != "FLAT"]
    cal["bucket"] = pd.cut(cal["conf"], [0.55, 0.60, 0.65, 0.70, 0.80, 1.0], right=False)
    calib = cal.groupby(["signal", "bucket"], observed=True).agg(
        days=("right", "size"), stated_confidence=("conf", "mean"), actual_hit_rate=("right", "mean"))
    return stats, yearly, calib


def plot_results(res: dict, path) -> None:
    bt, sig, rets = res["bt"], res["sig"], res["rets"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(bt.index, bt["equity"], label="Strategy", lw=2, color="#1f4e79")
    for name, key, c in [("Buy & hold SVXY", TRADE_SHORT, "#c0504d"), ("Buy & hold VXX", TRADE_LONG, "#888888"),
                         ("Buy & hold SPY", "SPY", "#4f9d5d")]:
        ax[0].plot(bt.index, INITIAL_CAPITAL * (1 + rets[key].reindex(bt.index).fillna(0)).cumprod(),
                   label=name, lw=1.2, color=c)
    ax[0].set_yscale("log"); ax[0].set_ylabel("Portfolio value ($, log scale)")
    ax[0].set_title(f"$10,000 start, {bt.index[0].date()} to {bt.index[-1].date()} (execution lag {res['lag']} day)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    pos = (bt["w_long"] - bt["w_short"])
    ax[1].fill_between(bt.index, pos, step="mid", color="#1f4e79", alpha=0.7)
    ax[1].set_ylabel("Position\n(+ VXX / - SVXY)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(refresh: bool = False, lag: int = 1) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 170)
    variants = [(lag, None, None, ""), (0, None, None, ""),
                (lag, 0.45, 0.50, "_longvariant")]   # variant chosen AFTER seeing run 1: treat with suspicion
    for L, lh, lf, suffix in variants:
        res = build_everything(L, refresh, long_half=lh, long_full=lf)
        stats, yearly, calib = summarize(res)
        tag = f"lag{L}{suffix}"
        stats.to_csv(OUT_DIR / f"backtest_stats_{tag}.csv")
        yearly.to_csv(OUT_DIR / f"backtest_yearly_{tag}.csv")
        calib.to_csv(OUT_DIR / f"backtest_calibration_{tag}.csv")
        res["bt"].to_csv(OUT_DIR / f"backtest_daily_{tag}.csv")
        res["sig"].loc[res["start"]:].to_csv(OUT_DIR / f"signals_{tag}.csv")
        plot_results(res, OUT_DIR / f"backtest_equity_{tag}.png")
        bt = res["bt"]
        print(f"\n===== {tag}: execution lag {L} day(s): {bt.index[0].date()} -> {bt.index[-1].date()} =====")
        print(stats.assign(final_value=stats["final_value"].round(0)).to_string(
            float_format=lambda v: f"{v:,.3f}"))
        print("\nCalendar-year returns:"); print(yearly.round(3).to_string())
        print("\nConfidence calibration (stated vs realized hit rate):"); print(calib.round(3).to_string())
        sg = res["sig"].loc[res["start"]:]
        print("\nSignal mix:", sg["signal"].value_counts(normalize=True).round(3).to_dict(),
              "| avg daily turnover", round(bt["turnover"].mean(), 3))
        print("Latest signal:", sg.iloc[-1][["signal", "confidence", "size"]].to_dict(), "as of", sg.index[-1].date())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--lag", type=int, default=1, help="extra execution delay in trading days")
    a = ap.parse_args()
    main(a.refresh, a.lag)
