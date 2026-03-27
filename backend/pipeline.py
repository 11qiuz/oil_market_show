import io
import os
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Use non-interactive backend for server rendering
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402


warnings.filterwarnings("ignore")


def _clip_outliers(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lower_bound = series.quantile(lower_q)
    upper_bound = series.quantile(upper_q)
    return series.clip(lower=lower_bound, upper=upper_bound)


def _categorize_risk_dynamic(
    ret: float,
    thresh: float,
    min_thresh: float = 0.015,
) -> int:
    # 0: 下跌风险, 1: 震荡, 2: 上涨风险
    if pd.isna(thresh) or thresh == 0:
        return 1
    actual_thresh = max(thresh, min_thresh)
    if ret > actual_thresh:
        return 2
    if ret < -actual_thresh:
        return 0
    return 1


@dataclass
class RunConfig:
    forward_horizon: int = 5
    plot_days: int = 500
    cost_rate: float = 0.001
    red_percentile: float = 95.0
    yellow_percentile: float = 85.0
    top_factors: int = 12


def run_v3_on_df(
    df_raw: pd.DataFrame,
    config: RunConfig,
    output_dir: str,
) -> Dict[str, Any]:
    """
    将原始 v3.py 逻辑封装成可返回结构化结果的方法。
    输出图片将写到 output_dir，前端通过 URL 读取。
    """
    df = df_raw.copy()
    if "Date" not in df.columns:
        raise ValueError("CSV 必须包含列：Date")
    if "WTI_Crude" not in df.columns:
        raise ValueError("CSV 必须包含列：WTI_Crude")

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    # 🔥🔥🔥 修复：新版 pandas 兼容，删除 method 参数
    df.ffill(inplace=True)

    # ==========================
    # 2. 特征工程
    # ==========================
    features = pd.DataFrame(index=df.index)

    for col in df.columns:
        clean_series = _clip_outliers(df[col])
        features[f"{col}_Return_1d"] = clean_series.pct_change(1)
        features[f"{col}_Return_5d"] = clean_series.pct_change(5)

    for col in ["WTI_Crude", "SP500", "USD_Index"]:
        clean_returns = features[f"{col}_Return_1d"]
        features[f"{col}_Vol_20d"] = clean_returns.rolling(window=20).std()

    # 宏观跨资产因子（与原脚本一致）
    features["WTI_Brent_Spread_Pct"] = (df["WTI_Crude"] - df["Brent_Crude"]) / df["Brent_Crude"]
    safe_wti = df["WTI_Crude"].clip(lower=1.0)
    features["Oil_Gold_Ratio_Log"] = np.log(safe_wti / df["Gold"])
    features["Macro_VIX_OVX_Spread"] = df["VIX"] - df["OVX"]

    # 技术指标
    features["WTI_vs_MA20"] = df["WTI_Crude"] / df["WTI_Crude"].rolling(window=20).mean() - 1
    features["WTI_USD_Corr_30d"] = df["WTI_Crude"].rolling(30).corr(df["USD_Index"])
    ema12 = df["WTI_Crude"].ewm(span=12, adjust=False).mean()
    ema26 = df["WTI_Crude"].ewm(span=26, adjust=False).mean()
    features["WTI_MACD_Pct"] = (ema12 - ema26) / ema26
    features["Month"] = df.index.month

    # 复杂宏观共振因子（V3 原脚本新增）
    features["VIX_USD_Resonance"] = features["VIX_Return_5d"] * features["USD_Index_Return_5d"]
    # 动量衰竭：短期 1d 与 20d 动量对比（原脚本写法保留）
    features["Momentum_Reversal"] = features["WTI_Crude_Return_1d"] - df["WTI_Crude"].pct_change(20)

    # ==========================
    # 3. 构建目标标签（动态阈值）
    # ==========================
    forward_return = df["WTI_Crude"].shift(-config.forward_horizon) / df["WTI_Crude"] - 1
    rolling_5d_vol = df["WTI_Crude"].pct_change(5).rolling(20).std()
    dynamic_threshold = rolling_5d_vol * 1.5

    target_df = pd.DataFrame(
        {"Forward_Return": forward_return, "Dynamic_Threshold": dynamic_threshold},
        index=df.index,
    )
    target = target_df.apply(
        lambda r: _categorize_risk_dynamic(r["Forward_Return"], r["Dynamic_Threshold"]),
        axis=1,
    )

    model_data = features.join(target.rename("Target")).dropna()
    X = model_data.drop("Target", axis=1)
    y = model_data["Target"]

    if len(X) < 50:
        raise ValueError("数据量过少（至少需要较长时间序列以完成训练）。")

    # ==========================
    # 4. 模型训练（带时序交叉验证）
    # ==========================
    tscv = TimeSeriesSplit(n_splits=5)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=42,
    )

    last_classification: Optional[str] = None
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = model.predict(X_test)
        last_classification = classification_report(
            y_test,
            y_pred,
            target_names=["下跌风险 (Class 0)", "震荡 (Class 1)", "上涨风险 (Class 2)"],
        )

    # 最终模型：全量拟合
    sample_weights_full = compute_sample_weight(class_weight="balanced", y=y)
    final_model = xgb.XGBClassifier(**model.get_params())
    final_model.fit(X, y, sample_weight=sample_weights_full)

    # ==========================
    # 5. SHAP 因子解释（影响因素）
    # ==========================
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)
    # 多分类 SHAP 的返回形态不同版本可能略有差异
    if isinstance(shap_values, list):
        shap_values_target = shap_values[2]
    else:
        # shape: (n_samples, n_features, n_classes) 或类似
        shap_values_target = shap_values[:, :, 2]

    feature_names = list(X.columns)
    mean_abs = np.abs(shap_values_target).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][: config.top_factors]

    factors = []
    for i in top_idx:
        factors.append(
            {
                "feature": feature_names[i],
                "importance": float(mean_abs[i]),
                "mean_shap": float(np.mean(shap_values_target[:, i])),
            }
        )

    shap_path = os.path.join(output_dir, "v3_shap_factor_importance.png")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_target, X, plot_type="bar", show=False)
    plt.title(
        "Citi AI: Top Risk Factors Driving Oil Price Upward (V3 Dynamic)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(shap_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ==========================
    # 6. 梯级预警 + 今日信号
    # ==========================
    historical_probs = final_model.predict_proba(X)
    prob_down_history = historical_probs[:, 0]
    prob_up_history = historical_probs[:, 2]

    threshold_down_red = np.percentile(prob_down_history, config.red_percentile)
    threshold_down_yellow = np.percentile(prob_down_history, config.yellow_percentile)
    threshold_up_red = np.percentile(prob_up_history, config.red_percentile)
    threshold_up_yellow = np.percentile(prob_up_history, config.yellow_percentile)

    obs_date = X.index[-1]
    last_day_features = X.iloc[-1:]
    pred_prob = final_model.predict_proba(last_day_features)[0]
    prob_down_today, prob_up_today = float(pred_prob[0]), float(pred_prob[2])

    final_signal = "🟢 正常/低风险"
    action_plan = "无需采取紧急对冲，保持常规敞口。"
    if prob_up_today > threshold_up_red:
        final_signal = "🔴 ⚠️ 上行【极高】风险触发"
        action_plan = "立即锁价：买入看涨期权（Call Options）或等价对冲。"
    elif prob_up_today > threshold_up_yellow:
        final_signal = "🟡 ⚡ 上行【中度】风险警报"
        action_plan = "建议分批建仓远期多头，并准备对冲预案。"
    elif prob_down_today > threshold_down_red:
        final_signal = "🔴 ⚠️ 下行【极高】风险触发"
        action_plan = "提高风控：买入看跌期权（Put Options）或等价对冲。"
    elif prob_down_today > threshold_down_yellow:
        final_signal = "🟡 ⚡ 下行【中度】风险警报"
        action_plan = "建议加速去库存/控成本，确保现金流安全。"

    # ==========================
    # 7. 历史梯级信号复盘图
    # ==========================
    plot_days = min(config.plot_days, len(df))
    df_plot = df.iloc[-plot_days:].copy()
    X_plot = X.iloc[-plot_days:]
    probs_plot = final_model.predict_proba(X_plot)
    pred_down_probs, pred_up_probs = probs_plot[:, 0], probs_plot[:, 2]

    up_red_dates = X_plot[pred_up_probs > threshold_up_red].index
    up_yellow_dates = X_plot[(pred_up_probs > threshold_up_yellow) & (pred_up_probs <= threshold_up_red)].index
    down_red_dates = X_plot[pred_down_probs > threshold_down_red].index
    down_yellow_dates = X_plot[(pred_down_probs > threshold_down_yellow) & (pred_down_probs <= threshold_down_red)].index

    tiered_path = os.path.join(output_dir, "v3_tiered_risk_signals.png")
    plt.figure(figsize=(16, 8))
    plt.plot(
        df_plot.index,
        df_plot["WTI_Crude"],
        label="WTI Crude Price (Actual)",
        color="#2c3e50",
        linewidth=1.5,
        alpha=0.8,
    )

    # 黄灯：早期预警
    plt.scatter(
        up_yellow_dates,
        df_plot.loc[up_yellow_dates, "WTI_Crude"],
        color="orange",
        marker="^",
        s=60,
        label="Yellow Alert: Upward Warning",
        zorder=4,
    )
    plt.scatter(
        down_yellow_dates,
        df_plot.loc[down_yellow_dates, "WTI_Crude"],
        color="#82e0aa",
        marker="v",
        s=60,
        label="Yellow Alert: Downward Warning",
        zorder=4,
    )

    # 红灯：极高风险
    plt.scatter(
        up_red_dates,
        df_plot.loc[up_red_dates, "WTI_Crude"],
        color="#e74c3c",
        marker="^",
        s=180,
        label="Red Alert: Extreme Upward Risk",
        zorder=5,
    )
    plt.scatter(
        down_red_dates,
        df_plot.loc[down_red_dates, "WTI_Crude"],
        color="#27ae60",
        marker="v",
        s=180,
        label="Red Alert: Extreme Downward Risk",
        zorder=5,
    )

    plt.title(
        "Citi Corporate Banking: Tiered AI Risk Alert System (V3 Adaptive)",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("WTI Crude Price (USD)", fontsize=12)
    plt.legend(loc="best", fontsize=10, framealpha=0.9, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(tiered_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ==========================
    # 8. 回测引擎（确定性最高的红灯）
    # ==========================
    bt_df = df_plot[["WTI_Crude"]].copy()
    bt_df["Return_1d"] = bt_df["WTI_Crude"].pct_change()
    bt_df["Signal"] = 0
    bt_df.loc[up_red_dates, "Signal"] = 1
    bt_df.loc[down_red_dates, "Signal"] = -1

    # 持仓周期 = forward_horizon（与原脚本逻辑一致）
    bt_df["Target_Position"] = bt_df["Signal"].replace(0, np.nan).ffill(limit=config.forward_horizon - 1).fillna(0)
    # 避免未来函数：触发日 t 的信号，收益从 t+1 开始计入
    bt_df["Actual_Position"] = bt_df["Target_Position"].shift(1).fillna(0)

    bt_df["Turnover"] = bt_df["Actual_Position"].diff().abs().fillna(0)
    bt_df["Trading_Cost"] = bt_df["Turnover"] * float(config.cost_rate)
    bt_df["Strategy_Return"] = bt_df["Actual_Position"] * bt_df["Return_1d"] - bt_df["Trading_Cost"]

    bt_df["Baseline_Equity"] = (1 + bt_df["Return_1d"].fillna(0)).cumprod()
    bt_df["Strategy_Equity"] = (1 + bt_df["Strategy_Return"].fillna(0)).cumprod()

    total_return = float(bt_df["Strategy_Equity"].iloc[-1] - 1)
    if len(bt_df) > 0:
        annualized_return = float((1 + total_return) ** (252 / len(bt_df)) - 1)
    else:
        annualized_return = 0.0

    rolling_max = bt_df["Strategy_Equity"].cummax()
    drawdown = bt_df["Strategy_Equity"] / rolling_max - 1
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    ret_std = float(bt_df["Strategy_Return"].std())
    sharpe_ratio = float((bt_df["Strategy_Return"].mean() / ret_std) * np.sqrt(252)) if ret_std != 0 else 0.0

    equity_path = os.path.join(output_dir, "trading_backtest_equity.png")
    plt.figure(figsize=(14, 7))
    plt.plot(
        bt_df.index,
        bt_df["Baseline_Equity"],
        label="Buy & Hold (Baseline)",
        color="gray",
        linestyle="--",
        alpha=0.7,
    )
    plt.plot(bt_df.index, bt_df["Strategy_Equity"], label="AI Strategy Equity", color="#d35400", linewidth=2)

    long_zones = bt_df["Actual_Position"] > 0
    short_zones = bt_df["Actual_Position"] < 0
    plt.fill_between(bt_df.index, 0.8, 1.5, where=long_zones, color="red", alpha=0.1, label="Holding Long")
    plt.fill_between(bt_df.index, 0.8, 1.5, where=short_zones, color="green", alpha=0.1, label="Holding Short")

    plt.title("AI Trading Model Backtest: Equity Curve vs Baseline", fontsize=16, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (Base = 1.0)")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.4)
    if len(bt_df):
        ymin = float(bt_df[["Baseline_Equity", "Strategy_Equity"]].min().min()) * 0.95
        ymax = float(bt_df[["Baseline_Equity", "Strategy_Equity"]].max().max()) * 1.05
        plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(equity_path, dpi=300)
    plt.close()

    # 买卖点：用持仓“进入时刻”来表示实际执行（更贴近回测）
    pos = bt_df["Actual_Position"]
    prev_pos = pos.shift(1).fillna(0)
    buy_entry_dates = bt_df.index[(pos == 1) & (prev_pos != 1)]
    sell_entry_dates = bt_df.index[(pos == -1) & (prev_pos != -1)]
    exit_long_dates = bt_df.index[(prev_pos == 1) & (pos != 1)]
    exit_short_dates = bt_df.index[(prev_pos == -1) & (pos != -1)]

    # ==========================
    # 组装结果
    # ==========================
    # 控制 JSON 大小：只返回最近一段日期列表
    def _dates_to_str(idx: pd.DatetimeIndex, n: int = 20) -> List[str]:
        if len(idx) == 0:
            return []
        return [d.strftime("%Y-%m-%d") for d in idx[-n:]]

    result: Dict[str, Any] = {
        "obs_date": obs_date.strftime("%Y-%m-%d"),
        "probabilities": {
            "prob_up": prob_up_today,
            "prob_down": prob_down_today,
        },
        "thresholds": {
            "up_red": float(threshold_up_red),
            "up_yellow": float(threshold_up_yellow),
            "down_red": float(threshold_down_red),
            "down_yellow": float(threshold_down_yellow),
        },
        "final_signal": final_signal,
        "action_plan": action_plan,
        "backtest_metrics": {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "backtest_days": int(len(bt_df)),
        },
        "buy_sell_points": {
            "buy_entry_dates": [d.strftime("%Y-%m-%d") for d in buy_entry_dates[-20:]],
            "sell_entry_dates": [d.strftime("%Y-%m-%d") for d in sell_entry_dates[-20:]],
            "exit_long_dates": [d.strftime("%Y-%m-%d") for d in exit_long_dates[-20:]],
            "exit_short_dates": [d.strftime("%Y-%m-%d") for d in exit_short_dates[-20:]],
        },
        "tiered_alerts": {
            "up_red_count": int(len(up_red_dates)),
            "down_red_count": int(len(down_red_dates)),
            "up_yellow_count": int(len(up_yellow_dates)),
            "down_yellow_count": int(len(down_yellow_dates)),
            "up_red_dates": _dates_to_str(up_red_dates),
            "down_red_dates": _dates_to_str(down_red_dates),
            "up_yellow_dates": _dates_to_str(up_yellow_dates),
            "down_yellow_dates": _dates_to_str(down_yellow_dates),
        },
        "factor_importance": factors,
        "model_eval_report": last_classification,
        "images": {
            "shap_factor_importance": os.path.basename(shap_path),
            "tiered_risk_signals": os.path.basename(tiered_path),
            "equity_curve": os.path.basename(equity_path),
        },
    }

    return result


def run_v3_from_csv_bytes(
    csv_bytes: bytes,
    config: RunConfig,
    output_dir: str,
) -> Dict[str, Any]:
    # 兼容用户可能用逗号/编码等差异：先尝试默认 utf-8，失败再兜底 gbk
    try:
        csv_text = csv_bytes.decode("utf-8")
    except UnicodeDecodeError:
        csv_text = csv_bytes.decode("gbk", errors="ignore")

    df_raw = pd.read_csv(io.StringIO(csv_text))
    return run_v3_on_df(df_raw=df_raw, config=config, output_dir=output_dir)
