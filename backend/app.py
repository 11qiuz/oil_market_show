import os
import json
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .pipeline import RunConfig, run_v3_from_csv_bytes


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
RESULTS_DIR = os.path.join(STATIC_DIR, "results")
#
# 将生成文件同步到你本地 v3.py/数据集所在目录
# 位置：D:\桌面\新建文件夹 (2)\
# 生成到：D:\桌面\新建文件夹 (2)\web_outputs_v3\
#
TARGET_BASE_DIR = r"D:\桌面\新建文件夹 (2)"
TARGET_OUTPUT_DIR = os.path.join(TARGET_BASE_DIR, "web_outputs_v3")

os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(title="Oil Quant AI Dashboard (V3)")

# 简单允许跨域（如果你把前端单独部署到别的域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态资源（返回图片 URL 使用）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/run")
async def run(
    file: UploadFile = File(...),
    forward_horizon: int = Form(5),
    plot_days: int = Form(500),
    cost_rate: float = Form(0.001),
    red_percentile: float = Form(95.0),
    yellow_percentile: float = Form(85.0),
    top_factors: int = Form(12),
):
    """
    输入：上传一份油价/宏观数据 CSV（必须包含 v3.py 用到的列名，如 Date/WIti_Crude/Brent_Crude/SP500/USD_Index/VIX/OVX/Gold 等）。
    输出：买卖点信号、回测指标、影响因素，以及三张图的文件名。
    """
    if file.content_type is None or "csv" not in (file.content_type or "").lower():
        # content_type 有时不准，这里不直接拒绝；仅做提示性校验
        pass

    run_id = uuid.uuid4().hex
    out_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    csv_bytes = await file.read()
    config = RunConfig(
        forward_horizon=int(forward_horizon),
        plot_days=int(plot_days),
        cost_rate=float(cost_rate),
        red_percentile=float(red_percentile),
        yellow_percentile=float(yellow_percentile),
        top_factors=int(top_factors),
    )

    try:
        result = run_v3_from_csv_bytes(csv_bytes=csv_bytes, config=config, output_dir=out_dir)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})

    # 将生成文件复制到用户指定目录（便于你在原始 v3.py/CSV 目录直接查看）
    try:
        os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)
        target_run_dir = os.path.join(TARGET_OUTPUT_DIR, run_id)
        os.makedirs(target_run_dir, exist_ok=True)

        # 复制三张关键图
        for key in ["v3_shap_factor_importance.png", "v3_tiered_risk_signals.png", "trading_backtest_equity.png"]:
            src_path = os.path.join(out_dir, key)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, os.path.join(target_run_dir, key))

        # 复制结果 JSON（把你在前端看到的结构化信息落盘）
        with open(os.path.join(target_run_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        # 不让复制失败影响前端展示；你后续如果需要我也可以把报错细节加回来
        pass

    # 将图片文件名转换成可直接访问的 URL
    base_url = f"/static/results/{run_id}"
    images = result.get("images", {})
    result["images"] = {
        "shap_factor_importance": f"{base_url}/{images.get('shap_factor_importance')}",
        "tiered_risk_signals": f"{base_url}/{images.get('tiered_risk_signals')}",
        "equity_curve": f"{base_url}/{images.get('equity_curve')}",
    }

    result["run_id"] = run_id
    result["ok"] = True
    return result


# 挂载前端页面（放在 API 路由之后，避免 /api/... 被静态资源路由拦截）
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

