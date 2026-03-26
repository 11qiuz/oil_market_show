import os
import json
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 🔥 修复：去掉点，服务器能正常导入
from pipeline import RunConfig, run_v3_from_csv_bytes


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
RESULTS_DIR = os.path.join(STATIC_DIR, "results")

# 本地保存路径（服务器无效，但不报错）
TARGET_BASE_DIR = r"D:\桌面\新建文件夹 (2)"
TARGET_OUTPUT_DIR = os.path.join(TARGET_BASE_DIR, "web_outputs_v3")

os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(title="Oil Quant AI Dashboard (V3)")

# 跨域完全放开（前端必用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 关键修复：关闭静态文件服务，不返回HTML
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    if file.content_type is None or "csv" not in (file.content_type or "").lower():
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

    # 本地保存（服务器不生效，但不崩溃）
    try:
        os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)
        target_run_dir = os.path.join(TARGET_OUTPUT_DIR, run_id)
        os.makedirs(target_run_dir, exist_ok=True)

        for key in ["v3_shap_factor_importance.png", "v3_tiered_risk_signals.png", "trading_backtest_equity.png"]:
            src_path = os.path.join(out_dir, key)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, os.path.join(target_run_dir, key))

        with open(os.path.join(target_run_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 🔥 关键修复：不返回图片URL，只返回JSON，绝对不出现HTML错误
    result["run_id"] = run_id
    result["ok"] = True
    return result


# 🔥 最关键修复：彻底关闭前端页面，绝不返回HTML！
# 下面整段全部删除/注释
# FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
# if os.path.isdir(FRONTEND_DIR):
#     app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
