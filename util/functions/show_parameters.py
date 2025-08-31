#@title show_parameters
import os
import pandas as pd
from IPython.display import display

def show_parameters(path="hp_results.csv", float_prec=5):
    if not os.path.exists(path):
        print(f"結果ファイルがありません: {path}")
        return

    df = pd.read_csv(path)

    # 旧ログ互換 & 正規化
    if "type" not in df.columns and "attn" in df.columns:
        df["type"] = df["attn"]
    elif "type" not in df.columns:
        df["type"] = "MLA"
    
    # 型の正規化
    df["type"] = df["type"].map({True: "MHA", False: "MLA"}).fillna(df["type"])
    df["type"] = df["type"].replace({"attn": "MHA"})  # 旧バージョンの互換性

    # 数値化（ある列だけ）
    numeric_columns = ["d_model", "n_heads", "d_head", "d_rope", "d_c", "d_cQ", 
                      "loss", "batch_size", "lr", "num_epochs"]
    for c in numeric_columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 比較キー（存在するものを優先順で採用）
    preferred_keys = ["d_model", "n_heads", "d_head", "d_rope", "d_c", "d_cQ", 
                     "batch_size", "lr", "num_epochs"]
    keys = [c for c in preferred_keys if c in df.columns]
    if not keys:
        # フォールバック: 時刻/メトリクス/フラグ以外をキーに
        keys = [c for c in df.columns if c not in ["timestamp", "type", "loss"]]

    # 同一 (keys, type) に複数行がある場合は「最後の実行」を採用
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=keys + ["type"], keep="last")

    # 列順を整える
    order = [c for c in preferred_keys if c in df.columns] + ["type", "loss"]
    df = df[order]

    # 並べ替え（小さいloss順）
    if "loss" in df.columns:
        df = df.sort_values("loss", ascending=True, na_position="last").reset_index(drop=True)

    # ---- HTMLで罫線つき表示（Styler）----
    sty = (df.style
           .format({"loss": f"{{:.{float_prec}f}}"})
           .hide(axis="index")
           .set_table_styles([
               {"selector": "table",
                "props": [("border-collapse", "collapse"), ("border", "1px solid #888")]},
               {"selector": "th, td",
                "props": [("border", "1px solid #888"), ("padding", "6px 10px")]},
               {"selector": "thead th",
                "props": [("background", "#000000"), ("font-weight", "600")]}
           ]))

    display(sty)