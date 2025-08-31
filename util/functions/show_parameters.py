#@title show_show_parameters
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

    # MLA と MHA を別々にしてから横結合（ピボット不使用で安全）
    cols_for_side = keys + ["loss"]
    mla = (df[df["type"] == "MLA"][cols_for_side]
           .rename(columns={"loss": "MLA_loss"}))
    mha = (df[df["type"] == "MHA"][cols_for_side]
           .rename(columns={"loss": "MHA_loss"}))

    out = pd.merge(mla, mha, on=keys, how="outer")

    # 列順を整える
    order = [c for c in preferred_keys if c in out.columns] + \
            [c for c in ["MLA_loss", "MHA_loss"] if c in out.columns]
    out = out[order]

    # 並べ替え（小さいloss順）
    sort_cols = [c for c in ["MLA_loss", "MHA_loss"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=True, na_position="last").reset_index(drop=True)

    # ---- HTMLで罫線つき表示（Styler）----
    sty = (out.style
           .format({c: f"{{:.{float_prec}f}}" for c in ["MLA_loss", "MHA_loss"] if c in out.columns})
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