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
    numeric_columns = ["lr", "loss", "d_model", "context_size", "n_layers", "n_heads", "d_head", "d_rope", "d_c", "d_cQ", 
                      "batch_size", "num_epochs"]
    for c in numeric_columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 整数で扱うカラムを指定
    integer_columns = ["context_size", "n_layers", "d_c", "d_cQ", "batch_size", "num_epochs"]
    for c in integer_columns:
        if c in df.columns:
            df[c] = df[c].astype('Int64')  # Int64はNaNを許容する整数型

    # 比較キー（存在するものを優先順で採用）- 実際のカラムから選択
    preferred_keys = ["type", "lr", "d_model", "context_size", "n_layers", "n_heads", "d_head", "d_rope", "d_c", "d_cQ", 
                     "batch_size", "num_epochs"]
    # 実際に存在するカラムのみをフィルタリング
    keys = [c for c in preferred_keys if c in df.columns]
    if not keys:
        # フォールバック: 時刻/メトリクス/フラグ以外をキーに
        keys = [c for c in df.columns if c not in ["timestamp", "loss", "attn"]]

    # 同一 (keys) に複数行がある場合は「最後の実行」を採用
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=keys, keep="last")

    # ★ 指定された順序で列を並べ替え
    desired_order = ["type", "lr", "loss", "d_model", "context_size", "n_layers", "n_heads", "d_head", "d_rope", "d_c", "d_cQ", 
                    "batch_size", "num_epochs", "timestamp"]
    # 実際に存在するカラムのみをフィルタリング
    display_order = [c for c in desired_order if c in df.columns]
    # 存在するがdesired_orderにないカラムを追加
    extra_cols = [c for c in df.columns if c not in display_order]
    display_order.extend(extra_cols)
    
    df = df[display_order]

    # 並べ替え（小さいloss順）
    if "loss" in df.columns:
        df = df.sort_values("loss", ascending=True, na_position="last").reset_index(drop=True)

    # ---- HTMLで罫線つき表示（Styler）----
    # フォーマット設定
    format_dict = {}
    if "loss" in df.columns:
        format_dict["loss"] = f"{{:.{float_prec}f}}"
    if "lr" in df.columns:
        format_dict["lr"] = f"{{:.{float_prec}e}}"  # lrを指数表記に
    
    # 整数カラムのフォーマット
    integer_format_cols = ["context_size", "n_layers", "d_c", "d_cQ", "batch_size", "num_epochs"]
    for col in integer_format_cols:
        if col in df.columns:
            format_dict[col] = "{:.0f}"
    
    sty = (df.style
           .format(format_dict)
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