#@title show_compare_MLA_MHA_html
import os
import pandas as pd
from IPython.display import display

def show_compare_MLA_MHA(path="hp_results.csv", float_prec=5):
    if not os.path.exists(path):
        print(f"結果ファイルがありません: {path}")
        return

    df = pd.read_csv(path)

    # 旧ログ互換 & 正規化
    if "attn" not in df.columns:
        df["attn"] = "MLA"
    df["attn"] = df["attn"].map({True: "MHA", False: "MLA"}).fillna(df["attn"])

    # 数値化（ある列だけ）
    for c in ["d_model","n_heads","d_head","d_rope","d_c","d_cQ","final_loss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 比較キー（存在するものを優先順で採用）
    preferred_keys = ["d_model","n_heads","d_head","d_rope","d_c","d_cQ"]
    keys = [c for c in preferred_keys if c in df.columns]
    if not keys:
        # フォールバック: 時刻/メトリクス/フラグ以外をキーに
        keys = [c for c in df.columns if c not in ["timestamp","attn","final_loss"]]

    # 同一 (keys, attn) に複数行がある場合は「最後の実行」を採用
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=keys + ["attn"], keep="last")

    # MLA と MHA を別々にしてから横結合（ピボット不使用で安全）
    cols_for_side = keys + ["final_loss"]
    mla = (df[df["attn"] == "MLA"][cols_for_side]
           .rename(columns={"final_loss": "MLA_loss"}))
    mha = (df[df["attn"] == "MHA"][cols_for_side]
           .rename(columns={"final_loss": "MHA_loss"}))

    out = pd.merge(mla, mha, on=keys, how="outer")

    # 列順を整える
    order = [c for c in preferred_keys if c in out.columns] + \
            [c for c in ["MLA_loss","MHA_loss"] if c in out.columns]
    out = out[order]

    # 並べ替え（小さいloss順）
    sort_cols = [c for c in ["MLA_loss","MHA_loss"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=True, na_position="last").reset_index(drop=True)

    # ---- HTMLで罫線つき表示（Styler）----
    sty = (out.style
           .format({c: f"{{:.{float_prec}f}}" for c in ["MLA_loss","MHA_loss"] if c in out.columns})
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
