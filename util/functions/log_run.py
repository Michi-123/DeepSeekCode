import os, csv
from datetime import datetime
LOG_PATH = "hp_results.csv"

import os, csv
from datetime import datetime
LOG_PATH = "hp_results.csv"

def log_run(args, losses_list, use_MLA, path=LOG_PATH):
    """
    実験1回分の結果をCSVに1行追記。
    use_MLA=False -> 'MLA'、True -> 'MHA' として保存します。
    """
    final_loss = float(losses_list[-1])
    attn_type = "MLA" if use_MLA else "MHA"
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        # 使っていなければ消してOK
        "d_head": getattr(args, "d_head", None),
        "d_rope": getattr(args, "d_rope", None),
        "d_c": getattr(args, "d_c", None),
        "d_cQ": getattr(args, "d_cQ", None),
        "type": attn_type,                  # ★ typeカラムに変更
        "loss": final_loss,                 # ★ final_loss → lossに変更
        "batch_size": getattr(args, "batch_size", None),  # ★ 追加
        "lr": getattr(args, "lr", None),                  # ★ 追加
        "num_epochs": getattr(args, "num_epochs", None),  # ★ 追加
    }
    header = ["timestamp", "d_model", "n_heads", "d_head", "d_rope", "d_c", "d_cQ", 
              "type", "loss", "batch_size", "lr", "num_epochs"]  # ★ ヘッダー更新
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)


def show_results(path=LOG_PATH, sort_by="loss"):  # ★ sort_byを"loss"に変更
    """CSVを読み込んで簡易表示（loss昇順）"""
    if not os.path.exists(path):
        print("まだ結果ファイルがありません。")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r[sort_by]))
    # 表示
    cols = rows[0].keys()
    print(" | ".join(cols))
    print("-" * 80)
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))

def show_results(path=LOG_PATH, sort_by="final_loss"):
    """CSVを読み込んで簡易表示（final_loss昇順）"""
    if not os.path.exists(path):
        print("まだ結果ファイルがありません。")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r[sort_by]))
    # 表示
    cols = rows[0].keys()
    print(" | ".join(cols))
    print("-" * 80)
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))