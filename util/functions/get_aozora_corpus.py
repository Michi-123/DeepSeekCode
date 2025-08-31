import re
import requests

CORPUS_DIR     = "corpus"

def get_aozora_corpus(urls):
    # 保存先ディレクトリ
    os.makedirs(CORPUS_DIR, exist_ok=True)

    texts = []

    for title, url in urls.items():
        print(f"Downloading: {title} ({url})")
        res = requests.get(url)
        res.encoding = "shift_jis"  # 青空文庫はShift_JISが多い

        html = res.text

        # ルビ除去 （例：外（はず）れ → 外れ）
        text = re.sub(r'（[^）]*）', '', html)

        # HTMLタグを削除
        clean_text = re.sub(r'<[^>]+>', '', text)

        texts.append(clean_text)

    # <eos> で結合して一つのコーパスに
    corpus = " <eos> ".join(texts)


    corpus = corpus.replace('\r','').replace('\n','').replace('\t','').replace('\u3000','')

    return corpus