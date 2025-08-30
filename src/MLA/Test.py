import torch

# @title create_causal_mask
def create_causal_mask(seq_len, device='cpu') -> torch.Tensor:
    # 形状: (T, T)。未来をTrueでマスク（上三角の+1オフセット）
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return torch.triu(ones, 1)



def test(model, text, freqs_cis, stoi, itos, max_seq_len=1000):

    input_ids = encode_text(text, stoi).to(freqs_cis.device)
    input_ids = input_ids.unsqueeze(0)  # バッチ次元を追加

    seq_len = input_ids.size(1)
    max_seq_len = max_seq_len - seq_len

    input_freqs_cis = freqs_cis[0:seq_len]

    for layer in model.layers:
        layer.attn.reset_kv_cache()

    model.eval()

    start_pos = seq_len

    for i in range(max_seq_len):
        seq_len = input_ids.size(1)
        # print("seq_len", seq_len)
        mask = create_causal_mask(seq_len, freqs_cis.device)

        # x = model.embedding(input_ids)  # [B, T, D]
        # for layer in model.layers:
        #     x = layer(x, input_freqs_cis, mask=mask, train=False)
        # x = model.norm(x)
        # logits = model.output_head(x)  # [B,T,V]

        logits = model(input_ids, freqs_cis=input_freqs_cis, mask=mask)  # (B, T, V)

        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = next_token_id.unsqueeze(0)
        input_freqs_cis = freqs_cis[start_pos:start_pos + 1]
        start_pos += 1

        print(decode_ids(next_token_id, itos), end="")
        

def decode_ids(ids, itos):
    return "".join(itos[int(i)] for i in ids)


def encode_text(text: str, stoi: dict):
    return torch.tensor([stoi[ch] for ch in text if ch in stoi], dtype=torch.long)



if __name__ == "__main__":
    text = "松江へ来て、" #@param{type:"string"}
    freqs_cis = precompute_freqs_cis(args, device=str(device))
    test(text, freqs_cis, stoi)
