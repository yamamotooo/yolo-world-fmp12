"""カード画像 -> Gemini で読み取り -> PokéWallet で価格取得 -> draw_boxes.html の setCardInfo 用 JSON。

使い方: python card_lookup.py 画像1.webp 画像2.webp ...   (結果の JSON を標準出力へ)
必要な環境変数 (.env): GEMINI_API_KEY, POKEWALLET_API_KEY。
任意: GEMINI_MODEL (既定 gemini-3.1-flash-lite)、USD_JPY (指定すると為替 API を呼ばずこの値を使う)。
"""
import base64
import json
import os
import sys

import requests

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
POKEWALLET = "https://api.pokewallet.io"
MIN_CONFIDENCE = 0.8   # これ未満の読み取りは価格を出さず「要確認」にする

PROMPT = """これはポケモンカード(日本語版)1枚の画像です。印字されている内容だけを読み取ってください。推測や補完はしないこと。
- name_ja: カード名 (ex などの接尾辞を含む。「LV.12」などは含めない)
- name_en: 英語名 (ポケモン名の公式英語名。トレーナーズ等で不明なら空文字)
- set_code: 左下付近の角丸の枠内に入った、英数字のセット記号 (例 M6a, MF)。単独の「J」は言語マークなのでセット記号ではない。無ければ空文字
- number: カード番号 (例 019/103)。無ければ空文字
- rarity_mark: 番号の右に印字されたレアリティ記号 (RR, SR, AR など)。セット記号は含めない。印字が無ければ空文字
- confidence: 0〜1。番号やセット記号が読みにくいときは下げる"""

SCHEMA = {
    "type": "OBJECT",
    "properties": {k: {"type": "STRING"} for k in ["name_ja", "name_en", "set_code", "number", "rarity_mark"]}
    | {"confidence": {"type": "NUMBER"}},
    "required": ["name_ja", "name_en", "set_code", "number", "rarity_mark", "confidence"],
}


def load_env(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def read_card(image_bytes, mime="image/webp"):
    """カード画像 1 枚を Gemini で読み取る。"""
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": mime, "data": base64.b64encode(image_bytes).decode()}},
            {"text": PROMPT}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA, "temperature": 0},
    }
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
                      headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, json=body, timeout=120)
    r.raise_for_status()
    return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])


def usd_jpy():
    if os.environ.get("USD_JPY"):
        return float(os.environ["USD_JPY"])
    r = requests.get("https://api.frankfurter.dev/v1/latest", params={"base": "USD", "symbols": "JPY"}, timeout=15)
    r.raise_for_status()
    return r.json()["rates"]["JPY"]


def find_card(name_en, number, set_code=""):
    """PokéWallet を「英語名 番号」で検索し、番号が一致して名前が合うカードを返す。無ければ None。

    セット記号は小さくて誤読しやすいので、検索には使わない(複数件のときの絞り込みにだけ使う)。
    「セット記号 番号」の検索は、カード名に番号が入るカード(例 "Pikachu - 019/103")にしか当たらない。
    """
    r = requests.get(f"{POKEWALLET}/search", params={"q": f"{name_en} {number}", "limit": 10},
                     headers={"X-API-Key": os.environ["POKEWALLET_API_KEY"]}, timeout=30)
    r.raise_for_status()
    hits = [c for c in r.json().get("results", [])
            if c["card_info"].get("card_number") == number
            and c["card_info"].get("name", "").lower().startswith(name_en.lower())]   # 数字の誤読で別のカードに当たるのを防ぐ
    if len(hits) > 1 and set_code:
        hits = [c for c in hits if c["card_info"].get("set_code", "").lower() == set_code.lower()]
    return hits[0] if len(hits) == 1 else None   # 0 件も複数件も曖昧なので価格は出さない


def lookup(read, rate):
    """読み取り結果 -> setCardInfo 用の 1 件。価格が取れなければ price を入れない。"""
    info = {"name": read["name_ja"], "number": read["number"], "set_code": read["set_code"],
            "rarity": read["rarity_mark"], "confidence": read["confidence"]}
    if not read["name_en"] or not read["number"]:
        info["note"] = "英語名または番号を読み取れない"
    elif read["confidence"] < MIN_CONFIDENCE:
        info["note"] = "読み取りの信頼度が低い"
    else:
        card = find_card(read["name_en"], read["number"], read["set_code"])
        if card:
            info["set_code"] = card["card_info"]["set_code"]   # セット記号は API の値を使う(誤読しやすいため)
        prices = [p for p in ((card or {}).get("tcgplayer") or {}).get("prices", []) if p.get("market_price") is not None]
        if not card:
            info["note"] = "PokéWallet で一意に特定できない"
        elif not prices:
            info["note"] = "市場価格なし"
        else:
            p = prices[0]
            info.update(price=round(p["market_price"] * rate), price_usd=p["market_price"],
                        variant=p["sub_type_name"], usd_jpy=rate, rarity=info["rarity"] or card["card_info"].get("rarity") or "")
    return info


if __name__ == "__main__":
    load_env()
    rate = usd_jpy()
    out = []
    for i, path in enumerate(sys.argv[1:]):
        mime = {"webp": "image/webp", "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}[path.rsplit(".", 1)[1].lower()]
        out.append({"index": i, **lookup(read_card(open(path, "rb").read(), mime), rate)})
    print(json.dumps(out, ensure_ascii=False, indent=2))
