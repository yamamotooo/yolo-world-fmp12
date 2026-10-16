"""samples/ のカード画像を Gemini で読み取り、正解 (同名 .json) と照合する。

使い方: python verify_samples.py
必要な環境変数 (.env): GEMINI_API_KEY。モデルは GEMINI_MODEL で変更できる (card_lookup.py)。
"""
import glob
import json
import os
import re

from card_lookup import MODEL, load_env, read_card

TITLE = re.compile(r"^(?P<name>.+?)\((?P<rarity>.*?)\)\{.*?\}〈(?P<number>\d+/\d+)〉\[(?P<set>.+?)\]$")


def main():
    load_env()
    print("model:", MODEL)
    ok = 0
    files = sorted(glob.glob("samples/*.webp"))
    for img in files:
        truth = json.load(open(img[:-5] + ".json", encoding="utf-8"))
        m = TITLE.match(truth["title"])
        got = read_card(open(img, "rb").read())
        checks = {
            "number": got["number"] == m["number"],
            "set": got["set_code"] == m["set"],
            "rarity": got["rarity_mark"] == (m["rarity"] if m["rarity"] != "-" else ""),
            "name": got["name_ja"] == m["name"],
        }
        ok += all(checks.values())
        print(f"{'OK ' if all(checks.values()) else 'NG '} {os.path.basename(img)[:10]} 正解={m['name']} {m['number']} [{m['set']}] ({m['rarity']})")
        print(f"     読取={got['name_ja']} / {got['name_en']} {got['number']} [{got['set_code']}] ({got['rarity_mark']}) conf={got['confidence']}"
              + "  ※不一致: " + ",".join(k for k, v in checks.items() if not v) * (not all(checks.values())))
    print(f"\n{ok}/{len(files)} 枚が全項目一致")


if __name__ == "__main__":
    main()
