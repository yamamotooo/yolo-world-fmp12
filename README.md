# yolo-world

YOLO-World(ゼロショット物体検出)でトランプのカードを検出し、FileMaker から座標を取得して描画する。

## 構成

| ファイル | 役割 |
|---|---|
| `convert.py` | `yolov8s-world.pt` に "card" を設定し、`playing_cards_yolo.pt` と `playing_cards_yolo.mlmodel`(NMS 入り)を書き出す |
| `build_vision_model.py` | `playing_cards_yolo.mlmodel` を、FileMaker の Vision で座標が取れる `playing_cards_vision.mlmodel` に変換する |
| `draw_boxes.html` | `ComputeModel` の結果から、画像の上にボックスを描く(FileMaker の Web ビューア用) |

## 手順

Python 3.9 以降。

```
python3 -m venv .venv && . .venv/bin/activate
pip install ultralytics coremltools "git+https://github.com/ultralytics/CLIP.git"

python convert.py
python build_vision_model.py playing_cards_yolo.mlmodel playing_cards_vision.mlmodel
```

`convert.py` は、`yolov8s-world.pt` を自動でダウンロードする。macOS が必要(Core ML の書き出しと確認のため)。生成したモデルは、このリポジトリには含まれない。

`playing_cards_vision.mlmodel` を、FileMaker のコンテナフィールドに入れる。

## FileMaker 側の仕様

- スクリプト: `機械学習モデルを構成 [ 処理: Vision ; 名前: "yolo" ; 次から: テーブル::モデル ]`
- 呼び出し: `ComputeModel ( "yolo" ; テーブル::写真 )`。オプションの `confidenceLowerLimit` は **0.25**(座標ラベルの確率が 0.5 のため、有効なラベルだけが返る)。
- 入力は、任意サイズの画像。モデルの先頭で、書き出し時の入力サイズ(`convert.py` の `IMGSZ`、現在 1280)にリサイズする。

### 結果の形式

分類結果の JSON 配列。各要素の `classification` が `カード番号:軸:値` で、`confidence` が 0.5(有効なラベル)。

```
[{"classification":"0:y2:0.415","confidence":0.5}, {"classification":"3:x2:0.615","confidence":0.5}, ...]
```

- カード番号: 0〜9。0 が最も信頼度の高いカード(最大 10 枚)。順序は並び順に依存しない(接頭辞で判別する)。
- 軸: `x1`、`y1`、`x2`、`y2`(左上と右下)、`c`(信頼度)。
- 値: 0〜1 の正規化値。0.005 刻み(1280px の画像で約 6px)。ピクセルにするには、画像の幅・高さを掛ける。

### 描画

`draw_boxes.html` を Web ビューアに渡す。`__IMAGE__` を `"data:image/jpeg;base64," & Base64EncodeRFC ( 4648 ; テーブル::写真 )`、`__JSON__` を `ComputeModel` の結果で Substitute する。引数の順は、RFC 番号が先で、データが後。逆にすると、写真ではなく `4648` の文字列が変換されて、何も描画されない。

注意: `Base64Encode` などで改行が入っても、HTML 側で空白を除去するので問題ない。プレースホルダは、`script` タグの中の2か所だけに書く。コメントなどに書くと、Substitute で全部置換されて、JavaScript が構文エラーになり、何も描画されない。

## 設計上の注意(動作確認の結果)

FileMaker の機械学習機能は制約が多く、次の方式でのみ動作を確認した(2026-10-15、実機)。

- **`.mlpackage` は読み込めない。** `.mlmodel`(NeuralNetwork)だけが読み込める。`format="mlmodel"` で書き出す。`coremltools` で `.mlpackage` を `.mlmodel` に変換することはできない(ML Program は `.mlmodel` で保存できない)。
- **配列や double の出力は使えない。** 多次元配列の出力は読み込みエラー、double の出力は `ComputeModel` が `?` を返す(エラーコードは 0)。
- **「処理: 一般」は、画像入力のモデルで `?` になる。** 「処理: Vision」を使う。
- **Vision は分類結果(ラベル+確率)だけを返す。** そのため、座標を「ラベル+確率」にエンコードしている(`build_vision_model.py`)。各座標の値を 201 段階に量子化し、該当するラベルだけに確率 0.5 を与える。
- 確率は **0.25 ではなく 0.5** にしている。softmax の誤差で 0.24999… となり、しきい値 0.25 で落ちたため。
- `GetModelAttributes` の出力は、型にかかわらず `unknown` と表示される(判断材料にならない)。

## モデルの仕様

- クラスは `"card"` の1つのみ。**`"playing card"` だと、トランプ画像でほぼ検出されない**(ゼロショットの限界)。種類(スート・数字)は区別しない。
- 入力サイズは `convert.py` の `IMGSZ`(32 の倍数、現在 1280)。大きいほど小さいカードに強いが、遅くなり、モデルも大きくなる。`build_vision_model.py` は、サイズを `.mlmodel` から自動で読み取る。YOLO-World は 640 で学習されているので、大きくしても精度が上がるとは限らない。
- YOLO-World は、テキストの埋め込みをモデルに焼き込む。クラスを変えるときは、再エクスポートが必要。
- `build_vision_model.py` の固定値: 最大 10 枚(`K_CARDS`)、検出しきい値 0.05(`SCORE`)、IoU 0.45(`IOU`)、刻み 201 段階(`N`)。実行時には変えられないので、変えるときは再生成する。
- NMS は、モデル内の NeuralNetwork の NMS レイヤーで行う。空きスロット(スコア 0)は、確率が 0 になる。

## 既知の制約

- ゼロショットの精度は高くない。重なったカードは、まとめて1つのボックスになることがある。精度が必要なら、カード画像でのファインチューニングが必要。
- `coremltools` は、torch 2.8.0 を未検証として警告を出す(変換自体は動作する)。
- 座標は、画像を `IMGSZ` 四方に引き伸ばして推論した結果。アスペクト比は保持していない(座標は正規化値なので、元画像にそのまま掛けられる)。

## ライセンス

AGPL-3.0(`LICENSE`)。依存する [Ultralytics](https://github.com/ultralytics/ultralytics) が AGPL-3.0 のため、それに合わせている。生成したモデルも、同ライセンスの影響を受ける可能性がある。
