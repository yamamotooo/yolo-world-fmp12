import shutil

from ultralytics import YOLO

# 推論時の入力サイズ(32の倍数)。大きいほど小さいカードに強いが遅くなる。
# build_vision_model.py は、このサイズを .mlmodel から自動で読み取る。
IMGSZ = 1280

# 1. YOLO-Worldモデルの読み込み
model = YOLO("yolov8s-world.pt")

# 2. 検出対象をトランプのカードに指定
# 種類(スート・数字)は区別しないので "card" の1クラスのみ。
# "playing card" だとトランプ画像でほぼ検出されなかったため "card" を使う。
# YOLO-Worldはテキスト埋め込みをモデルに焼き込むため、クラスを変えたい場合は再エクスポートが必要。
model.set_classes(["card"])

# 3. カスタムモデルとして一度保存
model.save("playing_cards_yolo.pt")

# 4. Core ML (.mlmodel) フォーマットへエクスポート
# FileMaker は .mlpackage を読み込めず .mlmodel のみ動作を確認済みのため、format="mlmodel" を使う
# nms=True にすることで、iOS側で扱いやすい行列データとして出力されます
exported = model.export(format="mlmodel", nms=True, imgsz=IMGSZ)

# export は元の重みファイル名(yolov8s-world.mlmodel)で出力されるため、リネームする
shutil.move(exported, "playing_cards_yolo.mlmodel")
print("saved: playing_cards_yolo.mlmodel")
