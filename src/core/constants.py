from __future__ import annotations

NUMBERS = ("1", "3", "4", "5", "6", "9", "10", "11", "12")  # index = class id
NUMBER_TO_IDX = {n: i for i, n in enumerate(NUMBERS)}
BIB_OF_ROSTER = {frozenset({"6", "9", "11"}): "green", frozenset({"6", "9", "10"}): "yellow",
                 frozenset({"1", "3", "5"}): "pink", frozenset({"4", "10", "12"}): "purple"}
CHALLENGE_VIDEOS = frozenset({"IMG_0114", "IMG_0115"})
FORBIDDEN_MODULES = ("mmdet", "mmcv", "mmengine", "mmpose", "mmocr", "doctr", "strhub",
                     "ultralytics", "groundingdino")
