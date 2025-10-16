# pnn_model.py

import numpy as np

# 根據 PDF 為四種犬種建立理想特徵向量
# 數值是基於 PDF 描述的典型特徵所設定的理想分數
IDEAL_VECTORS = {
    "美國比特鬥牛犬 (APBT)": {
        "head_shape": 0.9, "body_is_square": 0.5, "muzzle_ratio": 0.9,
        "coat_is_short_stiff": 1.0, "stockiness": 0.6, "nose_is_black": 0.5, # 鼻子可以是任何顏色，黑色是其中一種可能
        "overall_apbt_amstaff_look": 1.0
    },
    "美國史大佛夏牛頭犬 (AmStaff)": {
        "head_shape": 1.0, "body_is_square": 0.5, "muzzle_ratio": 0.7,
        "coat_is_short_stiff": 1.0, "stockiness": 0.8, "nose_is_black": 1.0, # 鼻子絕對是黑色
        "overall_apbt_amstaff_look": 1.0
    },
    "史大佛夏牛頭犬 (SBT)": {
        "head_shape": 0.8, "body_is_square": 1.0, "muzzle_ratio": 0.6, # 身體呈正方形，吻部短
        "coat_is_short_stiff": 1.0, "stockiness": 0.7, "nose_is_black": 1.0, # 鼻子絕對是黑色
        "overall_apbt_amstaff_look": 0.5 # 體型較小，整體感不同
    },
    "美國惡霸犬 (American Bully)": {
        "head_shape": 1.0, "body_is_square": 0.2, "muzzle_ratio": 0.2, # 吻部極短
        "coat_is_short_stiff": 1.0, "stockiness": 1.0, "nose_is_black": 0.5, # 鼻子可以是任何顏色
        "overall_apbt_amstaff_look": 0.6 # 外觀誇張，與 APBT/AmStaff 有別
    }
}

REGULATED_BREEDS = ["美國比特鬥牛犬 (APBT)", "美國史大佛夏牛頭犬 (AmStaff)"]

def classify_breed(feature_dict):
    """
    計算輸入特徵與四種理想向量的距離，找出最可能的犬種。

    Args:
        feature_dict (dict): 從 VLM 提取的特徵字典。

    Returns:
        dict: 包含最可能犬種和是否受管制的字典。
    """
    if not feature_dict:
        return {"breed": "未知", "status": "無法分類"}

    feature_keys = sorted(feature_dict.keys())
    input_vector = np.array([feature_dict[k] for k in feature_keys])

    min_distance = float('inf')
    best_match_breed = "未知"

    for breed, ideal_feature_dict in IDEAL_VECTORS.items():
        ideal_vector = np.array([ideal_feature_dict[k] for k in feature_keys])
        distance = np.linalg.norm(input_vector - ideal_vector)
        print(f"與「{breed}」理想向量的距離: {distance:.4f}")

        if distance < min_distance:
            min_distance = distance
            best_match_breed = breed

    status = "管制犬種" if best_match_breed in REGULATED_BREEDS else "非管制犬種"
    
    return {"breed": best_match_breed, "status": status}
