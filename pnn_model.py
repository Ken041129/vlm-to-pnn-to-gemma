# pnn_model.py (已升級為 8 特徵 + 加權 + 前端調紐)

import numpy as np

# 您的 8 特徵理想向量 (保持不變)
IDEAL_VECTORS = {
    "美國比特鬥牛犬 (APBT)": {
        "ShoulderHeight_norm": 0.78, "BodyWeight_norm": 0.60, "MuzzleHeadRatio": 0.70,
        "BlackNoseRequired": 0.00, "BlueEyesForbidden": 1.00, "ChestWidthDepth": 0.55,
        "BodySquareness": 0.55, "HeadBreadthIndex": 0.60,
    },
    "美國史大佛夏牛頭犬 (AmStaff)": {
        "ShoulderHeight_norm": 0.75, "BodyWeight_norm": 0.75, "MuzzleHeadRatio": 0.45,
        "BlackNoseRequired": 1.00, "BlueEyesForbidden": 1.00, "ChestWidthDepth": 0.65,
        "BodySquareness": 0.60, "HeadBreadthIndex": 0.65
    },
    "史大佛夏牛頭犬 (SBT)": {
        "ShoulderHeight_norm": 0.40, "BodyWeight_norm": 0.70, "MuzzleHeadRatio": 0.50,
        "BlackNoseRequired": 1.00, "BlueEyesForbidden": 1.00, "ChestWidthDepth": 0.80,
        "BodySquareness": 0.65, "HeadBreadthIndex": 0.75
    },
    "美國惡霸犬 (American Bully)": {
        "ShoulderHeight_norm": 0.35, "BodyWeight_norm": 0.95, "MuzzleHeadRatio": 0.35,
        "BlackNoseRequired": 0.00, "BlueEyesForbidden": 1.00, "ChestWidthDepth": 0.95,
        "BodySquareness": 0.80, "HeadBreadthIndex": 0.95,
    },
}

# 您的特徵權重 (保持不變)
IDEAL_VECTORS_WEIGHTS = {
    "美國比特鬥牛犬 (APBT)": {
        "ShoulderHeight_norm": 1.0, "BodyWeight_norm": 1.0, "MuzzleHeadRatio": 1.0,
        "BlackNoseRequired": 0.0, "BlueEyesForbidden": 1.0, "ChestWidthDepth": 1.0,
        "BodySquareness": 1.0, "HeadBreadthIndex": 1.0,
    },
    "美國史大佛夏牛頭犬 (AmStaff)": {
        "ShoulderHeight_norm": 1.0, "BodyWeight_norm": 1.0, "MuzzleHeadRatio": 1.0,
        "BlackNoseRequired": 1.0, "BlueEyesForbidden": 1.0, "ChestWidthDepth": 1.0,
        "BodySquareness": 1.0, "HeadBreadthIndex": 1.0
    },
    "史大佛夏牛頭犬 (SBT)": {
        "ShoulderHeight_norm": 1.0, "BodyWeight_norm": 1.0, "MuzzleHeadRatio": 1.0,
        "BlackNoseRequired": 1.0, "BlueEyesForbidden": 1.0, "ChestWidthDepth": 1.0,
        "BodySquareness": 1.0, "HeadBreadthIndex": 1.0
    },
    "美國惡霸犬 (American Bully)": {
        "ShoulderHeight_norm": 1.0, "BodyWeight_norm": 1.0, "MuzzleHeadRatio": 1.0,
        "BlackNoseRequired": 0.0, "BlueEyesForbidden": 1.0, "ChestWidthDepth": 1.0,
        "BodySquareness": 1.0, "HeadBreadthIndex": 1.0,
    },
}

REGULATED_BREEDS = ["美國比特鬥牛犬 (APBT)", "美國史大佛夏牛頭犬 (AmStaff)"]
REGULATED_PENALTY_MULTIPLIER = 1.15

# --- 1. 關鍵修改：修改函式簽名 ---
# 我們加入了三個新的參數，並給予 1.0 (代表"啟用/可信") 的預設值
def classify_breed(feature_dict, 
                   eye_toggle=1.0, 
                   nose_toggle=1.0, 
                   clothes_toggle=1.0):
    """
    計算輸入特徵與四種理想向量的「加權」距離，找出最可能的犬種。

    Args:
        feature_dict (dict): 從 VLM 提取的特徵字典。
        eye_toggle (float): 前端傳來的 "眼睛" 可信度 (0.0=不可信, 1.0=可信)。
        nose_toggle (float): 前端傳來的 "鼻子" 可信度 (0.0=不可信, 1.0=可信)。
        clothes_toggle (float): 前端傳來的 "衣服" 影響 (0.0=有穿/不可信, 1.0=沒穿/可信)。

    Returns:
        dict: 包含最可能犬種和是否受管制的字典。
    """
    if not feature_dict:
        return {"breed": "未知", "status": "無法分類"}

    try:
        feature_keys = sorted(feature_dict.keys())
        
        expected_keys = sorted(IDEAL_VECTORS["美國比特鬥牛犬 (APBT)"].keys())
        if feature_keys != expected_keys:
            print(f"錯誤：VLM 回傳的特徵鍵 ({feature_keys}) 與 PNN 預期的鍵 ({expected_keys}) 不符。")
            return {"breed": "未知", "status": "分類失敗 (特徵鍵不匹配)"}

        input_vector_dict = {k: float(feature_dict[k]) for k in feature_keys}

        min_distance = float('inf')
        best_match_breed = "未知"

        for breed, ideal_feature_dict in IDEAL_VECTORS.items():
            ideal_vector = ideal_feature_dict
            ideal_weights = IDEAL_VECTORS_WEIGHTS[breed] 
            
            weighted_distance_sq = 0.0 

            for k in feature_keys:
                input_val = input_vector_dict[k]
                ideal_val = ideal_vector[k]
                
                # 獲取基礎權重
                base_weight = ideal_weights[k]
                
                # --- 2. 關鍵修改：應用前端調紐 ---
                final_weight = base_weight
                if k == "BlueEyesForbidden":
                    final_weight *= eye_toggle
                elif k == "BlackNoseRequired":
                    final_weight *= nose_toggle
                elif k == "BodyWeight_norm":
                    final_weight *= clothes_toggle
                # --- 結束修改 ---
                
                # 計算單一特徵的加權距離
                weighted_distance_sq += ((input_val - ideal_val) ** 2) * final_weight
            
            distance = np.sqrt(weighted_distance_sq)
            
            if breed in REGULATED_BREEDS:
                distance *= REGULATED_PENALTY_MULTIPLIER
            
            print(f"與「{breed}」理想向量的(加權)距離: {distance:.4f}")

            if distance < min_distance:
                min_distance = distance
                best_match_breed = breed

        status = "管制犬種" if best_match_breed in REGULATED_BREEDS else "非管制犬種"
        
        return {"breed": best_match_breed, "status": status}
        
    except (ValueError, TypeError) as e:
        print(f"錯誤：無法將 VLM 的回傳值轉換為數字。錯誤訊息: {e}")
        print(f"收到的原始特徵值: {feature_dict}")
        return {"breed": "未知", "status": "分類失敗 (資料格式錯誤)"}
