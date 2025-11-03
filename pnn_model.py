# pnn_model.py (最終版：8特徵 + 加權 + 懲罰 + 調紐 + 閾值)

import numpy as np

# 您的 8 特徵理想向量
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

# 您的特徵權重
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

# 管制犬懲罰
REGULATED_BREEDS = ["美國比特鬥牛犬 (APBT)", "美國史大佛夏牛頭犬 (AmStaff)"]
REGULATED_PENALTY_MULTIPLIER = 1.15

# 否決閾值
DISTANCE_THRESHOLD = 2.0 # 這是可調參數


def classify_breed(feature_dict, 
                   eye_toggle=1.0, 
                   nose_toggle=1.0, 
                   clothes_toggle=1.0):
    """
    計算輸入特徵與四種理想向量的「加權」距離，找出最可能的犬種。
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
                
                base_weight = ideal_weights[k]
                
                # 應用前端調紐
                final_weight = base_weight
                if k == "BlueEyesForbidden":
                    final_weight *= eye_toggle
                elif k == "BlackNoseRequired":
                    final_weight *= nose_toggle
                elif k == "BodyWeight_norm":
                    final_weight *= clothes_toggle
                
                weighted_distance_sq += ((input_val - ideal_val) ** 2) * final_weight
            
            distance = np.sqrt(weighted_distance_sq)
            
            # 應用管制犬懲罰
            if breed in REGULATED_BREEDS:
                distance *= REGULATED_PENALTY_MULTIPLIER
            
            print(f"與「{breed}」理想向量的(加權)距離: {distance:.4f}")

            if distance < min_distance:
                min_distance = distance
                best_match_breed = breed

        # 應用否決閾值
        if min_distance > DISTANCE_THRESHOLD:
            print(f"否決！最小距離 {min_distance:.4f} > 閾值 {DISTANCE_THRESHOLD}。")
            best_match_breed = "其他犬種" # 強制覆寫分類結果

        status = "管制犬種" if best_match_breed in REGULATED_BREEDS else "非管制犬種"
        
        return {"breed": best_match_breed, "status": status}
        
    except (ValueError, TypeError) as e:
        print(f"錯誤：無法將 VLM 的回傳值轉換為數字。錯誤訊息: {e}")
        print(f"收到的原始特徵值: {feature_dict}")
        return {"breed": "未知", "status": "分類失敗 (資料格式錯誤)"}
