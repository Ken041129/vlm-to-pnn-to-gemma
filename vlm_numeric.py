# vlm_numeric.py

import ollama
import base64
import json
import os

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_features_from_vlm(image_path):
    """
    使用 VLM 從圖片中提取更詳細、具區分性的犬隻特徵數值。
    """
    if not os.path.exists(image_path):
        print(f"錯誤：圖片路徑不存在 {image_path}")
        return None

    b64_image = image_to_base64(image_path)

    # 中文版提示（特徵名稱維持英文）
    prompt = """
        你是一位專業的犬隻品種鑑定員。請分析圖片中的狗，針對以下每一個特徵給予 0.0 到 1.0 的分數：
        0.0 代表「完全不符合」，1.0 代表「完全符合」。
        請只回傳嚴格的 JSON 格式，不要包含其他文字或說明。

        特徵說明如下：
        1) ShoulderHeight_norm：目視推估肩高的相對值，正規化到 0~1（越高越接近 1，越矮越接近 0）。
        2) BodyWeight_norm：目視推估體重或肌肉量的相對值，正規化到 0~1（越壯或重越接近 1）。
        3) MuzzleHeadRatio：吻長 ÷ 頭長。長吻（如 APBT）≈ 高分；短吻或立方（如 Bully）≈ 低分。
        4) BlackNoseRequired：鼻子是否明顯為黑色。黑色=1，其他顏色=0，不確定=0.5。
        5) BlueEyesForbidden：眼睛是否非藍色。若明顯不是藍色=1，藍眼=0，不確定=0.5。
        6) ChestWidthDepth：胸寬 ÷ 胸深。胸寬小於胸深（如 APBT）≈ 低分，胸寬與胸深相近或較寬≈ 高分。
        7) BodySquareness：身體比例是否接近方形。肩高≈身長=1，身長略大於肩高≈0.4，明顯長身或低矮≈更低。
        8) HeadBreadthIndex：頭部寬度與方正度。顱骨寬闊、立方感重（如 Bully）≈ 高分；楔形或較窄（如 APBT）≈ 低分。

        請嚴格依下列 JSON 格式回傳：
        {
          "ShoulderHeight_norm": 0.0,
          "BodyWeight_norm": 0.0,
          "MuzzleHeadRatio": 0.0,
          "BlackNoseRequired": 0.0,
          "BlueEyesForbidden": 0.0,
          "ChestWidthDepth": 0.0,
          "BodySquareness": 0.0,
          "HeadBreadthIndex": 0.0
        }
    """

    try:
        print(f"正在呼叫 VLM 分析圖片: {os.path.basename(image_path)}...")
        response = ollama.chat(
            model='gemma3:27b-it-qat',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [b64_image]
                }
            ]
        )
        
        response_content = response['message']['content']
        json_str = response_content.strip().replace('```json', '').replace('```', '')
        features = json.loads(json_str)
        
        print(f"VLM 分析完成。")
        return features

    except Exception as e:
        print(f"從 VLM 獲取特徵時發生錯誤: {e}")
        print(f"VLM 原始回傳內容: {response_content if 'response_content' in locals() else 'N/A'}")
        return None
