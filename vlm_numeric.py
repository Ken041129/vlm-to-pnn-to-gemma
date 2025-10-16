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
    使用 LLaVA VLM 從圖片中提取更詳細、具區分性的犬隻特徵數值。
    """
    if not os.path.exists(image_path):
        print(f"錯誤：圖片路徑不存在 {image_path}")
        return None

    b64_image = image_to_base64(image_path)

    # 升級版 Prompt，要求 LLaVA 提供更細緻的特徵分數以區分不同犬種
    prompt = """
    你是一位專業的犬隻品種鑑定員，你的任務是嚴格根據我提供的「比特型犬種特徵指南」來分析圖片中的狗。
    請仔細觀察圖片，並根據以下每一項特徵的符合程度給予一個 0.0 到 1.0 之間的分數。
    0.0 表示「完全不符合」，1.0 表示「完美符合」。

    特徵指南如下：
    1.  **head_shape**: 頭部正面呈圓形，側面呈楔形，臉頰因咬肌而突出。
    2.  **body_is_square**: 身體側面是否為正方形（肩高=身長）。1.0 代表完全正方（像史大佛夏牛頭犬），0.5 代表身長略大於肩高（像 APBT/AmStaff），0.2 代表明顯更長或更矮壯。
    3.  **muzzle_ratio**: 吻部長度與顱骨長度的比例。1.0 代表約 2:3 的標準比例（像 APBT），0.6 代表吻部較短（像 AmStaff/SBT），0.2 代表吻部極短呈立方體（像美國惡霸犬）。
    4.  **coat_is_short_stiff**: 毛髮是否為短、硬、粗的單層毛。
    5.  **stockiness**: 身材的壯碩與肌肉感。0.5 為精實，1.0 為非常壯碩粗獷（像美國惡霸犬）。
    6.  **nose_is_black**: 鼻子是否「絕對是黑色」。1.0 代表絕對是黑色，0.0 代表是其他顏色（如粉色、肝色），0.5 代表因光線或解析度無法確定。
    7.  **overall_apbt_amstaff_look**: 整體外觀符合管制犬種（APBT/AmStaff）的程度。
    
    請將你的分析結果以一個嚴格的 JSON 格式回傳，不要有任何額外的說明或文字。
    JSON 格式範例：
    {
      "head_shape": 0.8,
      "body_is_square": 0.5,
      "muzzle_ratio": 0.9,
      "coat_is_short_stiff": 1.0,
      "stockiness": 0.6,
      "nose_is_black": 0.0,
      "overall_apbt_amstaff_look": 0.8
    }
    """

    try:
        print(f"正在呼叫 LLaVA 分析圖片: {os.path.basename(image_path)}...")
        response = ollama.chat(
            model='llava',
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
        
        print(f"LLaVA 分析完成。")
        return features

    except Exception as e:
        print(f"從 LLaVA 獲取特徵時發生錯誤: {e}")
        print(f"LLaVA 原始回傳內容: {response_content if 'response_content' in locals() else 'N/A'}")
        return None
