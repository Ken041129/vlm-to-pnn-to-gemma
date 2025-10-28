# gemma_report.py 

import ollama

# 擴充知識庫，包含所有四種犬種的關鍵特徵
PDF_KNOWLEDGE = {
    "美國比特鬥牛犬 (APBT)": "管制犬種。特徵：頭呈楔形（長三角柱型），兩耳之間顱骨寬平或略圓，吻部與頭顱長度比例約為2:3。高耳位。眼睛可為除了藍色以外的所有顏色，中等大小，圓形。鼻子大且鼻孔寬，鼻子可以是任何顏色。胸腔寬度不超過其深度",
    "美國史大佛夏牛頭犬 (AmStaff)": "管制犬種。特徵：顱骨寬，吻部背側呈圓弧狀，嘴唇平貼不鬆弛。高耳位的短小玫瑰耳或半折耳，少許耳尖向外側或向前折，可看見耳道。眼睛深色且圓。鼻子絕對是黑色的。短而濃密的毛",
    "史大佛夏牛頭犬 (SBT)": "非管制犬種。特徵：顱骨寬且短，吻部短，嘴唇平貼不鬆弛。眼睛中等大小，深色且圓，但有時眼睛顏色與毛色可能相關，眼圈顏色多為深色。鼻子絕對是黑色的",
    "美國惡霸犬 (American Bully)": "非管制犬種。特徵：顱骨寬闊，吻部短且寬呈現輕微立方體，吻部長度較顱骨長度短，吻部約占頭部長度25-35%。高耳位。眼睛中等大小，橢圓型到杏型，不可為藍色。鼻子大且鼻孔寬，鼻子可以是任何顏色。"
}

def generate_gemma_report(image_filename, features, classification_result):
    """
    使用 Gemma 生成包含品種對比的詳細分析報告。
    """
    if not features:
        return f"### 圖片 '{image_filename}' 分析報告 ###\n\n無法生成報告，因為視覺特徵提取失敗。"

    breed = classification_result['breed']
    status = classification_result['status']

    prompt = f"""
    你是一位專業的犬隻品種鑑定報告撰寫員。
    你的任務是根據 VLM 的「特徵分數」和 PNN 的「分類結果」，並嚴格參考「知識庫」，
    為圖片 '{image_filename}' 撰寫一份詳細的鑑定報告。

    ---
    **分析數據:**
    - **圖片名稱:** {image_filename}
    - **最可能品種:** {breed}
    - **管制狀態:** {status}
    - **特徵分數:** {features}

    ---
    **知識庫 (摘錄自農業部比特犬分析指南):**
    - APBT: {PDF_KNOWLEDGE["美國比特鬥牛犬 (APBT)"]}
    - AmStaff: {PDF_KNOWLEDGE["美國史大佛夏牛頭犬 (AmStaff)"]}
    - SBT: {PDF_KNOWLEDGE["史大佛夏牛頭犬 (SBT)"]}
    - American Bully: {PDF_KNOWLEDGE["美國惡霸犬 (American Bully)"]}

    ---
    **報告撰寫要求:**
    1.  以「### 圖片 '{image_filename}' 分析報告 ###」為標題。
    2.  **一、綜合評估**: 明確指出最可能的品種是 '{breed}'，以及它屬於 '{status}'。
    3.  **二、列出特徵分數**: 將vlm給的十項特徵分數列出來。
    4.  **三、鑑定依據**: 詳細解釋為什麼判斷為 '{breed}'。結合特徵分數和知識庫進行說明。例如：「鑑定為 AmStaff 的主要依據是其鼻子為純黑色的可能性極高 (nose_is_black: {features.get('nose_is_black')})，且身材壯碩 (stockiness: {features.get('stockiness')})，這完全符合 AmStaff 的標準。」
    5.  **四、品種比較分析**: 這是最重要的部分。解釋為什麼它**不**是其他三種易混淆的犬種。
        - 如果鑑定為 AmStaff，請解釋為何它不是 APBT（例如：鼻子是黑色而非其他顏色）、SBT（例如：體型較大，身體非正方形）、或惡霸犬（例如：吻部沒有那麼短）。
        - 請根據鑑定結果，靈活撰寫此段落。
    6.  **五、免責聲明**: 加入「本報告僅為基於提供之照片與分析指南的初步AI評估，不具法律效力。最終品種認定應由專業獸醫師或相關權責單位進行。」
    """
    
    try:
        print("正在呼叫 Gemma 生成詳細報告...")
        response = ollama.chat(
            model='gemma3:4b-it-qat',
            messages=[{'role': 'user', 'content': prompt}]
        )
        report = response['message']['content']
        print("Gemma 報告生成完畢。")
        return report
    except Exception as e:
        print(f"呼叫 Gemma 時發生錯誤: {e}")
        return f"生成報告時發生錯誤: {e}"
