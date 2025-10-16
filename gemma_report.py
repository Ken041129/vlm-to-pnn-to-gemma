# gemma_report.py 

import ollama

# 擴充知識庫，包含所有四種犬種的關鍵特徵
PDF_KNOWLEDGE = {
    "美國比特鬥牛犬 (APBT)": "管制犬種。特徵：頭呈楔形，身長略大於肩高，吻部與頭顱比例約2:3，鼻子顏色不拘，體態精實。",
    "美國史大佛夏牛頭犬 (AmStaff)": "管制犬種。特徵：顱骨寬，身長略大於肩高，比 APBT 更壯碩，鼻子『絕對是黑色』。",
    "史大佛夏牛頭犬 (SBT)": "非管制犬種。特徵：體型明顯較小，身長與肩高相等（呈正方形），吻部短，鼻子『絕對是黑色』。",
    "美國惡霸犬 (American Bully)": "非管制犬種。特徵：頭顱寬闊，吻部『極短』且寬（呈立方體），身材非常壯碩矮 chunky，鼻子顏色不拘。"
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
    3.  **二、鑑定依據**: 詳細解釋為什麼判斷為 '{breed}'。結合特徵分數和知識庫進行說明。例如：「鑑定為 AmStaff 的主要依據是其鼻子為純黑色的可能性極高 (nose_is_black: {features.get('nose_is_black')})，且身材壯碩 (stockiness: {features.get('stockiness')})，這完全符合 AmStaff 的標準。」
    4.  **三、品種比較分析**: 這是最重要的部分。解釋為什麼它**不**是其他三種易混淆的犬種。
        - 如果鑑定為 AmStaff，請解釋為何它不是 APBT（例如：鼻子是黑色而非其他顏色）、SBT（例如：體型較大，身體非正方形）、或惡霸犬（例如：吻部沒有那麼短）。
        - 請根據鑑定結果，靈活撰寫此段落。
    5.  **四、免責聲明**: 加入「本報告僅為基於提供之照片與分析指南的初步AI評估，不具法律效力。最終品種認定應由專業獸醫師或相關權責單位進行。」
    """
    
    try:
        print("正在呼叫 Gemma 生成詳細報告...")
        response = ollama.chat(
            model='gemma:4b-it-qat',
            messages=[{'role': 'user', 'content': prompt}]
        )
        report = response['message']['content']
        print("Gemma 報告生成完畢。")
        return report
    except Exception as e:
        print(f"呼叫 Gemma 時發生錯誤: {e}")
        return f"生成報告時發生錯誤: {e}"
