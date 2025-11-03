# gemma_report.py 

import ollama
import base64
import os

# --- 1. 圖片轉 Base64 輔助函式 (不變) ---
def image_to_base64(image_path):
    """讀取圖片檔並回傳 Base64 編碼的字串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- 2. 「二次鑑定」函式 (不變) ---
def get_preliminary_judgment(image_path):
    """
    第二次呼叫 VLM，只為了獲取它的「初步專家意見」。
    """
    if not os.path.exists(image_path):
        return "無法讀取圖片"

    b64_image = image_to_base64(image_path)
    
    prompt = """
    你是一位頂尖的犬隻品種鑑定專家。
    請只看這張圖片，憑你的第一直覺，判斷這隻狗最接近以下哪個分類？
    - 美國比特鬥牛犬 (APBT)
    - 美國史大佛夏牛頭犬 (AmStaff)
    - 史大佛夏牛頭犬 (SBT)
    - 美國惡霸犬 (American Bully)
    - 其他犬種

    請只回傳你選擇的「一個」分類名稱，不要有任何其他文字或解釋。
    """
    
    try:
        print("正在呼叫 VLM 進行「初步專家意見」鑑定...")
        response = ollama.chat(
            model='gemma3:27b-it-qat', # 使用 VLM 模型
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [b64_image]
                }
            ]
        )
        # 清理 VLM 可能回傳的引號
        judgment = response['message']['content'].strip().replace("\"", "")
        print(f"VLM 初步意見為: {judgment}")
        return judgment
    except Exception as e:
        print(f"獲取 VLM 初步意見時發生錯誤: {e}")
        return "鑑定失敗"

# 知識庫 (不變)
PDF_KNOWLEDGE = {
    "美國比特鬥牛犬 (APBT)": "管制犬種。特徵：頭呈楔形（長三角柱型），兩耳之間顱骨寬平或略圓，吻部與頭顱長度比例約為2:3。高耳位。眼睛可為除了藍色以外的所有顏色，中等大小，圓形。鼻子大且鼻孔寬，鼻子可以是任何顏色。胸腔寬度不超過其深度",
    "美國史大佛夏牛頭犬 (AmStaff)": "管制犬種。特徵：顱骨寬，吻部背側呈圓弧狀，嘴唇平貼不鬆弛。高耳位的短小玫瑰耳或半折耳，少許耳尖向外側或向前折，可看見耳道。眼睛深色且圓。鼻子絕對是黑色的。短而濃密的毛",
    "史大佛夏牛頭犬 (SBT)": "非管制犬種。特徵：顱骨寬且短，吻部短，嘴唇平貼不鬆弛。眼睛中等大小，深色且圓，但有時眼睛顏色與毛色可能相關，眼圈顏色多為深色。鼻子絕對是黑色的",
    "美國惡霸犬 (American Bully)": "非管制犬種。特徵：顱骨寬闊，吻部短且寬呈現輕微立方體，吻部長度較顱骨長度短，吻部約占頭部長度25-35%。高耳位。眼睛中等大小，橢圓型到杏型，不可為藍色。鼻子大且鼻孔寬，鼻子可以是任何顏色。"
}

def format_features_for_report(features):
    """將特徵字典轉換為易讀的條列式字串"""
    report_lines = []
    for key, value in features.items():
        try:
            formatted_value = f"{float(value):.2f}"
        except (ValueError, TypeError):
            formatted_value = str(value)
        report_lines.append(f"- {key}: {formatted_value}")
    return "\n".join(report_lines)

# --- 3. 修改主函式：加入 VLM 否決邏輯 ---
def generate_gemma_report(image_filename, features, classification_result, image_path):
    """
    使用 Gemma 生成包含「PNN計算」與「VLM初判」對比的詳細分析報告。
    """
    if not features:
        return f"### 圖片 '{image_filename}' 分析報告 ###\n\n無法生成報告，因為視覺特徵提取失敗。"

    # PNN 的結果 (可能被愚弄)
    pnn_breed = classification_result['breed']
    
    # VLM 的初步意見 (可靠的守門員)
    prelim_judgment = get_preliminary_judgment(image_path)
    
    # 目標犬種清單 (用於檢查 VLM 的意見)
    target_breeds_check = [
        "american pit bull terrier", "apbt",
        "american staffordshire terrier", "amstaff",
        "staffordshire bull terrier", "sbt",
        "american bully", "bully"
    ]
    
    is_target_breed = False
    for target in target_breeds_check:
        if target in prelim_judgment.lower():
            is_target_breed = True
            break
            
    final_breed = pnn_breed # 最終結論先採用 PNN 的計算結果
    final_status = classification_result['status']
    judgment_source_note = "" # 用於報告中的額外說明

    if not is_target_breed:
        # VLM 一票否決 (例如 VLM 說是 "黃金獵犬")
        print(f"VLM 否決！ 初步意見 '{prelim_judgment}' 非目標犬種。")
        final_breed = "其他犬種" 
        final_status = "非管制犬種"
        judgment_source_note = f"註：VLM 初步專家意見 ({prelim_judgment}) 否決了 PNN 模組的計算結果 ({pnn_breed})，因為該犬隻明顯不屬於比特型犬種。"
    
    elif prelim_judgment.lower() != "其他犬種" and pnn_breed == "其他犬種":
        # PNN 一票否決 (例如 VLM 說是 "APBT"，但 PNN 距離太遠)
        print(f"PNN 否決！ VLM 初步意見為 '{prelim_judgment}'，但 PNN 計算距離過遠。")
        final_breed = "其他犬種" 
        final_status = "非管制犬種"
        judgment_source_note = f"註：VLM 初步專家意見為 '{prelim_judgment}'，但 PNN 模組基於特徵分數的嚴格計算判定其不符合標準（距離超過閾值）。"
    
    
    features_list_str = format_features_for_report(features)
    prompt = ""

    # --- 4. 關鍵修改：為最終報告的 Prompt 準備圖片 ---
    try:
        b64_image_for_report = image_to_base64(image_path)
        image_list_for_report = [b64_image_for_report]
    except Exception as e:
        print(f"錯誤：無法編碼圖片 {image_path} 以用於最終報告: {e}")
        image_list_for_report = [] # 報告將在沒有圖片的情況下生成
    # --- 結束修改 ---

    
    if final_breed == "其他犬種":
        # --- 「其他犬種」的報告模板 (升級) ---
        prompt = f"""
        你是一位專業的犬隻品種鑑定報告撰寫員。
        **你現在正在查看一張照片。**

        ---
        **分析數據:**
        - **圖片名稱:** {image_filename}
        - **VLM 初步專家意見:** {prelim_judgment}
        - **PNN 計算結果:** {pnn_breed} (基於特徵分數與閾值判斷)
        - **最終鑑定結論:** {final_breed}
        - **最終管制狀態:** {final_status}

        ---
        **報告撰寫要求:**
        1.  以「### 圖片 '{image_filename}' 分析報告 ###」作為標題。
        2.  **一、綜合評估**: 
            - 明確指出最終鑑定結論為「{final_breed}」，屬於「{final_status}」。
            - {judgment_source_note}
        3.  **二、VLM 特徵分數**: 列出 VLM 模組給出的 8 項原始分數：
            {features_list_str}
        4.  **三、鑑定依據**: 
            - **請看著你眼前的照片**，簡要描述此犬隻的**實際外觀**（例如吻部長度、毛髮、體型等）**如何**與比特型犬種的特徵（短吻、寬頭、短毛、壯碩身軀）不符，從而印證了 VLM 的初步判斷。
        5.  **四、免責聲明**: 加入「本報告僅為基於提供之照片與分析指南的初步AI評估，不具法律效力。最終品種認定應由專業獸醫師或相關權責單位進行。」
        """
    else:
        # --- 「四種比特犬」的報告模板 (升級) ---
        
        # 建立「一致性」的說明
        consistency_note = ""
        if prelim_judgment == pnn_breed:
            consistency_note = f"系統判斷一致：VLM 初步專家意見與 PNN 嚴格計算結果均為「{final_breed}」。"
        else:
            # 處理 VLM 說 APBT 而 PNN 說 AmStaff 的情況
            is_prelim_target = False
            for target in target_breeds_check:
                if target in prelim_judgment.lower():
                    is_prelim_target = True
                    break
            
            if is_prelim_target: # VLM 意見也是四種之一
                consistency_note = f"**系統判斷衝突**：VLM 初步專家意見為「{prelim_judgment}」，但 PNN 模組的嚴格特徵計算結果為「{final_breed}」。本報告將以 PNN 的計算結果為最終結論。"
            else: # VLM 意見是 "其他犬種" (但 PNN 覆寫了它) - 這種情況理論上已被 "if final_breed == '其他犬種'" 捕捉，但作為備援
                consistency_note = f"**系統判斷衝突**：VLM 初步專家意見為「{prelim_judgment}」，但 PNN 模組的嚴格特徵計算結果為「{final_breed}」。本報告將以 PNN 的計算結果為最終結論。"


        prompt = f"""
        你是一位專業的犬隻品種鑑定報告撰寫員。
        **你現在正在查看一張照片**，同時你手上有 VLM 模組的「初步意見」和 PNN 模組的「客觀計算」結果。

        ---
        **分析數據:**
        - **圖片名稱:** {image_filename}
        - **VLM 初步專家意見:** {prelim_judgment}
        - **PNN 計算結果:** {pnn_breed}
        - **最終鑑定結論:** {final_breed}
        - **最終管制狀態:** {final_status}

        ---
        **知識庫 (摘錄自農業部比特犬分析指南):**
        - APBT: {PDF_KNOWLEDGE["美國比特鬥牛犬 (APBT)"]}
        - AmStaff: {PDF_KNOWLEDGE["美國史大佛夏牛頭犬 (AmStaff)"]}
        - SBT: {PDF_KNOWLEDGE["史大佛夏牛頭犬 (SBT)"]}
        - American Bully: {PDF_KNOWLEDGE["美國惡霸犬 (American Bully)"]}

        ---
        **報告撰寫要求:**
        1.  以「### 圖片 '{image_filename}' 分析報告 ###」作為標題。
        2.  **一、綜合評估**: 
            - 明確指出最終鑑定結論為：「{final_breed}」，屬於「{final_status}」。
            - {consistency_note}
        3.  **二、VLM 特徵分數**: 列出 VLM 模組給出的 8 項原始分數：
            {features_list_str}
        4.  **三、PNN 鑑定依據**: 
            - **請看著你眼前的照片**，將它的**實際外觀**與 PNN 的計算結果（{final_breed}）和特徵分數進行交叉比對。
            - **舉例說明**：例如，如果 PNN 判斷為 AmStaff，請說明「如照片所示，此犬隻的吻部確實較短、頭骨寬闊... 這與 PNN 計算出的低 MuzzleHeadRatio 分數 (xxx) 和高 HeadBreadthIndex 分數 (xxx) 一致。」
        5.  **四、品種比較分析**: 
            - **請繼續看著照片**，解釋為什麼它**不**是其他三種易混淆的犬種，並將你的視覺觀察與特徵分數連結起來。
        6.  **五、免責聲明**: 加入「本報告僅為基於提供之照片與分析指南的初步AI評估，不具法律效力。最終品種認定應由專業獸醫師或相關權責單位進行。」
        """
    
    try:
        print("正在呼叫 Gemma (VLM 報告模式) 生成最終報告...")
        # --- 5. 關鍵修改：在呼叫中傳入圖片 ---
        response = ollama.chat(
            model='gemma3:27b-it-qat', # 使用 LLM/VLM 模型
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': image_list_for_report # <-- 在此傳入圖片
                }
            ]
        )
        # --- 結束修改 ---
        
        report = response['message']['content']
        print("Gemma 報告生成完畢。")
        return report
    except Exception as e:
        print(f"呼叫 Gemma (VLM 報告模式) 時發生錯誤: {e}")
        return f"生成報告時發生錯誤: {e}"
