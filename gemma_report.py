# gemma_report.py

import ollama
import base64
import os
# --- 1. 新增：從 pnn_model 導入 IDEAL_VECTORS ---
# 這樣我們就可以在 Prompt 中使用 PNN 的理想值
try:
    from pnn_model import IDEAL_VECTORS
except ImportError:
    print("錯誤：無法從 pnn_model.py 導入 IDEAL_VECTORS。")
    # 如果導入失敗，使用一個備用的（這不應該發生，但作為安全措施）
    IDEAL_VECTORS = {"美國比特鬥牛犬 (APBT)": {"MuzzleHeadRatio": 0.70}}
# --- 結束新增 ---


# --- 2. 圖片轉 Base64 輔助函式  ---
def image_to_base64(image_path):
    """讀取圖片檔並回傳 Base64 編碼的字串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- 3. 「二次鑑定」函式  ---
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

    [目標犬種]
    - 美國比特鬥牛犬 (APBT)
    - 美國史大佛夏牛頭犬 (AmStaff)
    - 史大佛夏牛頭犬 (SBT)
    - 美國惡霸犬 (American Bully)
    - 其他犬種

    [重要規則]
    如果圖片中的犬隻明顯是 拳師犬(Boxer), 法國鬥牛犬(French Bulldog), 
    英國鬥牛犬(Bulldog), 阿根廷杜告犬 (Dogo Argentino), 土佐犬 (Tosa), 
    紐波利頓犬(Neapolitan Mastiff), 波士頓㹴(Boston Terrier),
    黃金獵犬(Golden Retriever), 或任何其他非[目標犬種]列表中的狗，
    請一律歸類為「其他犬種」。

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
        judgment = response['message']['content'].strip().replace("\"", "")
        print(f"VLM 初步意見為: {judgment}")
        return judgment
    except Exception as e:
        print(f"獲取 VLM 初步意見時發生錯誤: {e}")
        return "鑑定失敗"

# 知識庫 (您的版本)
PDF_KNOWLEDGE = {
    "美國比特鬥牛犬 (APBT)": "管制犬種。特徵：頭呈楔形（長三角柱型），兩耳之間顱骨寬平或略圓，吻部與頭顱長度比例約為2:3。高耳位。眼睛可為除了藍色以外的所有顏色，中等大小，圓形。鼻子大且鼻孔寬，鼻子可以是任何顏色。胸腔寬度不超過其深度",
    "美國史大佛夏牛頭犬 (AmStaff)": "管制犬種。特徵：顱骨寬，吻部背側呈圓弧狀，嘴唇平貼不鬆弛。高耳位的短小玫瑰耳或半折耳，少許耳尖向外側或向前折，可看見耳道。眼睛深色且圓。鼻子絕對是黑色的。短而濃密的毛",
    "史大佛夏牛頭犬 (SBT)": "非管制犬種。特徵：顱骨寬且短，吻部短，嘴唇平貼不鬆弛。眼圈顏色多為深色。鼻子絕對是黑色的",
    "美國惡霸犬 (American Bully)": "非管制犬種。特徵：顱骨寬闊，吻部短且寬呈現輕微立方體，吻部長度較顱骨長度短，吻部約占頭部長度25-35%。高耳位。眼睛不可為藍色。鼻子大且鼻孔寬，鼻子可以是任何顏色。"
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

# --- 4. 新增：將 IDEAL_VECTORS 轉換為字串的輔助函式 ---
def format_vectors_for_prompt():
    """將 PNN 的理想值轉換為給 VLM 看的字串"""
    lines = []
    for breed, features in IDEAL_VECTORS.items():
        # 我們只挑選幾個最關鍵的特徵放入 Prompt，避免太長
        line = (
            f"- {breed}:\n"
            f"  - MuzzleHeadRatio (理想值): {features['MuzzleHeadRatio']:.2f}\n"
            f"  - HeadBreadthIndex (理想值): {features['HeadBreadthIndex']:.2f}\n"
            f"  - BodySquareness (理想值): {features['BodySquareness']:.2f}\n"
            f"  - BlackNoseRequired (理想值): {features['BlackNoseRequired']:.2f}"
        )
        lines.append(line)
    return "\n".join(lines)
# --- 結束新增 ---

# --- 主函式：已升級 VLM 否決權優先邏輯 ---
def generate_gemma_report(image_filename, features, classification_result, image_path):
    """
    使用 Gemma 生成包含「PNN計算」與「VLM初判」對比的詳細分析報告。
    """
    
    prelim_judgment = get_preliminary_judgment(image_path)
    
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
            
    prompt = "" 

    try:
        b64_image_for_report = image_to_base64(image_path)
        image_list_for_report = [b64_image_for_report]
    except Exception as e:
        print(f"錯誤：無法編碼圖片 {image_path} 以用於最終報告: {e}")
        image_list_for_report = [] 
    
    if not is_target_breed:
        #
        # --- 情況 A：VLM 判斷為「其他犬種」(例如 "黃金獵犬") ---
        # (此區塊不變)
        #
        print(f"VLM 否決！ 初步意見 '{prelim_judgment}' 非目標犬種。將使用簡化報告。")
        
        final_breed = "其他犬種"
        final_status = "非管制犬種"
        
        prompt = f"""
        你是一位專業的犬隻品種鑑定報告撰寫員。
        **你現在正在查看一張照片。**

        ---
        **分析數據:**
        - **圖片名稱:** {image_filename}
        - **VLM 初步專家意見:** {prelim_judgment}
        - **最終鑑定結論:** {final_breed}
        - **最終管制狀態:** {final_status}

        ---
        **報告撰寫要求:**
        1.  以「### 圖片 '{image_filename}' 分析報告 ###」作為標題。
        2.  **一、綜合評估**: 
            - 明確指出最終鑑定結論為「{final_breed}」，屬於「{final_status}」。
            - 說明理由：VLM 的初步專家意見判定此犬隻為「{prelim_judgment}」，此非農業部定義的四種比特型犬種。
        3.  **二、鑑定依據**: 
            - **請看著你眼前的照片**，簡要描述此犬隻的**實際外觀**（例如吻部長度、毛髮、體型等）**如何**與比特型犬種的特徵（短吻、寬頭、短毛、壯碩身軀）不符。
        4.  **三、免責聲明**: 加入「本報告僅為基於提供之照片與分析指南的初步AI評估，不具法律效力。最終品種認定應由專業獸醫師或相關權責單位進行。」
        """
    else:
        #
        # --- 情況 B：VLM 判斷為「四種比特犬之一」 ---
        #
        
        pnn_breed = classification_result['breed']
        final_breed = pnn_breed 
        final_status = classification_result['status']
        judgment_source_note = ""
        features_list_str = format_features_for_report(features) 
        
        if final_breed == "其他犬種":
            #
            # --- 子情況 B1: VLM 說 "APBT"，但 PNN 否決 (距離太遠) ---
            # (此區塊不變)
            #
            judgment_source_note = f"**系統判斷衝突**：VLM 初步專家意見為「{prelim_judgment}」，但 PNN 模組基於特徵分數的嚴格計算判定其**不符合**任何已知標準（距離超過閾值）。本報告將以 PNN 的計算結果為最終結論。"

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
            4.  **三、鑑定依據**: 說明此犬隻被判定為「其他犬種」的理由（PNN 計算距離超過 {classification_result.get('threshold', 'N/A')} 的閾值）。
            5.  **四、免責聲明**: 加入「本報告僅為基於提供之照片與分析指南的初步AI評G估，不具法律效力。最終品種認定應由專業獸醫師或相關權責單位進行。」
            """
        else:
            #
            # --- 子情況 B2: VLM 和 PNN 都認為是「四種比特犬之一」 ---
            #
            consistency_note = ""
            if final_breed.lower().split(" ")[-1].strip("()") in prelim_judgment.lower():
                 consistency_note = f"系統判斷一致：VLM 初步專家意見 ({prelim_judgment}) 與 PNN 嚴格計算結果 ({final_breed}) 均指向同一犬種。"
            else:
                 consistency_note = f"**系統判斷衝突**：VLM 初步專家意見為「{prelim_judgment}」，但 PNN 模組的嚴格特徵計算結果為「{final_breed}」。本報告將以 PNN 的計算結果為最終結論。"

            # --- *** 這裡是您要求的修改 *** ---
            
            # 獲取 PNN 理想值小抄
            pnn_vectors_str = format_vectors_for_prompt()
            
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
            **知識庫 1 (PNN 理想值小抄):**
            (這是在 pnn_model.py 中定義的理想分數)
            {pnn_vectors_str}
            
            **知識庫 2 (PDF 文字描述):**
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
                - **舉例說明**：例如，如果 PNN 判斷為 AmStaff，請說明「如照片所示，此犬隻的吻部確實較短、頭骨寬闊... 這與 PNN 計算出的低 MuzzleHeadRatio 分數 ({features.get('MuzzleHeadRatio', 'N/A')}) 和高 HeadBreadthIndex 分數 ({features.get('HeadBreadthIndex', 'N/A')}) 一致。」
            5.  **四、品種比較分析**: 
                - **[新指令]** **請繼續看著照片**，將此犬隻（{final_breed}）與其他三種易混淆的犬種進行**客觀的特徵比對**。
                - **[新指令]** 你的目標是**找出關鍵的差異點**，以說明為何 PNN 判定它為「{final_breed}」。
                - **[強制規則]** 你**必須**使用「二、VLM 特徵分數」中的**具體數值**，並將其與「知識庫 1 (PNN 理想值小抄)」中的**理想值**進行**正確的數學比較**。
                - **[新指令]** 如果某個特徵**相符**（例如鼻子都是黑色），**請先誠實地指出這一點**，然後**再強調其他不相符的特徵**。
                - **[範例]**：「...與 APBT 進行比較：此犬隻的 MuzzleHeadRatio 分數為 {features.get('MuzzleHeadRatio', 'N/A')}，這**遠低於** APBT 的理想值 (0.70)，這是一個關鍵差異點。」
            6.  **五、免責聲明**: 加入「本報告僅為基於提供之照片與分析指南的初步AI評估，不具法律效力。最終品種認定應由專業獸醫師或相關權責單位進行。」
            """
            # --- *** 修改結束 *** ---
    # --- 結束邏輯 ---
    
    try:
        print("正在呼叫 Gemma (VLM 報告模式) 生成最終報告...")
        response = ollama.chat(
            model='gemma3:27b-it-qat', # 使用 LLM/VLM 模型
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': image_list_for_report 
                }
            ]
        )
        
        report = response['message']['content']
        print("Gemma 報告生成完畢。")
        # --- (回傳值) ---
        return report # <-- 您的 batch_numeric.py 預期一個回傳值
    except Exception as e:
        error_msg = f"呼叫 Gemma (VLM 報告模式) 時發生錯誤: {e}"
        print(error_msg)
        # --- (回傳值) ---
        return error_msg # <-- 您的 batch_numeric.py 預期一個回傳值
