# batch_numeric.py 

import os
import vlm_numeric
import pnn_model
import gemma_report
import base64
import re

IMAGE_DIR = 'images'
OUTPUT_DIR = 'out'

# --- 輔助函式 (不變) ---
def image_to_base64(image_path):
    """讀取圖片檔並回傳 Base64 編碼的字串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def text_to_html(text_content):
    """將 Gemma 產出的純文字 (類-Markdown) 轉換為基礎 HTML"""
    # 移除 ### (Gemma 可能還是會自己加)
    html_content = re.sub(r'^\s*###\s*(.*)\s*###\s*$', r'<h3>\1</h3>', text_content, flags=re.MULTILINE)
    # 轉換 h4 標題
    html_content = re.sub(r'^\s*(一、|二、|三、|四、|五、|六、)\s*(.*)\s*$', r'<h4>\1 \2</h4>', html_content, flags=re.MULTILINE)
    # 轉換換行
    html_content = html_content.replace('\n', '<br>\n')
    return f'<div class="report-text">{html_content}</div>'
# --- 輔助函式結束 ---

def process_all_images():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"已建立 '{IMAGE_DIR}' 資料夾。請將待測圖片放入此處。")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_paths_to_process = [] 
    
    for dirpath, dirnames, filenames in os.walk(IMAGE_DIR):
        for filename in filenames:
            if filename.lower().endswith(supported_formats):
                full_path = os.path.join(dirpath, filename)
                image_paths_to_process.append(full_path)

    if not image_paths_to_process:
        print(f"在 '{IMAGE_DIR}' 資料夾及其子資料夾中未找到任何圖片。")
        return

    print(f"找到 {len(image_paths_to_process)} 張圖片，開始處理...")

    for image_path in image_paths_to_process:
        
        filename = os.path.basename(image_path)
        
        print(f"\n{'='*50}")
        print(f"處理中: {image_path}") 
        print(f"{'='*50}")
        
        # 步驟 1: VLM 提取特徵 (不變)
        features = vlm_numeric.get_features_from_vlm(image_path)
        
        if features is None:
            print(f"無法從 {filename} 提取特徵，跳過此圖片。")
            continue
            
        # 步驟 2: PNN 進行分類 (不變)
        frontend_eye_toggle = 1.0     
        frontend_nose_toggle = 1.0    
        frontend_clothes_toggle = 1.0 
        
        classification_result = pnn_model.classify_breed(
            features, 
            eye_toggle=frontend_eye_toggle, 
            nose_toggle=frontend_nose_toggle, 
            clothes_toggle=frontend_clothes_toggle
        )
        print(f"PNN 分類結果: {classification_result}")
        
        # 步驟 3: Gemma 生成報告
        final_report_text_raw = gemma_report.generate_gemma_report(
            image_filename=filename, 
            features=features, 
            classification_result=classification_result, 
            image_path=image_path 
        )
        
        # --- 1. 關鍵修改：在這裡清理 Gemma 的輸出 ---
        # 移除所有 `**` 符號，防止它們破壞 HTML 轉換
        final_report_text = final_report_text_raw.replace("**", "")
        # --- 結束修改 ---
        
        # 步驟 4: 格式化與打包
        # 現在傳入的是清理過的 `final_report_text`
        report_html_body = text_to_html(final_report_text)
        
        try:
            b64_image = image_to_base64(image_path) 
            image_html = f'<img src="data:image/jpeg;base64,{b64_image}" alt="{filename}" style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">'
        except Exception as e:
            print(f"錯誤：無法編碼圖片 {filename}。錯誤：{e}")
            image_html = f"<p>無法載入圖片: {filename}</p>"

        # (報告命名邏輯不變)
        relative_path = os.path.relpath(image_path, IMAGE_DIR)
        base_name_with_subdir = os.path.splitext(relative_path)[0]
        unique_report_name = base_name_with_subdir.replace(os.sep, '_')
        report_filename = f"{unique_report_name}_report.html"
        report_path = os.path.join(OUTPUT_DIR, report_filename)
        
        # (HTML 模板和儲存邏輯不變)
        final_html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-Hant">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>分析報告: {filename}</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }}
                .container {{ max-width: 900px; margin: 20px auto; background: #ffffff; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); overflow: hidden; }}
                .header {{ background-color: #0056b3; color: white; padding: 20px 30px; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .content {{ display: flex; flex-wrap: wrap; padding: 30px; }}
                .image-column {{ flex: 1; min-width: 300px; padding-right: 30px; box-sizing: border-box; }}
                .report-column {{ flex: 1.5; min-width: 400px; box-sizing: border-box; }}
                .report-text h3 {{ color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 5px; margin-top: 0; }}
                .report-text h4 {{ color: #333; margin-top: 20px; margin-bottom: 10px; }}
                .report-text br {{ content: " "; display: block; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>犬隻品種分析報告</h1>
                </div>
                <div class="content">
                    <div class="image-column">
                        <h2>分析照片:</h2>
                        <p>{filename}</p>
                        {image_html}
                    </div>
                    <div class="report-column">
                        {report_html_body}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_html_content)
            
        print(f"HTML 報告已儲存至: {report_path}")

    print(f"\n{'='*50}")
    print("所有圖片處理完畢。")
    print(f"{'='*50}")

if __name__ == '__main__':
    process_all_images()
