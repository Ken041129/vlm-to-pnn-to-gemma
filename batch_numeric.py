# batch_numeric.py (微調)

import os
import vlm_numeric
import pnn_model
import gemma_report

IMAGE_DIR = 'images'
OUTPUT_DIR = 'out'

def process_all_images():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"已建立 '{IMAGE_DIR}' 資料夾。請將待測圖片放入此處。")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"在 '{IMAGE_DIR}' 資料夾中未找到任何圖片。")
        return

    print(f"找到 {len(image_files)} 張圖片，開始處理...")

    for filename in image_files:
        print(f"\n{'='*50}")
        print(f"處理中: {filename}")
        print(f"{'='*50}")
        
        image_path = os.path.join(IMAGE_DIR, filename)
        
        # 步驟 1: VLM 提取特徵 (使用升級版)
        features = vlm_numeric.get_features_from_vlm(image_path)
        
        if features is None:
            print(f"無法從 {filename} 提取特徵，跳過此圖片。")
            continue
            
        # 步驟 2: PNN 進行分類 (使用升級版)
        classification_result = pnn_model.classify_breed(features)
        print(f"PNN 分類結果: {classification_result}")
        
        # 步驟 3: Gemma 生成報告 (使用升級版)
        final_report = gemma_report.generate_gemma_report(filename, features, classification_result)
        
        # 步驟 4: 儲存報告
        report_filename = f"{os.path.splitext(filename)[0]}_report.txt"
        report_path = os.path.join(OUTPUT_DIR, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
            
        print(f"報告已儲存至: {report_path}")

    print(f"\n{'='*50}\n所有圖片處理完畢。\n{'='*50}")

if __name__ == '__main__':
    process_all_images()
