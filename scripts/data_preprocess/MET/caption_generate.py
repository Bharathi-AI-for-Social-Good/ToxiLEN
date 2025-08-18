import pandas as pd
import os
from torch import nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

device = 'cuda'
model_name = 'Salesforce/blip2-opt-2.7b'

import json

with open('data/MET/converted_data.json',"r",encoding='utf-8') as f:
    data = json.load(f)

train_df = pd.DataFrame(data)
train_df.head()

file_names = train_df['filename'].to_list()

# 进度保存文件路径
progress_file = 'data/MET/caption_progress.json'
output_file = 'data/MET/train_caption_version.csv'

processor =  AutoProcessor.from_pretrained(model_name)
model = AutoModelForVisualQuestionAnswering.from_pretrained(model_name).to(device)

# 加载之前的进度
def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"completed_indices": [], "captions": []}

# 保存进度
def save_progress(completed_indices, captions):
    progress_data = {
        "completed_indices": completed_indices,
        "captions": captions
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

# 保存最终结果
def save_final_result(train_df, captions):
    train_df['captions'] = captions
    train_df.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")

def generate_caption(image_path):
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image,return_tensors='pt').to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"生成字幕时出错 {image_path}: {e}")
        return "Error generating caption"


# 加载之前的进度
progress_data = load_progress()
completed_indices = set(progress_data["completed_indices"])
caption_results = progress_data["captions"].copy()

# 确保caption_results长度与file_names一致
if len(caption_results) < len(file_names):
    caption_results.extend([None] * (len(file_names) - len(caption_results)))

print(f"总共需要处理 {len(file_names)} 个图片")
print(f"已完成 {len(completed_indices)} 个图片")
print(f"剩余 {len(file_names) - len(completed_indices)} 个图片")

try:
    for i, image_path in enumerate(file_names):
        # 跳过已完成的
        if i in completed_indices:
            continue
            
        image = rf"data/MET/cn_images/Image- {image_path[6: ]}"
        
        print(f"处理中 [{i+1}/{len(file_names)}]: {image_path}")
        caption = generate_caption(image_path=image)   
        
        caption_results[i] = caption
        completed_indices.add(i)
        
        print(f"完成: {caption}")
        
        # 每10个图片保存一次进度
        if (i + 1) % 10 == 0:
            save_progress(list(completed_indices), caption_results)
            print(f"进度已保存 ({len(completed_indices)}/{len(file_names)})")
    
    # 保存最终结果
    save_final_result(train_df, caption_results)
    
    # 删除进度文件
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print("进度文件已删除")
        
except KeyboardInterrupt:
    print("\n用户中断，正在保存进度...")
    save_progress(list(completed_indices), caption_results)
    print(f"进度已保存，完成了 {len(completed_indices)}/{len(file_names)} 个图片")
    print("下次运行脚本将从中断处继续")
except Exception as e:
    print(f"发生错误: {e}")
    save_progress(list(completed_indices), caption_results)
    print("进度已保存")