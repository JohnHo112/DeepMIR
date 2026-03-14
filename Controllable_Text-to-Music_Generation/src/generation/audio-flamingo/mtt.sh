#!/bin/bash
rm caption.json

# 指定要讀取的資料夾
FOLDER="../../../home/fundwotsai/Deep_MIR_hw2/target_music_list_60s"

# 逐一讀取資料夾中的檔案
for file in "$FOLDER"/*; do
    # 取得不含路徑的檔名
    filename=$(basename "$file")

    echo "執行中：$FOLDER/$filename"
    # 執行你的 Python 指令
    python llava/cli/infer_audio.py --model-base nvidia/audio-flamingo-3 --conv-mode auto --text "Please describe the audio in detail and it's bpm" --media "$FOLDER/$filename" --think-mode
    # python your_script.py --name "$filename"
done
