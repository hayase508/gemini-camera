#ライブラリ
import cv2
from ultralytics import YOLO
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from gtts import gTTS
import pygame
import time
import threading
from PIL import Image
import io
import os

#Gemini APIキー
API_KEY = 'YOUR_API_KEY'

# 動作設定
COOLDOWN_SECONDS = 20       # 音声再生後の待機時間（秒）
CONFIDENCE_THRESHOLD = 0.5  # YOLOの人認識感度

#画質設定
AI_IMAGE_WIDTH = 320       

# Geminiへの命令
SYSTEM_PROMPT = """
あなたはカフェの前のフレンドリーな客引きロボットです。
送られた画像に写っている通行人（1人または複数人）の特徴を見て、短く声をかけてください。

【ルール】
1. とても明るく、親しみやすい口調で話してください。
2. 相手の人数に合わせて自然に呼びかけてください（例：「そこのお兄さん」「お二人組の方」「皆さん」など）。
3. 35文字以内の短い一言にしてください。
4. 最後に必ず「コーヒー飲んでいきませんか？」と繋げてください。

【注目ポイント（ここから1つ選んで褒めるか言及する）】
* 服装の色や柄（「赤い服が素敵ですね」など）
* 持ち物（「大きな荷物ですね」「素敵な帽子ですね」など）
* 雰囲気や行動（「楽しそうですね」「お散歩中ですか？」など）

【禁止事項】
* 身体的特徴（太っている、痩せている、顔の造作など）には絶対に触れないでください。
* 年齢や性別を決めつけるような強い表現（おじさん、おばさん等）は避けてください。
* ネガティブな表現（疲れてそう、暗そう）は禁止。
"""

# 初期化処理

# Gemini設定
genai.configure(api_key=API_KEY)

print("Geminiモデル(gemini-2.5-flash-lite)を準備中...")
try:
    gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
except Exception as e:
    print(f"モデル設定エラー: {e}")
    exit()

# 安全設定
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# YOLO設定
print("YOLOモデルを読み込んでいます...")
os.environ['YOLO_VERBOSE'] = 'False'
yolo_model = YOLO('yolov8n.pt')

# 音声再生 (Pygame)
pygame.mixer.init(frequency=24000, buffer=1024)

# 状態管理
last_process_time = 0
is_processing = False

#画像分析と音声生成関数
def ai_process_and_speak(frame_rgb, person_count):
    
    global is_processing, last_process_time

    try:
        print(f">>> {person_count}人を検知。Geminiに画像を送信中...")

        pil_image = Image.fromarray(frame_rgb)

        # リサイズ
        if pil_image.width > AI_IMAGE_WIDTH:
            aspect_ratio = pil_image.height / pil_image.width
            new_height = int(AI_IMAGE_WIDTH * aspect_ratio)
            pil_image = pil_image.resize((AI_IMAGE_WIDTH, new_height))
            print(f"   (画像を {AI_IMAGE_WIDTH}x{new_height} に縮小しました)")

        current_prompt = SYSTEM_PROMPT + f"\n\n【現在の状況】\n画像には{person_count}人の人が写っています。"

        # 計測開始
        start_time = time.time()

        # Gemini呼び出し
        response = gemini_model.generate_content(
            [current_prompt, pil_image],
            safety_settings=safety_settings
        )
        
        # 計測終了
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Gemini思考時間: {elapsed_time:.2f}秒")

        # テキスト取得
        try:
            generated_text = response.text.replace("\n", "").strip()
        except ValueError:
            print("Geminiが回答を生成できませんでした。")
            return

        print(f"セリフ: 「{generated_text}」")

        if not generated_text:
            return

        # 音声合成 (gTTS)
        mp3_fp = io.BytesIO()
        tts = gTTS(text=generated_text, lang='ja')
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # 再生
        pygame.mixer.music.load(mp3_fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        mp3_fp.close()

    except Exception as e:
        print(f"エラーが発生しました: {e}")

    finally:
        last_process_time = time.time()
        is_processing = False
        print(f">>> 待機モードに戻ります（クールダウン: {COOLDOWN_SECONDS}秒）")


def main():
    global is_processing, last_process_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが見つかりません。")
        return

    print("監視開始: 'q'キーで終了")
    last_process_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)
        frame_visual = frame.copy()
        person_count = 0

        for result in results:
            boxes = result.boxes
            person_count = len(boxes)
            frame_visual = result.plot()

        current_time = time.time()
        
        if (person_count > 0 and 
            not is_processing and 
            (current_time - last_process_time > COOLDOWN_SECONDS)):
            
            is_processing = True
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            threading.Thread(
                target=ai_process_and_speak, 
                args=(frame_rgb, person_count), 
                daemon=True
            ).start()

        # ステータス表示
        if is_processing:
            status_text = "AI Thinking..."
            status_color = (0, 0, 255)
        else:
            remain = max(0, COOLDOWN_SECONDS - (current_time - last_process_time))
            if remain > 0:
                status_text = f"Cooldown: {remain:.1f}s"
                status_color = (0, 165, 255)
            else:
                status_text = "Ready"
                status_color = (0, 255, 0)

        cv2.putText(frame_visual, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame_visual, f"Count: {person_count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Gemini AI Camera', frame_visual)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()
