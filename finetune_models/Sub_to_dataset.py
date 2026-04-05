import os
import re
import shutil
import subprocess
import random
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- НАСТРОЙКИ ---
BASE_DIR = Path(__file__).parent.absolute()
FFMPEG_PATH = BASE_DIR / "ffmpeg.exe" 
WORK_DIR = BASE_DIR / 'work'
DATASET_DIR = BASE_DIR / 'dataset'
WAVS_DIR = DATASET_DIR / 'wavs'
TRAIN_CSV = DATASET_DIR / 'metadata_train.csv'
EVAL_CSV = DATASET_DIR / 'metadata_eval.csv'
LANG_FILE = DATASET_DIR / 'lang.txt'
DIALOGS_FILE = BASE_DIR / 'Dialogs.txt'

AUDIO_EXTENSIONS = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
SRT_PROG = re.compile(r"([0-9:\.]+).+?([0-9:\.]+)")

def check_env():
    if not FFMPEG_PATH.exists():
        print(f"!!! ОШИБКА: {FFMPEG_PATH.name} не найден в {BASE_DIR}")
        sys.exit(1)
    if not WORK_DIR.exists():
        print(f"!!! ОШИБКА: Папка '{WORK_DIR.name}' не найдена.")
        sys.exit(1)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if not LANG_FILE.exists():
        LANG_FILE.write_text("en", encoding='utf-8')

def find_audio_source(srt_path):
    for ext in AUDIO_EXTENSIONS:
        audio_path = srt_path.with_suffix(ext)
        if audio_path.exists():
            return audio_path
    return None

def process_single_srt(srt_path):
    in_file = find_audio_source(srt_path)
    if not in_file:
        return srt_path.stem, None

    target_folder = srt_path.with_suffix('')
    target_folder.mkdir(parents=True, exist_ok=True)
    folder_name = target_folder.name
    local_map = {}

    with open(srt_path, "r", encoding='utf-8-sig') as f:
        content = f.read().strip().split('\n')

    i = 0
    while i < len(content):
        line = content[i].strip()
        if line.isdigit():
            idx = line.rjust(3, "0")
            if i + 2 < len(content):
                time_line = content[i+1].strip().replace(",", ".")
                text = content[i+2].strip()
                sm_match = SRT_PROG.match(time_line)
                
                if sm_match and text:
                    ss, to = sm_match.group(1), sm_match.group(2)
                    entry_name = f"{folder_name}_{idx}"
                    out_path = target_folder / f"{entry_name}.wav"
                    
                    cmd = [str(FFMPEG_PATH), "-loglevel", "quiet", "-y", "-i", str(in_file),
                           "-ss", ss, "-to", to, "-f", "wav", str(out_path)]
                    subprocess.run(cmd)
                    local_map[entry_name] = text
                i += 2
        i += 1
    
    for wav in target_folder.glob('*.wav'):
        shutil.move(str(wav), str(WAVS_DIR / wav.name))
    shutil.rmtree(target_folder)
    return srt_path.stem, local_map

def main():
    check_env()
    if WAVS_DIR.exists(): shutil.rmtree(WAVS_DIR)
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    if DIALOGS_FILE.exists(): os.remove(DIALOGS_FILE)

    srt_files = list(WORK_DIR.glob('*.srt'))
    if not srt_files:
        print("Нет .srt файлов в 'work'.")
        return

    total_files = len(srt_files)
    print(f"--- Начинаю нарезку ({total_files} файлов) в 4 потока... ---")
    
    all_dialogs = {}
    completed = 0

    # Используем as_completed для вывода прогресса по мере готовности
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_srt = {executor.submit(process_single_srt, srt): srt for srt in srt_files}
        
        for future in as_completed(future_to_srt):
            completed += 1
            prefix, res_map = future.result()
            if res_map:
                all_dialogs.update(res_map)
                print(f"[{completed}/{total_files}] Завершен: {prefix}")
            else:
                print(f"[{completed}/{total_files}] Ошибка (нет аудио): {prefix}")

    # Запись Dialogs.txt
    with open(DIALOGS_FILE, 'w', encoding='utf-8') as f:
        # Сортируем ключи, чтобы в файле был порядок
        for k in sorted(all_dialogs.keys()):
            f.write(f"{k}={all_dialogs[k]}\n")

    print("\n--- Генерация CSV метаданных ---")
    header = "audio_file|text|speaker_name\n"
    TRAIN_CSV.write_text(header, encoding='utf-8')
    EVAL_CSV.write_text(header, encoding='utf-8')

    valid_wavs = list(WAVS_DIR.glob('*.wav'))
    random.shuffle(valid_wavs)
    eval_count = max(1, int(len(valid_wavs) * 0.15))

    def write_csv(files, path):
        with open(path, 'a', encoding='utf-8') as f:
            for wav in files:
                if wav.stem in all_dialogs:
                    f.write(f"wavs/{wav.name}|{all_dialogs[wav.stem]}|coqui\n")

    write_csv(valid_wavs[:eval_count], EVAL_CSV)
    write_csv(valid_wavs[eval_count:], TRAIN_CSV)

    print(f"Готово! Всего фрагментов: {len(valid_wavs)}")
    print(f"Результаты в папке: {DATASET_DIR.name}")

if __name__ == "__main__":
    main()
