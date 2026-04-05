import os
import re
import shutil
import subprocess
import random
from pathlib import Path

# --- НАСТРОЙКИ ---
BASE_DIR = Path(__file__).parent.absolute()
FFMPEG_PATH = str(BASE_DIR / "ffmpeg.exe") 
WORK_DIR = BASE_DIR / 'work'
DATASET_DIR = BASE_DIR / 'dataset'
WAVS_DIR = DATASET_DIR / 'wavs'
TRAIN_CSV = DATASET_DIR / 'metadata_train.csv'
EVAL_CSV = DATASET_DIR / 'metadata_eval.csv'
LANG_FILE = DATASET_DIR / 'lang.txt'
DIALOGS_FILE = BASE_DIR / 'Dialogs.txt'

SRT_PROG = re.compile(r"([0-9:\.]+).+?([0-9:\.]+)")

def check_lang_file():
    """Проверяет наличие lang.txt, создает если отсутствует."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if not LANG_FILE.exists():
        print(f"--- Файл {LANG_FILE.name} не найден. Создаю (по умолчанию: en) ---")
        LANG_FILE.write_text("en", encoding='utf-8')
    else:
        current_lang = LANG_FILE.read_text(encoding='utf-8').strip()
        print(f"--- Язык обучения: {current_lang} ---")

def process_srt_to_audio(srt_path):
    """Нарезка аудио по таймкодам из SRT."""
    in_file = srt_path.with_suffix('.wav')
    if not in_file.exists():
        print(f"Ошибка: Исходный аудиофайл {in_file.name} не найден в {WORK_DIR}")
        return

    target_folder = srt_path.with_suffix('')
    target_folder.mkdir(parents=True, exist_ok=True)
    folder_name = target_folder.name

    with open(srt_path, "r", encoding='utf-8-sig') as srt_file:
        lines = srt_file.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            out_file_idx = line
            i += 1
            if i < len(lines):
                time_line = lines[i].strip().replace(",", ".")
                sm_match = SRT_PROG.match(time_line)
                if sm_match:
                    ss, to = sm_match.group(1), sm_match.group(2)
                    file_name = f"{folder_name}_{out_file_idx.rjust(3, '0')}.wav"
                    output_path = target_folder / file_name
                    
                    cmd = [
                        FFMPEG_PATH, "-loglevel", "quiet", "-y",
                        "-i", str(in_file),
                        "-ss", ss, "-to", to, 
                        "-f", "wav", str(output_path)
                    ]
                    try:
                        subprocess.run(cmd, check=True)
                    except Exception as e:
                        print(f"Ошибка ffmpeg на файле {file_name}: {e}")
        i += 1

def main():
    if not WORK_DIR.exists():
        print(f"Ошибка: Папка {WORK_DIR} не найдена!")
        return

    check_lang_file()

    if WAVS_DIR.exists():
        shutil.rmtree(WAVS_DIR)
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Очищаем старый Dialogs.txt
    if DIALOGS_FILE.exists():
        os.remove(DIALOGS_FILE)

    dialog_map = {}
    srt_files = list(WORK_DIR.glob('*.srt'))

    for srt_path in srt_files:
        file_prefix = srt_path.stem
        print(f"\nОбработка: {file_prefix}")

        process_srt_to_audio(srt_path)

        output_lines = []
        with open(srt_path, "r", encoding='utf-8-sig') as fp:
            lines = fp.read().strip().split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.isdigit():
                    idx = line.rjust(3, "0")
                    if i + 2 < len(lines):
                        text = lines[i + 2].strip()
                        if text:
                            entry_name = f"{file_prefix}_{idx}"
                            output_lines.append(f"{entry_name}={text}")
                            dialog_map[entry_name] = text
                    i += 2
                i += 1
        
        if output_lines:
            result = '\n'.join(output_lines)
            print(result)
            with open(DIALOGS_FILE, 'a', encoding='utf-8') as f:
                f.write(result + '\n')

        source_folder = srt_path.with_suffix('')
        if source_folder.exists() and source_folder.is_dir():
            for wav_file in source_folder.glob('*.wav'):
                shutil.move(str(wav_file), str(WAVS_DIR / wav_file.name))
            shutil.rmtree(source_folder)

    print("\n--- Генерация CSV метаданных ---")
    header = "audio_file|text|speaker_name\n"
    # Пишем заголовки с UTF-8
    TRAIN_CSV.write_text(header, encoding='utf-8')
    EVAL_CSV.write_text(header, encoding='utf-8')

    valid_wavs = [w for w in WAVS_DIR.glob('*.wav') if w.stem in dialog_map]
    if not valid_wavs:
        print("Ошибка: Аудиофайлы не созданы. Проверьте ffmpeg.exe.")
        return

    random.shuffle(valid_wavs)
    eval_count = max(1, int(len(valid_wavs) * 0.15))
    
    eval_files = valid_wavs[:eval_count]
    train_files = valid_wavs[eval_count:]

    # Запись CSV с UTF-8
    for csv_path, files, label in [(TRAIN_CSV, train_files, "TRAIN"), (EVAL_CSV, eval_files, "EVAL")]:
        with open(csv_path, 'a', encoding='utf-8') as f:
            for wav in files:
                text = dialog_map[wav.stem]
                f.write(f"wavs/{wav.name}|{text}|coqui\n")
                print(f"wavs/{wav.name} [{label}]")

    print(f"\nГотово! Кодировка UTF-8 применена ко всем файлам.")

if __name__ == "__main__":
    main()
