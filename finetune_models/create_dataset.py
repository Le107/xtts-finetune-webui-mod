import os
import random
from pathlib import Path

# Переходим в папку с аудио
os.chdir('dataset/wavs')

# Пути к файлам
train_path = Path('../../dataset/metadata_train.csv')
eval_path = Path('../../dataset/metadata_eval.csv')
dialogs_path = Path('../../Dialogs.txt')

# Создаем/очищаем файлы и записываем заголовки
header = "audio_file|text|speaker_name\n"
train_path.write_text(header, encoding='utf-8')
eval_path.write_text(header, encoding='utf-8')

# 1. Читаем диалоги в словарь {имя_файла: текст}
dialog_map = {}
if dialogs_path.exists():
    with open(dialogs_path, "r", encoding='utf-8-sig') as f:
        for line in f:
            if '=' in line:
                name_part, text_part = line.strip().split('=', 1)
                # Сохраняем только если текст после '=' не пустой
                if text_part.strip():
                    dialog_map[name_part] = text_part.strip()

# 2. Собираем список всех существующих wav файлов
all_wavs = list(Path('.').glob('*.wav'))

# Фильтруем список: оставляем только те файлы, для которых есть текст в Dialogs.txt
valid_wavs = [w for w in all_wavs if w.stem in dialog_map]

# Выводим предупреждение о файлах без текста
missing = len(all_wavs) - len(valid_wavs)
if missing > 0:
    print(f"Предупреждение: Пропущено {missing} файлов (нет текста в Dialogs.txt)\n")

# 3. Распределяем файлы (минимум 1 в eval, если есть хотя бы 1 файл)
if valid_wavs:
    random.shuffle(valid_wavs)
    
    # Считаем количество для eval (минимум 1, максимум 15%)
    eval_count = max(1, int(len(valid_wavs) * 0.15))
    
    # Если файлов всего 1, он пойдет в eval (или можно изменить на train)
    eval_files = valid_wavs[:eval_count]
    train_files = valid_wavs[eval_count:]

    def write_to_csv(file_list, csv_path, mode_label):
        with open(csv_path, 'a', encoding='utf-8') as f:
            for wav_path in file_list:
                dialog = dialog_map[wav_path.stem]
                line = f"wavs/{wav_path.name}|{dialog}|coqui"
                f.write(line + '\n')
                print(f"{line} [{mode_label}]")

    # Записываем результаты
    if train_files:
        write_to_csv(train_files, train_path, "TRAIN")
    if eval_files:
        write_to_csv(eval_files, eval_path, "EVAL")

print(f"\nГотово! В train: {len(train_files)}, в eval: {len(eval_files)}")
