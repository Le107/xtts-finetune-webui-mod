import os
import re
from pathlib import Path

def load_custom_dict(dict_path):
    cdict = {}
    if not dict_path.exists():
        print(f"[!] Файл словаря {dict_path.name} не найден. Пропускаю...")
        return cdict
    
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                word, stressed = line.strip().split("=", 1)
                cdict[word.lower()] = stressed
    return cdict

def add_stress_manual():
    input_file = Path("Dialogs_ru.txt")
    output_file = Path("Dialogs_ru_stressed.txt")
    dict_file = Path("custom_dict.txt")
    
    if not input_file.exists():
        print(f"[!] Ошибка: {input_file.name} не найден!")
        return

    # Загружаем ваш словарь из файла
    CUSTOM_DICT = load_custom_dict(dict_file)
    print(f"[*] Загружено слов в словарь: {len(CUSTOM_DICT)}")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    print(f"[*] Обработка диалогов...")

    for line in lines:
        if "=" in line:
            line_id, text = line.strip().split("=", 1)
            
            # Применяем замены из словаря
            for word, stressed in CUSTOM_DICT.items():
                # Регулярное выражение ищет слово целиком, игнорируя регистр
                pattern = re.compile(rf'\b{word}\b', re.IGNORECASE)
                text = pattern.sub(stressed, text)
            
            results.append(f"{line_id}={text}")
        else:
            results.append(line.strip())

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"--- ГОТОВО! ---")
    print(f"Результат сохранен в: {output_file.name}")

if __name__ == "__main__":
    add_stress_manual()
