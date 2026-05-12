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
            line = line.strip()
            if "=" in line:
                # Берем только первое вхождение "=", остальное — значение
                word, stressed = line.split("=", 1)
                # Ключ храним в нижнем регистре для удобного поиска
                cdict[word.lower().strip()] = stressed.strip()
    return cdict

def add_stress_manual():
    input_file = Path("Dialogs_ru.txt")
    output_file = Path("Dialogs_ru_stressed.txt")
    dict_file = Path("custom_dict.txt")
    
    if not input_file.exists():
        print(f"[!] Ошибка: {input_file.name} не найден!")
        return

    # Загружаем словарь
    CUSTOM_DICT = load_custom_dict(dict_file)
    print(f"[*] Загружено записей в словарь: {len(CUSTOM_DICT)}")

    # Сортируем ключи: сначала самые длинные фразы, потом короткие слова.
    # Это нужно, чтобы "не на что" заменилось раньше, чем "не".
    sorted_keys = sorted(CUSTOM_DICT.keys(), key=len, reverse=True)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    print(f"[*] Обработка диалогов...")

    for line in lines:
        if "=" in line:
            line_id, text = line.strip().split("=", 1)
            
            for word in sorted_keys:
                stressed_value = CUSTOM_DICT[word]
                
                # Экранируем спецсимволы в ключе и ставим границы слова
                # Если в ключе есть пробелы, \b сработает корректно по краям фразы
                pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)

                def replace_func(match):
                    original_text = match.group(0)
                    # Если исходное слово/фраза начинается с большой буквы
                    if original_text and original_text[0].isupper():
                        # Делаем первую букву замены заглавной
                        return stressed_value[0].upper() + stressed_value[1:]
                    return stressed_value

                text = pattern.sub(replace_func, text)
            
            results.append(f"{line_id}={text}")
        else:
            results.append(line.strip())

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"--- ГОТОВО! ---")
    print(f"Результат сохранен в: {output_file.name}")

if __name__ == "__main__":
    add_stress_manual()
