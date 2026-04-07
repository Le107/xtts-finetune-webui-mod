import os
import time
from pathlib import Path
import translators as ts

# Список языков XTTS v2
XTTS_LANGS = {
    "1": ("en", "English"),
    "2": ("ru", "Russian"),
    "3": ("de", "German"),
    "4": ("fr", "French"),
    "5": ("es", "Spanish"),
    "6": ("it", "Italian"),
    "7": ("pt", "Portuguese"),
    "8": ("pl", "Polish"),
    "9": ("tr", "Turkish"),
    "10": ("nl", "Dutch"),
    "11": ("cs", "Czech"),
    "12": ("ar", "Arabic"),
    "13": ("zh-cn", "Chinese"),
    "14": ("hu", "Hungarian"),
    "15": ("ko", "Korean"),
    "16": ("ja", "Japanese"),
    "17": ("hi", "Hindi")
}

def get_lang_choice(prompt):
    print(f"\nВыберите {prompt}:")
    for num, (code, name) in XTTS_LANGS.items():
        print(f"{num.rjust(2)}. {name} [{code}]")
    
    while True:
        choice = input(f"Введите номер (1-17): ").strip()
        if choice in XTTS_LANGS:
            return XTTS_LANGS[choice][0]
        print("Неверный номер, попробуйте еще раз.")

def main():
    print("--- ПЕРЕВОД ДИАЛОГОВ ---")
    
    # 1. Выбор сервиса
    print("\nВыберите сервис перевода:")
    print("1. Google (Быстро)")
    print("2. Yandex (Качественно)")
    engine_choice = input("Введите номер (1 или 2): ").strip()
    translator_name = 'yandex' if engine_choice == '2' else 'google'
    
    # 2. Выбор языков по номерам
    src_code = get_lang_choice("ИСХОДНЫЙ язык (с какого переводим)")
    tgt_code = get_lang_choice("ЦЕЛЕВОЙ язык (на какой переводим)")
    
    input_file = Path("Dialogs.txt")
    output_file = Path(f"Dialogs_{tgt_code}.txt")

    if not input_file.exists():
        print("\n[!] Ошибка: Файл Dialogs.txt не найден!")
        return

    # Загрузка прогресса
    translated_map = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    translated_map[key] = val

    with open(input_file, "r", encoding="utf-8") as f:
        original_lines = [l.strip() for l in f.readlines()]

    results = []
    total = len(original_lines)
    print(f"\n--- Начинаю перевод через {translator_name.upper()} [{src_code} -> {tgt_code}] ---")
    
    try:
        for idx, line in enumerate(original_lines):
            if "=" in line:
                name, text = line.split("=", 1)
                
                if name in translated_map:
                    results.append(f"{name}={translated_map[name]}")
                    continue

                if not text.strip():
                    results.append(line)
                    continue
                
                try:
                    translated = ts.translate_text(
                        text, 
                        translator=translator_name, 
                        from_language=src_code, 
                        to_language=tgt_code
                    )
                    translated = translated.replace("=", " ")
                    
                    final_line = f"{name}={translated}"
                    results.append(final_line)
                    print(f"[{idx+1}/{total}] {final_line}")
                    
                    if len(results) % 5 == 0:
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write("\n".join(results + original_lines[idx+1:]))

                    time.sleep(1.5 if engine_choice == '2' else 0.4)
                except Exception as e:
                    print(f"\n[!] Остановка. Ошибка: {e}")
                    break
            else:
                results.append(line)

    except KeyboardInterrupt:
        print("\n[!] Прервано пользователем.")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results + original_lines[len(results):]))
        
    print(f"\n--- ГОТОВО. Файл: {output_file.name} ---")

if __name__ == "__main__":
    main()
