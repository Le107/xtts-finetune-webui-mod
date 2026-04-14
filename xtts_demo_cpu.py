import argparse
import os
import sys
import tempfile
from pathlib import Path

import os
import shutil
import glob
import re
import time
import pandas as pd
from sklearn.model_selection import train_test_split

import gradio as gr
import librosa.display
import numpy as np

import torch
import torchaudio
import traceback
from pydub import AudioSegment
from utils.formatter import format_audio_list,find_latest_best_model, list_audios
from utils.gpt_train import train_gpt

from faster_whisper import WhisperModel

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Очистить журналы
def remove_log_file(file_path):
     log_file = Path(file_path)

     if log_file.exists() and log_file.is_file():
         log_file.unlink()

# remove_log_file(str(Path.cwd() / "log.out"))

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None
def load_model(xtts_checkpoint, xtts_config, xtts_vocab,xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "Вам необходимо выполнить предыдущие шаги или вручную установить поля «Путь контрольной точки XTTS», «Путь конфигурации XTTS» и «Путь vocab XTTS»!!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Загрузка модели XTTS! ")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab,speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Модель загружена!")
    return "Модель загружена!"

def run_tts(lang, tts_text, speaker_audio_file, temperature, speed, length_penalty,repetition_penalty,top_k,top_p,sentence_split,use_config):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "Для загрузки модели необходимо выполнить предыдущий шаг!!", None, None

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    
    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature, # Добавить сюда пользовательские параметры
            speed=XTTS_MODEL.config.speed,
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting = True
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature, # Добавить сюда пользовательские параметры
            speed=speed,
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting = sentence_split
        )

    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    if not os.path.exists("output"):
        os.makedirs("output")
    
    save_path = os.path.join("output", os.path.basename(speaker_audio_file))
    torchaudio.save(save_path, out["wav"], 24000)

    try:
        ref_audio = AudioSegment.from_file(speaker_audio_file)
        gen_audio = AudioSegment.from_file(save_path)
        # Приводим и к частоте, и к количеству каналов (стерео/моно) оригинала
        new_audio = gen_audio.set_frame_rate(ref_audio.frame_rate).set_channels(ref_audio.channels)
        new_audio.export(save_path, format="wav")
    except Exception as e:
        print(f" [!] Ошибка обработки одиночного вывода: {e}")

    return "Речь сгенерирована!", save_path, speaker_audio_file

def mass_predict_tts(dialogs_file, speaker_wav_dir, lang, temperature, speed, length_penalty, repetition_penalty, top_k, top_p, sentence_split, use_config):
    global XTTS_MODEL
    if XTTS_MODEL is None:
        return "Ошибка: Сначала загрузите модель (Шаг 3)!"
    
    dialogs_path = Path(args.out_path) / dialogs_file
    if not dialogs_path.exists():
        return f"Ошибка: Файл {dialogs_file} не найден в {args.out_path}"

    mass_out_dir = Path("translated_output")
    mass_out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = Path(args.out_path) / speaker_wav_dir
    
    with open(dialogs_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if "=" in l]

    total = len(lines)
    processed = 0
    print(f"--- Начало массовой озвучки (Всего фраз: {total}) ---")

    for line in lines:
        line_id, raw_text = line.split("=", 1)

        # --- ОЧИСТКА ТЕКСТА ---
        # 1. Удаляем кавычки, скобки и спецсимволы
        tts_text = raw_text.translate(str.maketrans('', '', '"\'«»“”()[]{}*<>'))
        
        # 2. Правим тире и убираем лишние пробелы
        tts_text = tts_text.replace('--', ' — ').replace(' - ', ' — ')
        tts_text = re.sub(r'\s+', ' ', tts_text)
        
        # 3. Убираем точку в конце (Борьба с "Point")
        tts_text = tts_text.strip()
        if tts_text.endswith('.') and len(tts_text) > 3:
            tts_text = tts_text[:-1]
            
        # 4. Финальный пробел для мягкого завершения фразы
        tts_text = tts_text.strip() + " "
        
        speaker_audio_file = str(wav_dir / f"{line_id}.wav")
        if not os.path.exists(speaker_audio_file):
            print(f" [!] Пропуск {line_id}: референс не найден")
            continue

        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=speaker_audio_file, 
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, 
            max_ref_length=XTTS_MODEL.config.max_ref_len, 
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs
        )
        
        if use_config:
            out = XTTS_MODEL.inference(
                text=tts_text, language=lang,
                gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding,
                temperature=XTTS_MODEL.config.temperature, speed=XTTS_MODEL.config.speed,
                length_penalty=XTTS_MODEL.config.length_penalty,
                repetition_penalty=XTTS_MODEL.config.repetition_penalty,
                top_k=XTTS_MODEL.config.top_k, top_p=XTTS_MODEL.config.top_p,
                enable_text_splitting=True
            )
        else:
            out = XTTS_MODEL.inference(
                text=tts_text, language=lang,
                gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding,
                temperature=temperature, speed=speed,
                length_penalty=length_penalty, repetition_penalty=float(repetition_penalty),
                top_k=top_k, top_p=top_p, enable_text_splitting=sentence_split
            )

        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        save_path = str(mass_out_dir / f"{line_id}.wav")
        torchaudio.save(save_path, out["wav"], 24000)
        
        try:
            ref_audio = AudioSegment.from_file(speaker_audio_file)
            gen_audio = AudioSegment.from_file(save_path)
            new_audio = gen_audio.set_frame_rate(ref_audio.frame_rate).set_channels(ref_audio.channels)
            new_audio.export(save_path, format="wav")
        except Exception as e:
            print(f" [!] Ошибка обработки аудио для {line_id}: {e}")

        processed += 1
        print(f"[{processed}/{total}] Готово: {line_id}, Текст: {tts_text}")

    return f"Массовая озвучка завершена! Обработано файлов: {processed}. Проверьте папку translated_output."

def update_dialogs():
    base_path = Path(args.out_path)
    files = [f.name for f in base_path.glob("Dialogs*.txt")]
    # Сортировка: Dialogs.txt всегда сверху
    files.sort(key=lambda x: (x != "Dialogs.txt", x))    
    # Возвращаем обновление с новым списком
    return gr.update(choices=files)

def load_params_tts(out_path,version):
    
    out_path = Path(out_path)

    # base_model_path = Path.cwd() / "models" / version 

    # if not base_model_path.exists():
    #     return "Base model not found !","","",""

    ready_model_path = out_path / "ready" 

    vocab_path =  ready_model_path / "vocab.json"
    config_path = ready_model_path / "config.json"
    speaker_path =  ready_model_path / "speakers_xtts.pth"
    reference_path  = ready_model_path / "reference.wav"

    model_path = ready_model_path / "model.pth"

    if not model_path.exists():
        model_path = ready_model_path / "unoptimize_model.pth"
        if not model_path.exists():
          return "Параметры для TTS не найдены", "", "", ""         

    return "Параметры для TTS не найдены", model_path, config_path, vocab_path,speaker_path, reference_path
     

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning demo\n\n"""
        """
        Example runs:
        python3 TTS/demos/xtts_ft_demo/xtts_demo.py --port 
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Порт для запуска gradio demo. По умолчанию: 5003",
        default=5003,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Путь вывода (куда будут сохраняться данные и контрольные точки) По умолчанию: output/",
        default=str(Path.cwd() / "finetune_models"),
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Количество эпох для тренировки. По умолчанию: 100",
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Объем серии. По умолчанию: 2",
        default=2,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Ступени накопления града. По умолчанию: 128",
        default=128,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Максимальный допустимый размер звука в секундах. По умолчанию: 20",
        default=20,
    )

    args = parser.parse_args()

    with gr.Blocks() as demo:
        with gr.Tab("1 - Обработка данных"):
            out_path = gr.Textbox(
                label="Путь вывода (куда будут сохраняться данные и контрольные точки):",
                value=args.out_path,
            )
            # upload_file = gr.Audio(
            #     sources="upload",
            #     label="Select here the audio files that you want to use for XTTS trainining !",
            #     type="filepath",
            # )
            upload_file = gr.File(
                file_count="multiple",
                label="Выберите здесь аудиофайлы, которые вы хотите использовать для обучения XTTS (поддерживаемые форматы: wav, mp3 и flac)",
            )
            
            audio_folder_path = gr.Textbox(
                label="Путь к папке с аудиофайлами (необязательно):",
                value="",
            )

            whisper_model = gr.Dropdown(
                label="Whisper Model",
                value="large-v3",
                choices=[
                    "large-v3",
                    "large-v2",
                    "large",
                    "medium",
                    "small"
                ],
            )

            lang = gr.Dropdown(
                label="Язык набора данных",
                value="en",
                choices=[
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "pl",
                    "tr",
                    "ru",
                    "nl",
                    "cs",
                    "ar",
                    "zh",
                    "hu",
                    "ko",
                    "ja"
                ],
            )
            progress_data = gr.Label(
                label="Прогресс:"
            )
            # demo.load(read_logs, None, logs, every=1)

            prompt_compute_btn = gr.Button(value="Шаг 1 - Создание набора данных")
        
            def preprocess_dataset(audio_path, audio_folder_path, language, whisper_model, out_path, train_csv, eval_csv, progress=gr.Progress(track_tqdm=True)):
                clear_gpu_cache()
            
                train_csv = ""
                eval_csv = ""
            
                out_path = os.path.join(out_path, "dataset")
                os.makedirs(out_path, exist_ok=True)
            
                if audio_folder_path:
                    audio_files = list(list_audios(audio_folder_path))
                else:
                    audio_files = audio_path
            
                if not audio_files:
                    return "Аудиофайлы не найдены! Предоставьте файлы через Gradio или укажите путь к папке.", "", ""
                else:
                    try:
                        # Loading Whisper
                        device = "cuda" if torch.cuda.is_available() else "cpu" 
                        
                        # Detect compute type 
                        if torch.cuda.is_available():
                            compute_type = "float32"
                        else:
                            compute_type = "float32"
                        
                        asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
                        train_meta, eval_meta, audio_total_size = format_audio_list(audio_files, asr_model=asr_model, target_language=language, out_path=out_path, gradio_progress=progress)
                    except:
                        traceback.print_exc()
                        error = traceback.format_exc()
                        return f"Обработка данных прервана из-за ошибки!! Проверьте консоль, чтобы получить полное сообщение об ошибке !\n Сводка об ошибке: {error}", "", ""
            
                # clear_gpu_cache()
            
                # if audio total len is less than 2 minutes raise an error
                if audio_total_size < 120:
                    message = "Общая продолжительность аудиозаписей должна составлять не менее 2 минут!"
                    print(message)
                    return message, "", ""
            
                print("Обработан набор данных!")
                return "Обработан набор данных!", train_meta, eval_meta


        with gr.Tab("2 - Детальная настройка кодера XTTS"):
            load_params_btn = gr.Button(value="Загрузить параметры из папки вывода")
            version = gr.Dropdown(
                label="Базовая версия XTTS",
                value="v2.0.2",
                choices=[
                    "v2.0.3",
                    "v2.0.2",
                    "v2.0.1",
                    "v2.0.0",
                    "main"
                ],
            )
            train_csv = gr.Textbox(
                label="Train CSV:",
            )
            eval_csv = gr.Textbox(
                label="Eval CSV:",
            )
            custom_model = gr.Textbox(
                label="(Необязательно) Пользовательский файл model.pth, оставьте пустым, если требуется использовать базовый файл.",
                value="",
            )
            num_epochs =  gr.Slider(
                label="Количество периодов:",
                minimum=1,
                maximum=100,
                step=1,
                value=args.num_epochs,
            )
            batch_size = gr.Slider(
                label="Объем серии:",
                minimum=2,
                maximum=512,
                step=1,
                value=args.batch_size,
            )
            grad_acumm = gr.Slider(
                label="Этапы накопления Града:",
                minimum=2,
                maximum=128,
                step=1,
                value=args.grad_acumm,
            )
            max_audio_length = gr.Slider(
                label="Максимальный допустимый размер звука в секундах:",
                minimum=2,
                maximum=20,
                step=1,
                value=args.max_audio_length,
            )
            clear_train_data = gr.Dropdown(
                label="Очистить данные тренировки, удалятся папки полсе тренировки",
                value="run",
                choices=[
                    "none",
                    "run",
                    "dataset",
                    "all"
                ])
            
            progress_train = gr.Label(
                label="Прогресс:"
            )

            # demo.load(read_logs, None, logs_tts_train, every=1)
            train_btn = gr.Button(value="Шаг 2 - Проведение обучения")

            def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, clear_train_data):
                clear_gpu_cache()
                
                # Проверка языка
                lang_file_path = Path(output_path) / "dataset" / "lang.txt"
                current_language = None
                if lang_file_path.exists():
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()
                        if current_language != language:
                            print("Язык датасета не совпадает, переключаюсь на:", current_language)
                            language = current_language
                        
                if not train_csv or not eval_csv:
                    return "Ошибка: Установите Train CSV и Eval CSV!", "", "", "", ""
                
                # Подготовка переменных перед циклом
                current_custom_model = custom_model
                max_attempts = 10
                dataset_path = Path(output_path) / "dataset"
                # Вычисляем длину в сэмплах ОДИН раз здесь, чтобы не умножать в цикле
                max_audio_length_samples = int(max_audio_length * 22050)
                
                # Переменные для финального возврата
                final_status = "Обучение завершено!"
                
                for i in range(max_attempts):
                    print(f"--- Итерация обучения {i+1}/{max_attempts} ---")
                    
                    # Перемешивание данных (начиная со второй итерации)
                    if i > 0:
                        df_train = pd.read_csv(train_csv, sep='|')
                        df_eval = pd.read_csv(eval_csv, sep='|')
                        df_all = pd.concat([df_train, df_eval])
            
                        # Вычисляем количество для теста (минимум 1 запись)
                        eval_count = max(1, int(len(df_all) * 0.15))
                        
                        # Перемешиваем и делим
                        new_train, new_eval = train_test_split(df_all, test_size=eval_count, random_state=42)
                        new_train.to_csv(train_csv, index=False, sep='|')
                        new_eval.to_csv(eval_csv, index=False, sep='|')
                        print(f" [+] Данные перемешаны для итерации {i+1}.")

                    try:
                        max_audio_length = max_audio_length_samples
                        # ЗАПУСК ТРЕНИРОВКИ (внутри цикла)
                        speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
                            current_custom_model, version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, 
                            output_path=output_path, max_audio_length=max_audio_length
                        )
                    except Exception as e:
                        traceback.print_exc()
                        return f"Ошибка тренировки на итерации {i+1}: {traceback.format_exc()}", "", "", "", ""
                        
                    loss_file_path = os.path.join(exp_path, "last_avg_loss.txt")
                    avg_loss = 4.0  # Значение по умолчанию (если файл не найдется)
                
                    if os.path.exists(loss_file_path):
                        with open(loss_file_path, "r") as f:
                            avg_loss = float(f.read().strip())
                        print(f" [📊] Считан Loss из файла: {avg_loss}")
                    else:
                        print(f" [!] Файл {loss_file_path} не найден. Используется Loss по умолчанию: 4.0")

                    # --- ЛОГИКА ПЕРЕНОСА И ОБНОВЛЕНИЯ ПУТИ ---
                    ready_dir = Path(output_path) / "ready"
                    ready_dir.mkdir(parents=True, exist_ok=True)

                    optimized_model_src = os.path.join(exp_path, "model.pth")
                    ft_xtts_checkpoint = ready_dir / "model.pth"

                    if os.path.exists(optimized_model_src):
                        shutil.move(optimized_model_src, ft_xtts_checkpoint)
                        # Теперь эта модель станет базовой для следующего шага
                        current_custom_model = str(ft_xtts_checkpoint)
                        print(f" [+] Модель перенесена в: {ft_xtts_checkpoint}")
                    else:
                        print(f" [!] ВНИМАНИЕ: Файл model.pth не найден в {exp_path}")

                    # Проверка Loss
                    if float(avg_loss) < 3:
                        print(f" [✅] Цель достигнута! Финальный Loss: {avg_loss}")
                        break 
                    else:
                        if i < max_attempts - 1:
                            print(f" [♻️] Loss {avg_loss:.4f} > 3. Запуск следующей итерации...")
                            clear_gpu_cache()
                        else:
                            print(f" [!] Попытки исчерпаны. Остановка на Loss: {avg_loss:.4f}")

                # --- ФИНАЛИЗАЦИЯ (ВНЕ ЦИКЛА) ---
                
                # Копируем референс
                speaker_reference_path = Path(speaker_wav)
                speaker_reference_new_path = ready_dir / "reference.wav"
                shutil.copy(speaker_reference_path, speaker_reference_new_path)

                # Очистка данных
                if clear_train_data in ["dataset", "all"]:
                    if dataset_path.exists():
                        shutil.rmtree(dataset_path, ignore_errors=True)
                        print(f" [🗑️] Папка датасета удалена.")

                if clear_train_data in ["run", "all"]:
                    run_dir = Path(output_path) / "run"
                    if run_dir.exists():
                        time.sleep(1)
                        shutil.rmtree(run_dir, ignore_errors=True)
                        print(f" [🗑️] Папка временных файлов (run) удалена.")

                return "Обучение завершено!", config_path, vocab_file, str(ft_xtts_checkpoint), speaker_xtts_path, str(speaker_reference_new_path)

            def load_params(out_path):
                path_output = Path(out_path)
                
                dataset_path = path_output / "dataset"

                if not dataset_path.exists():
                    return "Папка вывода не существует!", "", ""

                eval_train = dataset_path / "metadata_train.csv"
                eval_csv = dataset_path / "metadata_eval.csv"

                # Write the target language to lang.txt in the output directory
                lang_file_path =  dataset_path / "lang.txt"

                # Check if lang.txt already exists and contains a different language
                current_language = None
                if os.path.exists(lang_file_path):
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()

                clear_gpu_cache()

                print(current_language)
                return "Данные обновлены", eval_train, eval_csv, current_language

        with gr.Tab("3 - Вывод"):
            with gr.Row():
                with gr.Column() as col1:
                    load_params_tts_btn = gr.Button(value="Загрузка параметров для TTS из папки вывода")
                    xtts_checkpoint = gr.Textbox(
                        label="Путь контрольной точки XTTS:",
                        value="",
                    )
                    xtts_config = gr.Textbox(
                        label="Путь к конфигурационному элементу XTTS:",
                        value="",
                    )

                    xtts_vocab = gr.Textbox(
                        label="Путь XTTS vocab:",
                        value="",
                    )
                    xtts_speaker = gr.Textbox(
                        label="Путь к динамику XTTS:",
                        value="",
                    )
                    progress_load = gr.Label(
                        label="Прогресс:"
                    )
                    load_btn = gr.Button(value="Шаг 3 - Загрузка точной настройки модели XTTS")

                with gr.Column() as col2:
                    speaker_reference_audio = gr.Textbox(
                        label="Эталонный звук динамика:",
                        value="",
                    )
                    tts_language = gr.Dropdown(
                        label="Язык",
                        value="ru",
                        choices=[
                            "en",
                            "es",
                            "fr",
                            "de",
                            "it",
                            "pt",
                            "pl",
                            "tr",
                            "ru",
                            "nl",
                            "cs",
                            "ar",
                            "zh",
                            "hu",
                            "ko",
                            "ja",
                        ]
                    )
                    tts_text = gr.Textbox(
                        label="Входной текст:",
                        value="Эта модель звучит действительно хорошо.",
                    )
                    with gr.Accordion("Дополнительные настройки", open=False) as acr:
                        temperature = gr.Slider(
                            label="температура",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.1,
                        )
                        speed = gr.Slider(
                            label="скорость",
                            minimum=0.1,
                            maximum=2,
                            step=0.1,
                            value=1,
                        )
                        length_penalty  = gr.Slider(
                            label="штраф за длину",
                            minimum=-10.0,
                            maximum=10.0,
                            step=0.5,
                            value=1,
                        )
                        repetition_penalty = gr.Slider(
                            label="штраф за повторение",
                            minimum=1,
                            maximum=10,
                            step=0.5,
                            value=5,
                        )
                        top_k = gr.Slider(
                            label="top_k",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=100,
                        )
                        top_p = gr.Slider(
                            label="top_p",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=1,
                        )
                        sentence_split = gr.Checkbox(
                            label="Включить разделение текста",
                            value=True,
                        )
                        use_config = gr.Checkbox(
                            label="Использовать настройки вывода из конфигурации, если они отключены, использовать настройки, указанные выше",
                            value=False,
                        )
                    tts_btn = gr.Button(value="Шаг 4 - Вывод")

                with gr.Column() as col3:
                    progress_gen = gr.Label(
                        label="Прогресс:"
                    )
                    tts_output_audio = gr.Audio(label="Сгенерированное аудио.")
                    reference_audio = gr.Audio(label="Использованный эталонный звук.")
                    
                with gr.Column() as col4:
                    gr.Markdown("### Массовая озвучка")
                    files = [f.name for f in Path(args.out_path).glob("Dialogs*.txt")]
                    
                    with gr.Row():
                        dialogs_dropdown = gr.Dropdown(
                            label="Файл диалогов:",
                            choices=files,
                            value="Dialogs.txt",
                            interactive=True # Явно разрешаем взаимодействие
                        )
                        
                    wavs_folder_input = gr.Textbox(
                        label="Папка с референсами (оригиналами):",
                        value="dataset/wavs",
                    )
                    mass_tts_btn = gr.Button(value="ЗАПУСТИТЬ МАССОВУЮ ОЗВУЧКУ", variant="primary")
                    mass_status = gr.Label(label="Статус:")

                    # Связываем новую кнопку с функцией обновления
                    dialogs_dropdown.select(fn=update_dialogs, outputs=dialogs_dropdown)
                    
            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[
                    upload_file,
                    audio_folder_path,
                    lang,
                    whisper_model,
                    out_path,
                    train_csv,
                    eval_csv
                ],
                outputs=[
                    progress_data,
                    train_csv,
                    eval_csv,
                ],
            )


            load_params_btn.click(
                fn=load_params,
                inputs=[out_path],
                outputs=[
                    progress_train,
                    train_csv,
                    eval_csv,
                    lang
                ]
            )


            train_btn.click(
                fn=train_model,
                inputs=[
                    custom_model,
                    version,
                    lang,
                    train_csv,
                    eval_csv,
                    num_epochs,
                    batch_size,
                    grad_acumm,
                    out_path,
                    max_audio_length,
                    clear_train_data,
                ],
                outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint,xtts_speaker, speaker_reference_audio],
            )
           
            load_btn.click(
                fn=load_model,
                inputs=[
                    xtts_checkpoint,
                    xtts_config,
                    xtts_vocab,
                    xtts_speaker
                ],
                outputs=[progress_load],
            )

            tts_btn.click(
                fn=run_tts,
                inputs=[
                    tts_language,
                    tts_text,
                    speaker_reference_audio,
                    temperature,
                    speed,
                    length_penalty,
                    repetition_penalty,
                    top_k,
                    top_p,
                    sentence_split,
                    use_config
                ],
                outputs=[progress_gen, tts_output_audio,reference_audio],
            )

            load_params_tts_btn.click(
                fn=load_params_tts,
                inputs=[
                    out_path,
                    version
                    ],
                outputs=[progress_load,xtts_checkpoint,xtts_config,xtts_vocab,xtts_speaker,speaker_reference_audio],
            )
            
            mass_tts_btn.click(
                fn=mass_predict_tts,
                inputs=[
                    dialogs_dropdown, wavs_folder_input, tts_language,
                    temperature, speed, length_penalty, repetition_penalty,
                    top_k, top_p, sentence_split, use_config
                ],
                outputs=mass_status
            )            

    demo.launch(
        server_name="127.0.0.1",
        share=False,
        debug=False,
        server_port=args.port,
        # inweb=True,
        # server_name="localhost"
    )
