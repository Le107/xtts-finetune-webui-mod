import argparse
import os
import sys
import tempfile
from pathlib import Path

import os
import shutil
import glob

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
    save_path=os.path.basename(speaker_audio_file)
    save_path=str(f'output\{save_path}')
    print(save_path)
    torchaudio.save(save_path, out["wav"], 24000)
    audio = AudioSegment.from_file(speaker_audio_file)
    new_framerate = audio.frame_rate
    audio = AudioSegment.from_file(save_path)
    new_audio = audio.set_frame_rate(new_framerate)
    new_audio.export(save_path, format="wav")

    return "Речь сгенерирована!", save_path, speaker_audio_file


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
        help="Путь вывода (куда будут сохраняться данные и контрольные точки) По умолчанию: finetune_models/",
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
        default=1,
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
                label="Очистить данные поезда, вы удалите выбранную папку после оптимизации",
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
            optimize_model_btn = gr.Button(value="Шаг 2.5 - Оптимизация модели")
            
            def train_model(custom_model,version,language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
                clear_gpu_cache()

                run_dir = Path(output_path) / "run"

                # # Remove train dir
                if run_dir.exists():
                    os.remove(run_dir)
                
                # Check if the dataset language matches the language you specified 
                lang_file_path = Path(output_path) / "dataset" / "lang.txt"

                # Check if lang.txt already exists and contains a different language
                current_language = None
                if lang_file_path.exists():
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()
                        if current_language != language:
                            print("Язык, подготовленный для набора данных, не соответствует указанному языку. Измените язык на язык, указанный в наборе данных")
                            language = current_language
                        
                if not train_csv or not eval_csv:
                    return "Необходимо выполнить шаг обработки данных или вручную установить поля Train CSV и Eval CSV!", "", "", "", ""
                try:
                    # convert seconds to waveform frames
                    max_audio_length = int(max_audio_length * 22050)
                    speaker_xtts_path,config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(custom_model,version,language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return f"Тренировка прервана из-за ошибки!! Проверьте консоль, чтобы получить полное сообщение об ошибке !\n Сводка об ошибке: {error}", "", "", "", ""

                # copy original files to avoid parameters changes issues
                # os.system(f"cp {config_path} {exp_path}")
                # os.system(f"cp {vocab_file} {exp_path}")
                
                ready_dir = Path(output_path) / "ready"

                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")

                shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")
                # os.remove(ft_xtts_checkpoint)

                ft_xtts_checkpoint = os.path.join(ready_dir, "unoptimize_model.pth")

                # Reference
                # Move reference audio to output folder and rename it
                speaker_reference_path = Path(speaker_wav)
                speaker_reference_new_path = ready_dir / "reference.wav"
                shutil.copy(speaker_reference_path, speaker_reference_new_path)

                print("Обучение модели закончено!")
                # clear_gpu_cache()
                return "Обучение модели закончено!", config_path, vocab_file, ft_xtts_checkpoint,speaker_xtts_path, speaker_reference_new_path

            def optimize_model(out_path, clear_train_data):
                # print(out_path)
                out_path = Path(out_path)  # Ensure that out_path is a Path object.
            
                ready_dir = out_path / "ready"
                run_dir = out_path / "run"
                dataset_dir = out_path / "dataset"
            
                # Clear specified training data directories.
                if clear_train_data in {"run", "all"} and run_dir.exists():
                    try:
                        shutil.rmtree(run_dir)
                    except PermissionError as e:
                        print(f"Ошибка при удалении {run_dir}: {e}")
            
                if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
                    try:
                        shutil.rmtree(dataset_dir)
                    except PermissionError as e:
                        print(f"Ошибка при удалении {dataset_dir}: {e}")
            
                # Get full path to model
                model_path = ready_dir / "unoptimize_model.pth"

                if not model_path.is_file():
                    return "Неоптимизированная модель не найдена в готовой папке", ""
            
                # Load the checkpoint and remove unnecessary parts.
                checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
                del checkpoint["optimizer"]

                for key in list(checkpoint["model"].keys()):
                    if "dvae" in key:
                        del checkpoint["model"][key]

                # Make sure out_path is a Path object or convert it to Path
                os.remove(model_path)

                  # Save the optimized model.
                optimized_model_file_name="model.pth"
                optimized_model=ready_dir/optimized_model_file_name
            
                torch.save(checkpoint, optimized_model)
                ft_xtts_checkpoint=str(optimized_model)

                clear_gpu_cache()
        
                return f"Модель оптимизирована и сохранена в {ft_xtts_checkpoint}!", ft_xtts_checkpoint

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
                        label="Входной текст.",
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
                            value=50,
                        )
                        top_p = gr.Slider(
                            label="top_p",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.85,
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
                ],
                outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint,xtts_speaker, speaker_reference_audio],
            )

            optimize_model_btn.click(
                fn=optimize_model,
                inputs=[
                    out_path,
                    clear_train_data
                ],
                outputs=[progress_train,xtts_checkpoint],
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

    demo.launch(
        share=False,
        debug=False,
        server_port=args.port,
        # inweb=True,
        # server_name="localhost"
    )
