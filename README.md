1. Добавлена скорость.
2. Добавлено авторесемплирование сгенерированой речи под оригинал(Пример: оригинал: 48000Hz, ген.речь:24000 > 48000 Hz. Оригинал по умолчанию это эталонный звук динамика: reference.wav. Либо указанный другой.)
3. Сохранение в папку 'output', с темже именем файла, что и эталонный звук динамика(по умолчанию: output\reference.wav. Либо указанный другой).
4. Переведено на русский.
5. config:
    "compute_f0": true,
    "compute_energy": true,
    "compute_linear_spec": true,
    "use_speaker_weighted_sampler": true,
    "use_grad_scaler": true,
    "use_noise_augment": true
6. Создание датасета по субтитрам. (см. в finetune_models)

Установка: скопировать всё с заменой на установленный репозиторий от daswer123 (daswer123/xtts-finetune-webui)


Не работает с torch cuda из ошибки https://github.com/SYSTRAN/faster-whisper/issues/42. Поэтому в коде изменено float16 на float32
```docker
                        # Detect compute type 
                        if torch.cuda.is_available():
                            compute_type = "float32"
                        else:
                            compute_type = "float32"
```
