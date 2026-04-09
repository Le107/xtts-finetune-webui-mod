Поместите сюда файл `ffmpeg.exe`, а в папку `work` звуковые файлы ('.wav', '.mp3', '.ogg', '.flac', '.m4a') и .srt с одинаковым именем. Далее `start.bat`(скрипт 
`Sub_to_dataset.py` нарежет звуковые файлы из папки `work` по субтитрам, также создаст файлы `metadata_eval.csv` `metadata_train.csv` и поместит все в `
dataset`, и еще создаст файл `Dialogs.txt`(про `Dialogs.txt` см. в основном `readme.md`)). При необходимости смените язык в `dataset\lang.txt`, по умолчанию: en. Для разделения субтитров по ролям используйте Aegisub.
