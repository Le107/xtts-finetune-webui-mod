import os
import shutil
from pathlib import Path
import numpy as np

os.chdir('work')

for item in Path('.').glob('*.srt'):
    name = str(item)
    res = name.rstrip('.srt')
    with open(name, "r", encoding='utf-8-sig') as fp:
        content = fp.read()
        lines = content.strip().split('\n')
        output = []
        i = 0
        while i < len(lines):
            if lines[i].isdigit():
                index = str(lines[i]).rjust(3,"0")
                i += 2  # Пропускаем тайм-коды
                output.append(f"{index}={lines[i]}")
            i += 1
        result = '\n'.join(output)
        print(result)
        print(result, file=open(res+"_conv.srt", 'a', encoding='utf-8'))
    os.rename (res, 'wavs')
    os.rename (res+'_conv.srt', 'Dialogs.txt')
    shutil.move('wavs', '..\dataset\wavs')    
    shutil.move('Dialogs.txt', '..\Dialogs.txt')