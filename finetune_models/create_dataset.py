import os
from pathlib import Path
import numpy as np

os.chdir('dataset\\wavs')

print("audio_file|text|speaker_name", file=open(f'..\..\dataset\metadata_train.csv', 'a', encoding='utf-8'))    
print("audio_file|text|speaker_name", file=open(f'..\..\dataset\metadata_eval.csv', 'a', encoding='utf-8'))    
lst=[1, 2]
for item in Path('.').glob('*.wav'):
    name = str(item)
    res = name.rstrip('.wav')
    r = np.random.choice(lst,1,p=[0.85, 0.15])
    with open(r"..\..\Dialogs.txt", "r", encoding='utf-8') as fp:
        lines = fp.readlines()
        for row in lines:
            if r == 1:
                if row.find(res) != -1:
                    idx = row.find('=')
                    dialog = str(row[idx + 1:])
                    dialog = dialog[:-1]
                    print("wavs/"+name+"|"+dialog+"|coqui", "train") 
                    print("wavs/"+name+"|"+dialog+"|coqui", file=open(f'..\..\dataset\metadata_train.csv', 'a', encoding='utf-8'))
            else:
                if row.find(res) != -1:
                    idx = row.find('=')
                    dialog = str(row[idx + 1:])
                    dialog = dialog[:-1]
                    print("wavs/"+name+"|"+dialog+"|coqui", "eval") 
                    print("wavs/"+name+"|"+dialog+"|coqui", file=open(f'..\..\dataset\metadata_eval.csv', 'a', encoding='utf-8'))
                