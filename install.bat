python -m venv venv
call venv/scripts/activate

.\venv\Scripts\python.exe -m pip install --upgrade pip
pip uninstall torch torchaudio
pip install torch torchaudio
pip install -r .\requirements.txt

python xtts_demo.py