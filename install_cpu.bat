python -m venv venv
call venv/scripts/activate

.\venv\Scripts\python.exe -m pip install --upgrade pip
pip uninstall torch torchaudio
pip install torch==2.5.1 torchaudio==2.5.1
pip install -r .\requirements.txt

cmd
