python -m venv venv
call venv/scripts/activate

.\venv\Scripts\python.exe -m pip install --upgrade pip
pip install -r .\requirements.txt
pip uninstall torch torchaudio
pip install torch==2.5.1 torchaudio==2.5.1

cmd
