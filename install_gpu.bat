python -m venv venv
call venv/scripts/activate

.\venv\Scripts\python.exe -m pip install --upgrade pip
pip uninstall torch torchaudio
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r .\requirements.txt

cmd
