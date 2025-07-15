python -m venv venv
call venv/scripts/activate

pip install -r .\requirements.txt
pip uninstall torch torchaudio
pip3 install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

cmd
