python -m venv venv
call venv/scripts/activate

pip install -r .\requirements.txt
pip uninstall -y torch torchaudio
pip install torch==2.5.1 torchaudio==2.5.1

cmd
