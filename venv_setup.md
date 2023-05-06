# FULL VENV SETUP:
virtualenv --python C:\Users\Andru\AppData\Local\Programs\Python\Python310\python.exe venv_310_64bit

.\venv_310_64bit\Scripts\activate

pipenv install --skip-lock

pip install tensorboard==1.15.0

pip install protobuf==3.20.*

pip install numpy==1.23.4


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python run_bert.py --train --data_name job_dataset --n_gpu "1" --epochs 3