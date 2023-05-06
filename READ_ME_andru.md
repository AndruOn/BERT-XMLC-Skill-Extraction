# how to create a Python Env ith all dependencies 
THEIR README HAS FUCKED THE SETUP:

Transfer all files from checkpoint folder (in google drive) to pybert/pretrain/bert/bert-uncased folder

NOT CORRECT: pybert/pretrain/bert/base-uncased folder

## venv
virtualenv SE_virtualenv -p python3.7
pip install scikit-learn==0.21.3 matplotlib==3.1.1 tensorboard==1.15.0
pip install pytorch-transformers==1.1.0 #bc this crashes: pytorch-transformers==1.2.0
if needed: pip install protobuf==3.20.*
## run 
source SE_virtualenv/bin/activate
python run_bert.py --train --data_name job_dataset



## conda sur linux
### todo
conda create --name SE_oneline python=3.7 scikit-learn==0.21.3 matplotlib==3.1.1 tensorboard==1.15.0
#remove bc crashing "pytorch-transformers==1.2.0"
conda install -c vikigenius pytorch-transformers #https://anaconda.org/search?q=pytorch-transformers #1.1.0 != 1.2.0

### result
CRASH has been resolved

[create features] 2030/2030 [==============================] 3.0ms/stepSaving features into cached file pybert/dataset/cached_valid_features_256_bert
initializing model
Model name 'pybert/pretrain/bert/base-uncased' was not found in model name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). We assumed 'pybert/pretrain/bert/base-uncased/config.json' was a path or url but couldn't find any file associated to this path or url.
Traceback (most recent call last):
  File "run_bert.py", line 248, in <module>
    main()
  File "run_bert.py", line 241, in main
    run_train(args)
  File "run_bert.py", line 76, in run_train
    model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list))
  File "/home/andruon/anaconda3/envs/SE_oneline/lib/python3.7/site-packages/pytorch_transformers/modeling_utils.py", line 430, in from_pretrained
    **kwargs
TypeError: cannot unpack non-iterable NoneType object



# windows
# venv
python3 -m venv env_name
pip install scikit-learn==0.21.3 matplotlib==3.1.1 tensorboard==1.15.0

pip install pytorch-transformers==1.1.0 #bc this crashes: pytorch-transformers==1.2.0
if needed: pip install protobuf==3.20.*


# Windows wsl


# widnows 10
python3.10
TODO: remove line torch from pipflie
pipenv install
pip install tensorboard==1.15.0


# widnows, CUda pip3 line from pytorch
´´´
virtualenv --python C:\Users\Andru\AppData\Local\Programs\Python\Python310\python.exe my_env_3_10_64b
.\my_env_3_10_64b\Scripts\activate
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
#####pip install -r requirements.txt #crashed for matplot and scikitlearn
pip install scikit-learn
pip install pytorch-transformers
pip install matplotlib
pip install tensorboard
´´´
ERROR:
 python run_bert.py --train --data_name job_dataset
Training/evaluation parameters Namespace(arch='bert', do_data=False, train=True, test=False, save_best=False, do_lower_case=False, data_name='job_dataset', epochs=10, resume_path='', test_path='', mode='min', monitor='valid_loss', valid_size=0.05, local_rank=-1, sorted=1, n_gpu='0', gradient_accumulation_steps=1, train_batch_size=4, eval_batch_size=4, train_max_seq_len=256, eval_max_seq_len=256, loss_scale=0, warmup_proportion=0.1, weight_decay=0.01, adam_epsilon=1e-08, grad_clip=1.0, learning_rate=0.0001, seed=42, fp16=False, fp16_opt_level='O1', predict_labels=False, predict_idx='0')
Traceback (most recent call last):
  File "C:\Users\Andru\OneDrive\Master Thesis\BERT-XMLC-Skill-Extraction\run_bert.py", line 248, in <module>
    main()
  File "C:\Users\Andru\OneDrive\Master Thesis\BERT-XMLC-Skill-Extraction\run_bert.py", line 241, in main
    run_train(args)
  File "C:\Users\Andru\OneDrive\Master Thesis\BERT-XMLC-Skill-Extraction\run_bert.py", line 37, in run_train
    train_examples = processor.create_examples(lines=train_data,
  File "C:\Users\Andru\OneDrive\Master Thesis\BERT-XMLC-Skill-Extraction\pybert\io\bert_processor.py", line 115, in create_examples
    label = [np.float(x) for x in list(label)]
  File "C:\Users\Andru\OneDrive\Master Thesis\BERT-XMLC-Skill-Extraction\pybert\io\bert_processor.py", line 115, in <listcomp>
    label = [np.float(x) for x in list(label)]
  File "C:\Users\Andru\OneDrive\Master Thesis\BERT-XMLC-Skill-Extraction\my_env_3_10_64b\lib\site-packages\numpy\__init__.py", line 305, in __getattr__
    raise AttributeError(__former_attrs__[attr])
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations. Did you mean: 'cfloat'?