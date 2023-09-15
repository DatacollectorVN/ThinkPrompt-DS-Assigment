ThinkPrompt DS Assigment

# 1. EDA 
Please see in `eda/EDA.ipynb`

# 2. Methodology of processing
- Drop NaN
- Oversampling --> solve imbalance classes
- Feature engineering --> create new columns (Month and hour) from date columns
- MinMaxScaler
- Standard Scaler

# 3. Methodology of training models
- GridSearchCV

# 4. Setting up

- Create and activate conda env:
```bash
conda create -n ThinkPrompt python=3.8 -y
conda activate ThinkPrompt
```

- Install dependent packages:
```bash
pip install -r requirements.txt
```

# 5. Training model
Config basic training in `config/train.yaml`. If you want to config model's parameters that is used in GridSerach, edit in `src/models.py`

*Start training*
```bash
python train.py
```

# 6. Evaluating
Config basic evaluation in `config/evaluate.yaml`

*Start evaluating*
```bash
python eval.py
```
