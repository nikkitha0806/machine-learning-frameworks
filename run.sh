export TRAINING_DATA=data/train_folds.csv
export TEST_DATA=data/test.csv
export MODEL=$1

#below lines are commented since training is already completed:
#FOLD=0 python -m app.src.train
#FOLD=1 python -m app.src.train
#FOLD=2 python -m app.src.train
#FOLD=3 python -m app.src.train
#FOLD=4 python -m app.src.train
python -m app.src.predict
