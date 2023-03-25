
:: Rain100H
python train_PRN_dense.py --preprocess True --save_path logs/Rain100H/PRN_dense --data_path datasets/train/RainTrainH
::pause

:: Rain100L
python train_PRN_dense.py --preprocess True --save_path logs/Rain100L/PRN_dense --data_path datasets/train/RainTrainL
::pause
:: Rain12600
python train_PRN_dense.py --preprocess True --save_path logs/Rain1400/PRN_dense --data_path datasets/train/Rain12600

::pause