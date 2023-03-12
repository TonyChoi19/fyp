#! bash

# Rain100H
python train_PReNet_3.py --preprocess True --save_path logs/Rain100H/PReNet_3 --data_path datasets/train/RainTrainH

# Rain100L
python train_PReNet_3.py --preprocess True --save_path logs/Rain100L/PReNet_3 --data_path datasets/train/RainTrainL

# Rain12600
python train_PReNet_3.py --preprocess True --save_path logs/Rain1400/PReNet_3 --data_path datasets/train/Rain12600
