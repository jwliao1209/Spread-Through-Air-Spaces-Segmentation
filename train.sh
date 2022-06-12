#!/bin/bash

python3 STEP4_Train.py --config Fold0.yaml

wait
python3 STEP4_Train.py --config Fold1.yaml

wait
python3 STEP4_Train.py --config Fold2.yaml

wait
python3 STEP4_Train.py --config Fold3.yaml
