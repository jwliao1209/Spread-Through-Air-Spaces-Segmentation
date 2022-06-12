#!/bin/bash
python3 STEP7_Inference.py

wait
matlab <  PostProcess.m

exit



