#!/bin/bash

python3 STEP7_Inference.py
wait
matlab <  STEP8_PostProcess.m
exit
