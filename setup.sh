#!/bin/bash

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
pip install -v -e .
pip install -r requirements/albu.txt
pip install -r requirements.txt
