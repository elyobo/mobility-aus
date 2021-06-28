#!/bin/bash

currentDir=$PWD
cd "$(dirname "$0")"

python3 fbpull.py
bash push.sh

cd $currentDir