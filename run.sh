#!/bin/bash
currentDir=$PWD
cd "$(dirname "$0")"
bash imdo.sh "bash cycle.sh"
cd $currentDir
sudo docker rm $(docker ps -a -q)
