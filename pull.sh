#!/bin/bash
MOUNTFROM=$PWD
MOUNTTO='/home/morpheus/workspace/mount'
IMAGE='rsbyrne/mobility-aus'
SOCK='/var/run/docker.sock'
sudo chmod -R 777 'data'
sudo rm -f geckodriver.log
docker run -v $MOUNTFROM:$MOUNTTO -v $SOCK:$SOCK --shm-size 2g $IMAGE python3 '/home/morpheus/workspace/mount/pull.py'
sudo chmod -R 777 'data'
