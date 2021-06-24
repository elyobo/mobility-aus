#!/bin/bash
MOUNTFROM=$PWD
MOUNTTO='/home/morpheus/workspace/mount'
IMAGE='rsbyrne/mobility-aus'
SOCK='/var/run/docker.sock'
sudo chmod -R 777 'products'
sudo chmod -R 777 'resources'
docker run -v $MOUNTFROM:$MOUNTTO -v $SOCK:$SOCK --shm-size 2g $IMAGE python3 '/home/morpheus/workspace/mount/update.py'
sudo chmod -R 777 'products'
sudo chmod -R 777 'resources'