#!/bin/bash
MOUNTFROM=$PWD
MOUNTTO='/home/morpheus/workspace/mount'
IMAGE='rsbyrne/mobility-aus'
SOCK='/var/run/docker.sock'
docker run -v $MOUNTFROM:$MOUNTTO -v $SOCK:$SOCK -it --shm-size 2g $IMAGE bash
