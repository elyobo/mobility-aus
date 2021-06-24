#!/bin/bash
currentDir=$PWD
cd "$(dirname "$0")"
#sh push.sh
docker build -t rsbyrne/mobility-portal:latest .
docker push rsbyrne/mobility-portal:latest
cd $currentDir
