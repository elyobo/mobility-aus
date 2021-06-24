#!/bin/bash
currentDir=$PWD
cd "$(dirname "$0")"
sh push.sh
docker build -t rsbyrne/mobility-aus:latest .
docker push rsbyrne/mobility-aus:latest
cd $currentDir
