#!/bin/sh
cd algorithmic-efficiency
git pull

chmod a+x docker/scripts/startup.sh
docker/scripts/startup.sh
