#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Install FFmpeg on the Render instance
apt-get -y update
apt-get install -y ffmpeg