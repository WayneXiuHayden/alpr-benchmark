#!/bin/bash

time alpr ../data/720p.mp4 > /dev/null

# gpu
time alpr -g ../data/720p.mp4 > /dev/null