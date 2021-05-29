#!/bin/sh
cmd="ffmpeg -f concat -safe 0"
$cmd -i 0.txt 0.mov &
$cmd -i 1.txt 1.mov &
$cmd -i 2.txt 2.mov &
wait
echo Done!
