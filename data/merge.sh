#!/bin/sh
ffmpeg -i 0.mov -i 1.mov -i 2.mov -filter_complex hstack=inputs=3 merged-uncropped.mov
echo Done!
