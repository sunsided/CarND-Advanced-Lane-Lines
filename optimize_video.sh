#!/usr/bin/env bash

INPUT=$1
OUTPUT=$2
 
if [ -z "$INPUT" ]; then
    echo "Specify an input video file."
    echo "$0 input.mp4 output.mp4"
    exit 1
fi

if [ -z "$OUTPUT" ]; then
    echo "Specify an output video file."
    echo "$0 input.mp4 output.mp4"
    exit 1
fi

ffmpeg -i "$INPUT" -vcodec libx265 -crf 20 "$OUTPUT"