#!/bin/bash

module load IMOD

numFrames=10;

#make the frame mic folders
for (( i=1; i<=$numFrames; i++ )); do
	mkdir frame_$(printf "%02d\n" $i);
done

#split the movies and put each frame in the respective directory
for micrograph in *.mrc; do

	newstack -split 1 $micrograph out.mrc;

	for (( i=1; i<=$numFrames; i++ )); do	    
		mv out.mrc.$(printf "%02d\n" $i) frame_$(printf "%02d\n" $i)/;
		mv frame_$(printf "%02d\n" $i)/out.mrc.$(printf "%02d\n" $i) frame_$(printf "%02d\n" $i)/$micrograph;
	done

done
