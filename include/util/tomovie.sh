#!/bin/bash

counter=0
num=75

while [ $counter -le $num ]
do
    convert "optimized_${counter}.jpg" "zero_motion_${counter}.jpg" +append "out${counter}"
    ((counter++))
done

output=""
counter=0
while [ $counter -le $num ]
do
    output+="out${counter} "
    ((counter++))
done

convert -set delay 50 $output video.mpeg
rm out*
