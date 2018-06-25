counter=0
num=3000
while [ $counter -le $num ]
do
    convert "optimized_${counter}.jpg" "zero_motion_${counter}.jpg" -quality 100 +append "out${counter}.jpg"
    ((counter++))
done

ffmpeg -i out%d.jpg -vcodec libx264 -b 1024k a.avi

rm out*
