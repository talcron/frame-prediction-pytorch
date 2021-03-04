for i in $(find "$1" -name "*.avi"); do
  ffmpeg -i "$i" -c:v mjpeg -vf scale=64:64 "${i%.*}.mjpeg" -y;
done

find "$1" -name "*.mjpeg" > mjpeg-index.txt;
