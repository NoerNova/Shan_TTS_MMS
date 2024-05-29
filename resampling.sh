find ./train_tmp -type f -name "*.wav" -exec sh -c 'for f; do ffmpeg -i "$f" -ar 22050 "./train/$(basename "$f")"; done' sh {} +

find ./train_tmp -type f -name "*.wav" -exec sh -c 'for f; do ffmpeg -i "$f" -ar 22050 -b:a 352k "./train/$(basename "$f")"; done' sh {} +

find ./audio-data/train --type -f -name "*.wav" -exec sh -c 'for f; do deepFilter "$f" -o "./labs"; done' sh {} +

# Get-ChildItem -Path .\audio-data -Recurse -Filter *.wav | ForEach-Object {
#     deepFilter $_.FullName -o .\tmp
#  }
