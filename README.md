# TTS MMS Finetune

- Text to Speech
- Auto Speech Recognition

## Finetune instructions

- https://github.com/ylacombe/finetune-hf-vits
- https://huggingface.co/blog/mms_adapters

## Audio Deepfilter

Get-ChildItem -Path .\audio-data -Recurse -Filter *.wav | ForEach-Object { deepFilter $_.FullName -o .\tmp }
