- Read text from image
  python read.py outputs/parseq/2022-09-28_23-03-26/checkpoints/last.ckpt --images demo_images/*
- Train from checkpoint
  python train.py ckpt_path=/home/bap/BAP/ocr/TextRecognition/parseq/outputs/parseq/2022-09-28_23-03-26/checkpoints/last.ckpt
