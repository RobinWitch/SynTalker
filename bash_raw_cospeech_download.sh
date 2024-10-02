mkdir -p datasets/BEAT_SMPL
cd datasets/BEAT_SMPL
gdown https://drive.google.com/uc?id=1_iXr0XiT_EdslXe4b0HwDr2OoOCrtlrB
unzip beat_v2.0.0.zip 
cd ../../
gdown https://drive.google.com/drive/folders/1tGTB40jF7v0RBXYU-VGRDsDOZp__Gd0_?usp=drive_link -O ./test --folder
gdown https://drive.google.com/drive/folders/1MCks7CMNBtAzU2XihYezNmiGT_6pWex8?usp=drive_link -O ./datasets/hub --folder