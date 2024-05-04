cd /data
rm -rf checkpoints
mkdir checkpoints
cd checkpoints
mkdir t2m
cd t2m 
echo -e "Downloading pretrained models for HumanML3D dataset"
wget https://huggingface.co/nanxiz/zcabnzhmm/resolve/main/humanml3d_models.zip
unzip humanml3d_models.zip
rm humanml3d_models.zip
cd /workspace/MoMask