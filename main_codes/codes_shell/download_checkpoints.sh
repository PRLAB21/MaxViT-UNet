MMSEG_HOME_PATH='/home/gpu02/maskrcnn-lymphocyte-detection/mmclassification'

# resnet50
# wget 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth' \
#     -O $MMSEG_HOME_PATH'/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth'

# resnet50
wget 'https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w26-s8_3rdparty_8xb32_in1k_20210927-f547a94b.pth' \
    -O $MMSEG_HOME_PATH'/checkpoints/res2net50-w26-s8_3rdparty_8xb32_in1k_20210927-f547a94b.pth'
