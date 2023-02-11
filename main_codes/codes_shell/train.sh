MMSEG_HOME_PATH='/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation'

# python3 "tools/train.py" "configs/lysto/fcn_d6_resnet50_s1_lysto.py"
# python3 "tools/train.py" "configs/unet/fcn_unet_s5-d16_256x256_40k_hrf.py"
# python3 "tools/train.py" "configs/lysto/fcn_unet_s5_s1_lysto.py"
# python3 $MMSEG_HOME_PATH"/tools/train.py" $MMSEG_HOME_PATH"/configs/lysto_hard/fcn_unet_s5_s1.py"
# python3 $MMSEG_HOME_PATH"/tools/train.py" $MMSEG_HOME_PATH"/configs/lysto_hard/fcn_unet_s5_s2.py"
# python3 $MMSEG_HOME_PATH"/tools/train.py" $MMSEG_HOME_PATH"/configs/lysto/fcn_unet_s5_s2.py"
# python3 $MMSEG_HOME_PATH"/tools/train.py" $MMSEG_HOME_PATH"/configs/lysto/fcn_unet_s5_s3.py"
# python3 $MMSEG_HOME_PATH"/tools/train.py" $MMSEG_HOME_PATH"/configs/lysto/efficient_unet_b5_s1.py"
# python3 $MMSEG_HOME_PATH"/tools/train.py" $MMSEG_HOME_PATH"/configs/lysto/efficient_unet_b5_s2.py"
python3 $MMSEG_HOME_PATH"/tools/train.py" $MMSEG_HOME_PATH"/configs/lysto/mobilenet_v3_s1.py"

# python3 $MMSEG_HOME_PATH"/tools/train.py" $MMSEG_HOME_PATH"/configs/lysto/seglymphnet2_s2.py"
# python3 $MMSEG_HOME_PATH"/tools/kfold-cross-valid.py" $MMSEG_HOME_PATH"/configs/lysto_lymph_vs_nolymph/resnet50_s1.py" --num-splits 5
