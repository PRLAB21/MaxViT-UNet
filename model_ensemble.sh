MMSEG_HOME_PATH='/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/'
MMCLS_HOME_PATH='/home/gpu02/maskrcnn-lymphocyte-detection/mmclassification/'
MMDET_HOME_PATH='/home/gpu02/maskrcnn-lymphocyte-detection/mmdetection/'

python3 tools/model_ensemble2_afterclass.py \
    --config $MMSEG_HOME_PATH"configs/lysto/fcn_unet_s5_s3.py" \
             $MMSEG_HOME_PATH"configs/lysto/seglymphnet_s1.py" \
             $MMSEG_HOME_PATH"configs/lysto/seglymphnet2_s2.py" \
             $MMDET_HOME_PATH"configs/lysto/maskrcnn_resnet50_s1_lysto.py" \
    --checkpoint $MMSEG_HOME_PATH"trained_models/lysto/fcn_unet_s5/setting3/epoch_12.pth" \
                 $MMSEG_HOME_PATH"trained_models/lysto/seglymphnet/setting1/epoch_10.pth" \
                 $MMSEG_HOME_PATH"trained_models/lysto/seglymphnet2/setting2/epoch_12.pth" \
                 $MMDET_HOME_PATH"trained_models/lysto/maskrcnn-resnet50/setting1/epoch_30.pth" \
    --model_name "fcn_unet-seglymphnet-seglymphnet2-160k" \
    --class_confg $MMCLS_HOME_PATH"configs/lysto_lymph_vs_nolymph/lympnet2_s2.py" \
    --class_checkpoint $MMCLS_HOME_PATH"trained_models/lymph_vs_nolymph/lysto_models/lympnet2/setting2-2022-12-05-13-47/epoch_28.pth" \
    --testpath "/home/gpu02/lyon_dataset/lyon_patch_overlap_onboundries"
