# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh "./configs/mask_rcnn_lymphocyte/maskrcnn_lymphocytenet3_cm1_s4_lysto.py" 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh "./configs/mask_rcnn_lymphocyte/maskrcnn_resnet50_s1_lysto.py" 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh "./configs/mask_rcnn_lymphocyte/maskrcnn_resnet_cbam50_s1_lysto.py" 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh "./configs/mask_rcnn_lymphocyte/maskrcnn_resnet50_dilation1223_s1_lysto.py" 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh "./configs/mask_rcnn_lymphocyte/maskrcnn_resnext50_s1_lysto.py" 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh "./configs/mask_rcnn_lymphocyte/cascade_maskrcnn_resnet50_s1_lysto.py" 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=39500 ./tools/dist_train.sh "./configs/mask_rcnn_lymphocyte/maskrcnn_lymphocytenet3_cm1_s5_lysto.py" 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=39500 ./tools/dist_train.sh "./configs/mask_rcnn_lymphocyte/maskrcnn_lymphocytenet3_cm1_s6_lysto.py" 4
# CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_train.sh "./configs/mask_rcnn_lyon/maskrcnn_lymphocytenet3_cm1_s6_lyon.py" 1
# CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_train.sh "./configs/lyon/maskrcnn_attention_3x3_s1_lyon.py" 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=39500 ./tools/dist_train.sh "./configs/mask_rcnn_lymphocyte/maskrcnn_lymphocytenet3_cm1_s7_lysto.py" 4
# CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh "./configs/lyon/retinanet_resnet50_s1_lyon.py" 2
# CUDA_VISIBLE_DEVICES=2,3 PORT=29500 ./tools/dist_train.sh "./configs/lyon/fasterrcnn_resnet50_s1_lyon.py" 2

# CUDA_VISIBLE_DEVICES=0 PORT=39500 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s10_lysto.py" 1 # fold1
# CUDA_VISIBLE_DEVICES=1 PORT=39501 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s10_lysto.py" 1 # fold2
# CUDA_VISIBLE_DEVICES=2 PORT=39502 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s10_lysto.py" 1 # fold3
# CUDA_VISIBLE_DEVICES=3 PORT=39503 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s10_lysto.py" 1 # fold4
# CUDA_VISIBLE_DEVICES=0 PORT=39504 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s10_lysto.py" 1 # fold5
# CUDA_VISIBLE_DEVICES=1 PORT=39505 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s10_lysto.py" 1 # fold6
# CUDA_VISIBLE_DEVICES=2 PORT=39506 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s10_lysto.py" 1 # fold7
# CUDA_VISIBLE_DEVICES=3 PORT=39507 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s10_lysto.py" 1 # fold8
# CUDA_VISIBLE_DEVICES=0 PORT=39508 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s10_lysto.py" 1 # fold9
# CUDA_VISIBLE_DEVICES=1 PORT=39509 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s10_lysto.py" 1 # fold10

# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=39500 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s11_lysto.py" 4 # fold1
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=39501 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s11_lysto.py" 4 # fold2
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=39502 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s11_lysto.py" 4 # fold3
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=39503 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s11_lysto.py" 4 # fold4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=39504 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s11_lysto.py" 4 # fold5
# CUDA_VISIBLE_DEVICES=3 PORT=39505 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s11_lysto_combined.py" 1 # combined

# CUDA_VISIBLE_DEVICES=0 PORT=39500 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s12_lysto.py" 1 # fold1
# CUDA_VISIBLE_DEVICES=1 PORT=39501 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s12_lysto.py" 1 # fold2
# CUDA_VISIBLE_DEVICES=2 PORT=39502 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s12_lysto.py" 1 # fold3
# CUDA_VISIBLE_DEVICES=3 PORT=39503 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s12_lysto.py" 1 # fold4
# CUDA_VISIBLE_DEVICES=3 PORT=39504 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s12_lysto.py" 1 # fold5
# CUDA_VISIBLE_DEVICES=2 PORT=39506 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s12_lysto_combined.py" 1 # combined

# CUDA_VISIBLE_DEVICES=0 PORT=39500 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s13_lysto.py" 1 # fold1
# CUDA_VISIBLE_DEVICES=1 PORT=39501 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s13_lysto.py" 1 # fold2
# CUDA_VISIBLE_DEVICES=2 PORT=39502 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s13_lysto.py" 1 # fold3
# CUDA_VISIBLE_DEVICES=3 PORT=39503 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s13_lysto.py" 1 # fold4
# CUDA_VISIBLE_DEVICES=2 PORT=39504 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s13_lysto.py" 1 # fold5
# CUDA_VISIBLE_DEVICES=1 PORT=39507 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s13_lysto_combined.py" 1 # combined

# CUDA_VISIBLE_DEVICES=0 PORT=39500 ./tools/dist_train.sh "./configs/lyon/maskrcnn_lymphocytenet3_cm1_s13_lyon.py" 1
# CUDA_VISIBLE_DEVICES=0 PORT=39500 ./tools/dist_train.sh "./configs/lyon/maskrcnn_lymphocytenet4_cm2_s14_lyon.py" 1
CUDA_VISIBLE_DEVICES=0 PORT=39500 ./tools/dist_train.sh "./configs/lyon/maskrcnn_lymphocytenet5_cm4_s13_lyon.py" 1

# CUDA_VISIBLE_DEVICES=0 PORT=39500 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s14_lysto.py" 1 # fold1
# CUDA_VISIBLE_DEVICES=1 PORT=39501 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s14_lysto.py" 1 # fold2
# CUDA_VISIBLE_DEVICES=2 PORT=39502 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s14_lysto.py" 1 # fold3
# CUDA_VISIBLE_DEVICES=3 PORT=39503 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s14_lysto.py" 1 # fold4
# CUDA_VISIBLE_DEVICES=1 PORT=39504 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s14_lysto.py" 1 # fold5
# CUDA_VISIBLE_DEVICES=0 PORT=39508 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s14_lysto_combined.py" 1 # combined

# CUDA_VISIBLE_DEVICES=0 PORT=39500 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s15_lysto.py" 1 # fold1
# CUDA_VISIBLE_DEVICES=1 PORT=39501 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s15_lysto.py" 1 # fold2
# CUDA_VISIBLE_DEVICES=2 PORT=39502 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s15_lysto.py" 1 # fold3
# CUDA_VISIBLE_DEVICES=3 PORT=39503 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s15_lysto.py" 1 # fold4
# CUDA_VISIBLE_DEVICES=0 PORT=39504 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s15_lysto.py" 1 # fold5
# CUDA_VISIBLE_DEVICES=3 PORT=39509 ./tools/dist_train.sh "./configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s15_lysto_combined.py" 1 # combined
