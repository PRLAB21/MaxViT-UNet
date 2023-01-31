BASE_PATH="trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting14/fold5"
mkdir $BASE_PATH"/analyze_logs_plots"
PLOTS_BASE_PATH=$BASE_PATH"/analyze_logs_plots"
JSON_LOG_PATH=$BASE_PATH"/20220609_211212.log.json"
SCRIPT="tools/analyze_logs.py"
python3 $SCRIPT $JSON_LOG_PATH --keys loss decode.loss_ce aux.loss_ce --legend loss decode_loss_ce aux_loss_ce --title "Loss Plot" --out $PLOTS_BASE_PATH"/plot_loss.jpg"
python3 $SCRIPT $JSON_LOG_PATH --keys decode.acc_seg aux.acc_seg --legend decode_acc_seg aux_acc_seg --title "Accuracy Plot" --out $PLOTS_BASE_PATH"/plot_acc.jpg"
python3 $SCRIPT $JSON_LOG_PATH --keys mAcc --legend mAcc --title "mAcc Plot" --out $PLOTS_BASE_PATH"/plot_mAcc.jpg"
python3 $SCRIPT $JSON_LOG_PATH --keys mIoU --legend mIoU --title "mIoU Plot" --out $PLOTS_BASE_PATH"/plot_mIoU.jpg"
python3 $SCRIPT $JSON_LOG_PATH --keys mDice --legend mDice --title "mDice Plot" --out $PLOTS_BASE_PATH"/plot_mDice.jpg"
python3 $SCRIPT $JSON_LOG_PATH --keys IoU.background IoU.lymphocyte --legend IoU_background IoU_lymphocyte --title "IoU Per Class Plot" --out $PLOTS_BASE_PATH"/plot_IoU_per_class.jpg"
python3 $SCRIPT $JSON_LOG_PATH --keys Acc.background Acc.lymphocyte --legend Acc_background Acc_lymphocyte --title "Acc Per Class Plot" --out $PLOTS_BASE_PATH"/plot_Dice_per_class.jpg"
python3 $SCRIPT $JSON_LOG_PATH --keys Dice.background Dice.lymphocyte --legend Dice_background Dice_lymphocyte --title "Dice Per Class Plot" --out $PLOTS_BASE_PATH"/plot_Dice_per_class.jpg"
python3 $SCRIPT $JSON_LOG_PATH --keys lr --legend lr --title "LR Plot" --out $PLOTS_BASE_PATH"/plot_lr.jpg"

# python3 $SCRIPT plot_curve $JSON_LOG_PATH --keys lr --legend lr --title "LR Plot" --out $PLOTS_BASE_PATH"/plot_lr.jpg"
# python3 $SCRIPT plot_curve $JSON_LOG_PATH --keys acc --legend acc --title "Accuracy Plot" --out $PLOTS_BASE_PATH"/plot_acc.jpg"
# python3 $SCRIPT plot_curve $JSON_LOG_PATH --keys loss loss_rpn_cls loss_rpn_bbox loss_cls loss_bbox loss_mask --legend loss loss_rpn_cls loss_rpn_bbox loss_cls loss_bbox loss_mask --title "All Loss Plot" --out $PLOTS_BASE_PATH"/plot_all_loss.jpg"
# python3 $SCRIPT plot_curve $JSON_LOG_PATH --keys loss --legend loss --title "Loss Plot" --out $PLOTS_BASE_PATH"/plot_loss.jpg"
# python3 $SCRIPT plot_curve $JSON_LOG_PATH --keys loss_rpn_cls loss_rpn_bbox --legend loss_rpn_cls loss_rpn_bbox --title "RPN Loss Plot" --out $PLOTS_BASE_PATH"/plot_rpn_loss.jpg"
# python3 $SCRIPT plot_curve $JSON_LOG_PATH --keys loss_cls loss_bbox loss_mask --legend loss_cls loss_bbox loss_mask --title "Head Loss Plot" --out $PLOTS_BASE_PATH"/plot_head_loss.jpg"

# python3 tools/analysis_tools/analyze_logs.py plot_curve \
#         "trained_models/lysto-models/maskrcnn-resnet50/setting2/20220411_123634.log.json" \
#         "trained_models/lysto-models/scnet-resnet50/setting1/20220411_220356.log.json" \
#         "trained_models/lysto-models/maskrcnn-lymphocytenet-pvt/setting2/20220410_140450.log-corrected.json" \
#         --title "Loss Curve" \
#         --keys "loss" \
#         --legend "MaskRCNN ResNet50" "SCNet ResNet50" "MaskRCNN LymphoNet (PVT + CBAM)" \
#         --out "trained_models/lysto-models/loss-curve-proposed_pvt_cbam.jpg" \

# python3 tools/analysis_tools/analyze_logs.py plot_curve \
#         "trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting9/20220511_140527.log.json" \
#         --title "Loss Curve" \
#         --keys "loss_rpn_cls" "loss_rpn_bbox" "loss_cls" "loss_bbox" "loss_mask" "loss" \
#         --out "trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting9/plot_curve-losses.jpg" \

# python3 tools/analysis_tools/analyze_logs.py plot_curve \
#         "trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting9/20220511_140527.log.json" \
#         --title "Loss Curve" \
#         --keys "loss" \
#         --out "trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting9/plot_curve-loss.jpg" \

# python3 tools/analysis_tools/analyze_logs.py plot_curve \
#         "trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting9/20220511_140527.log.json" \
#         --title "Accuracy Curve" \
#         --keys "acc" \
#         --out "trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting9/plot_curve-acc.jpg" \

# python3 tools/analysis_tools/analyze_logs.py plot_curve \
#         "trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting9/20220511_140527.log.json" \
#         --title "LR Curve" \
#         --keys "lr" \
#         --out "trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting9/plot_curve-lr.jpg" \
