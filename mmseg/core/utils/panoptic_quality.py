import numpy as np

# Compute Panoptic quality metric for each image
def panoptic_quality(ground_truth_image, predicted_image, iou_thresh=0.5):
    TAG = '[panoptic_quality]'
    eta = 1e-10
    TP = 0
    FP = 0
    FN = 0
    sum_IOU = 0
    matched_instances = {}  # Create a dictionary to save ground truth indices in keys and 
                            # predicted matched instances as values. It will also save IOU 
                            # of the matched instance in [indx][1]

    # Find matched instances and save it in a dictionary
    for i in np.unique(ground_truth_image):
        if i == 0:
            pass
        else:
            temp_image = np.array(ground_truth_image)
            temp_image = temp_image == i
            matched_image = temp_image * predicted_image
        
            for j in np.unique(matched_image):
                if j == 0:
                    pass
                else:
                    pred_temp = predicted_image == j
                    intersection = sum(sum(temp_image * pred_temp))
                    union = sum(sum(temp_image + pred_temp))
                    IOU = intersection / union
                    if IOU > iou_thresh:
                        matched_instances[i] = j, IOU

    # Compute TP, FP, FN and sum of IOU of the matched instances to compute Panoptic Quality
    pred_indx_list = np.unique(predicted_image) # Find all predicted instances
    pred_indx_list = np.array(pred_indx_list[1:]) # Remove 0 from the predicted instances

    # Loop on ground truth instances
    for indx in np.unique(ground_truth_image):
        if indx == 0:
            pass
        else:
            if indx in matched_instances.keys():
                pred_indx_list = np.delete(pred_indx_list, np.argwhere(pred_indx_list == matched_instances[indx][0]))
                TP = TP + 1
                sum_IOU = sum_IOU + matched_instances[indx][1]
            else:
                FN = FN + 1

    FP = len(np.unique(pred_indx_list))
    PQ = sum_IOU / (TP + 0.5 * FP + 0.5 * FN + eta)
    # print(TAG, f'TP={TP}, FN={FN}, FP={FP}, PQ={PQ}, sum_IOU={sum_IOU}, \
    #       <denominator>={TP + 0.5 * FP + 0.5 * FN + eta}')

    return PQ
