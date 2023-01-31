import os
import shutil
opj = os.path.join

def copyfiles1():
    # des_fol_train = r'/home/zunaira/lyon_dataset/combined_dataset/Training'
    # des_fol_val = r'/home/zunaira/lyon_dataset/combined_dataset/Validation'
    des_fol_train = r'/home/zunaira/lyon_dataset/combined_dataset/Training_masks'
    des_fol_val = r'/home/zunaira/lyon_dataset/combined_dataset/Validation_masks'
    # src_fol_train = r'/home/zunaira/lyon_dataset/lyon_dataset_small/Train'
    # src_fol_val = r'/home/zunaira/lyon_dataset/lyon_dataset_small/Validation'
    src_fol_train = r'/home/zunaira/lyon_dataset/lyon_dataset_small/Train_binary'
    src_fol_val = r'/home/zunaira/lyon_dataset/lyon_dataset_small/Validation_binary'
    dab= r'/home/zunaira/lyon_dataset/full_DAB'
    hsv = r'/home/zunaira/lyon_dataset/full_nuclick_HSV'
    train_files = os.listdir(src_fol_train)
    val_files = os.listdir(src_fol_val)

    print('train_files:', len(train_files))
    print('val_files:', len(val_files))

    for f in train_files:
        shutil.copy(opj(src_fol_train, f), opj(des_fol_train, f.split('.')[0]+'-dab'+'.png'))
        shutil.copy(opj(src_fol_train, f), opj(des_fol_train, f.split('.')[0]+'-hsv'+'.png'))
        shutil.copy(opj(src_fol_train, f), opj(des_fol_train, f.split('.')[0]+'-ihc'+'.png'))
        
    for f in val_files:
        shutil.copy(opj(src_fol_val, f), opj(des_fol_val, f.split('.')[0]+'-dab'+'.png'))
        shutil.copy(opj(src_fol_val, f), opj(des_fol_val, f.split('.')[0]+'-hsv'+'.png'))
        shutil.copy(opj(src_fol_val, f), opj(des_fol_val, f.split('.')[0]+'-ihc'+'.png'))

    print('combined_train_files:', len(os.listdir(des_fol_train)))
    print('combined_val_files:', len(os.listdir(des_fol_val)))

def copyfiles2():
    des = r'/home/zunaira/lyon_dataset/lyon_dataset_small/Train_DAB'
    desv = r'/home/zunaira/lyon_dataset/lyon_dataset_small/Validation_DAB'
    HSV_src  = r'/home/zunaira/lyon_dataset/full_DAB'
    train_files = os.listdir(r'/home/zunaira/lyon_dataset/lyon_dataset_small/Train')
    val_files = os.listdir(r'/home/zunaira/lyon_dataset/lyon_dataset_small/Validation')

    for f in train_files:
        shutil.copy(opj(HSV_src,f), opj(des,f))

    for f in val_files:
        shutil.copy(opj(HSV_src,f), opj(desv, f))
