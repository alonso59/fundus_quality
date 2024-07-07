import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from eyepy.core.base import Oct
import pandas as pd
from sklearn.model_selection import train_test_split


def vol_files(name):
    vol_filename = os.path.join('dataset/2_OCTAnnotated', name + '.vol')
    oct_read = Oct.from_heyex_vol(vol_filename)
    return oct_read


def get_filenames(path, ext):
    X0 = []
    for i in sorted(os.listdir(path)):
        if i.endswith(ext):
            X0.append(os.path.join(path, i))
    return X0


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_annotations(oct_read):
    ILM = np.round(
        oct_read.bscans[0].annotation["layers"]['ILM']).astype('uint16')
    try:
        RNFL = np.round(
            oct_read.bscans[0].annotation["layers"]['RNFL']).astype('uint16')
    except:
        RNFL = np.zeros((ILM.shape[0])).astype('uint16')
        pass
    try:
        GCL = np.round(
            oct_read.bscans[0].annotation["layers"]['GCL']).astype('uint16')
    except:
        GCL = np.zeros((ILM.shape[0])).astype('uint16')
        pass
    try:
        IPL = np.round(
            oct_read.bscans[0].annotation["layers"]['IPL']).astype('uint16')
    except:
        IPL = np.zeros((ILM.shape[0])).astype('uint16')
        pass
    # ## OPL ##
    OPL = np.round(
        oct_read.bscans[0].annotation["layers"]['OPL']).astype('uint16')
    INL = np.round(
        oct_read.bscans[0].annotation["layers"]['INL']).astype('uint16')
    # ## EZ ##
    PR2 = np.round(
        oct_read.bscans[0].annotation["layers"]['PR2']).astype('uint16')
    PR1 = np.round(
        oct_read.bscans[0].annotation["layers"]['PR1']).astype('uint16')
    # ## BM ##
    try:
        BM = np.round(
            oct_read.bscans[0].annotation["layers"]['BM']).astype('uint16')
    except:
        BM = np.zeros((PR1.shape[0])).astype('uint16')
        pass
    # ## ELM ##
    try:
        ELM = np.round(
            oct_read.bscans[0].annotation["layers"]['ELM']).astype('uint16')
    except:
        ELM = np.zeros((PR1.shape[0])).astype('uint16')
        pass
    return OPL, INL, PR2, PR1, BM, ELM, ILM, RNFL, GCL, IPL


def get_images_masks(file):
    name = os.path.splitext(os.path.split(file)[1])[0]

    oct_read = vol_files(name)

    data = oct_read.bscans[0].scan
    data = np.expand_dims(data, axis=-1)

    zeros = np.zeros((data.shape[0], data.shape[1], 3)).astype('uint8')
    annotation = np.add(data, zeros)
    OPL, INL, PR2, PR1, BM, ELM, ILM, RNFL, GCL, IPL = get_annotations(oct_read)

    # Generate ground truth
    mask = np.zeros((data.shape[0], data.shape[1])).astype('uint8')
    for i in range(OPL.shape[0]):
        annotation[INL[i], i,  :] = [0, 255, 0]
        annotation[OPL[i], i,  :] = [0, 255, 0]
        annotation[ELM[i], i, :] = [0, 255, 0]
        annotation[PR1[i], i, :] = [0, 255, 0]
        annotation[PR2[i], i, :] = [0, 255, 0]
        annotation[BM[i], i, :] = [0, 255, 0]
        # annotation[ILM[i], i, :] = [255, 0, 200]  
        # annotation[RNFL[i], i, :] = [255, 0, 200] 
        # annotation[GCL[i], i, :] = [255, 120, 152] 
        # annotation[IPL[i], i, :] = [152, 120, 255] 

        # # ILM-RNFL
        # mask[ILM[i]:RNFL[i], i] = 5 if ILM[i] <= RNFL[i] and ILM[i] > 0 and RNFL[i] > 0 else mask[ILM[i]:RNFL[i], i]
        # # GCL
        # mask[RNFL[i]:GCL[i], i] = 6 if RNFL[i] <= GCL[i] and RNFL[i] > 0 and GCL[i] > 0 else mask[RNFL[i]:GCL[i], i]
        # # IPL
        # mask[GCL[i]:IPL[i], i] = 7 if GCL[i] <= IPL[i] and GCL[i] > 0 and IPL[i] > 0 else mask[GCL[i]:IPL[i], i]
        # # INL
        # mask[IPL[i]:INL[i], i] = 8 if IPL[i] <= INL[i] and IPL[i] > 0 and INL[i] > 0 else mask[IPL[i]:INL[i], i]
        # OPL
        mask[INL[i]:OPL[i], i] = 2 if INL[i] <= OPL[i] and INL[i] > 0 and OPL[i] > 0 else mask[INL[i]:OPL[i], i]
        # ONL
        # mask[OPL[i]:PR1[i], i] = 9 if OPL[i] <= PR1[i] and OPL[i] > 0 and ELM[i] > 0 else mask[OPL[i]:PR1[i], i]
        # ELM
        mask[ELM[i]:PR1[i], i] = 3 if ELM[i] <= PR1[i] and ELM[i] > 0 and PR1[i] > 0 else mask[ELM[i]:PR1[i], i]
        # EZ
        mask[PR1[i]:PR2[i], i] = 1 if PR1[i] <= PR2[i] and PR1[i] > 0 and PR2[i] > 0 else mask[PR1[i]:PR2[i], i]
        # BM
        mask[PR2[i]:BM[i], i] = 4 if PR2[i] <= BM[i] and PR2[i] > 0 and BM[i] > 0 else mask[PR2[i]:BM[i], i]

    return mask, annotation


def crop_overlap(file, image, mask, path_img, path_msk, patient_id, size=128, shift=64):
    """
    file: OCT file extention .vol
    image: Numpy array
    mask: Numpy array
    path_img: path to save patches
    path_msk: path to save patches
    size: image size
    """
    name = os.path.splitext(os.path.split(file)[1])[0]
    oct_read = vol_files(name)
    OPL, INL, PR2, PR1, BM, ELM, ILM, RNFL, GCL, IPL = get_annotations(oct_read)
    j = 1
    k = 0
    for i in range((size), (image.shape[1]), shift):
        min_pixel = np.max(ILM[k:i])
        max_pixel = np.max(BM[k:i])
        if min_pixel != 0 and max_pixel != 0 and max_pixel > min_pixel:
            delta1 = max_pixel - min_pixel
            delta2 = size - delta1
            delta3 = delta2 // 2
            delta4 = min_pixel - delta3
            delta5 = max_pixel + delta3
            if delta2 % 2 != 0:
                delta5 += 1
            if delta4 < 0:
                delta4 = 0
                delta5 = size
            if delta5 > image.shape[0]:
                delta5 = image.shape[0]
                delta4 = delta5 - size
            img_save = image[delta4:delta5, i - size:i]
            msk_save = mask[delta4:delta5, i - size:i]
            img = Image.fromarray(img_save)
            msk = Image.fromarray(msk_save)
            img.save(path_img + '_' + patient_id + f"_{j}.png")
            msk.save(path_msk + '_' + patient_id + f"_{j}.png")
            j += 1
        k = i


def slicing(file, image, mask, path_img, path_msk, patient_id, size=128, shift=64):
    name = os.path.splitext(os.path.split(file)[1])[0]
    oct_read = vol_files(name)
    j = 1
    for i in range((size), (image.shape[1]), shift):
        img_save = image[:, i - size:i]
        msk_save = mask[:, i - size:i]
        img = Image.fromarray(img_save)
        msk = Image.fromarray(msk_save)
        img.save(path_img + '_' + patient_id + f"_{j}.png")
        msk.save(path_msk + '_' + patient_id + f"_{j}.png")
        j += 1

def main():
    base_path = 'dataset1/dataset_rpe65/'
    train_path_images = base_path + 'train/Images/'
    train_path_masks = base_path + 'train/Masks/'
    val_path_images = base_path + 'val/Images/'
    val_path_masks = base_path + 'val/Masks/'
    train_path_ann = base_path + 'train/Annotations/'
    val_path_ann = base_path + 'val/Annotations/'
    patches_images_train = base_path + 'train/images_slices/'
    patches_masks_train = base_path + 'train/masks_slices/'

    patches_images_val = base_path + 'val/images_slices/'
    patches_masks_val = base_path + 'val/masks_slices/'

    create_dir(train_path_images)
    create_dir(train_path_masks)
    create_dir(val_path_images)
    create_dir(val_path_masks)
    create_dir(patches_images_train)
    create_dir(patches_images_val)
    create_dir(patches_masks_train)
    create_dir(patches_masks_val)
    create_dir(train_path_ann)
    create_dir(val_path_ann)
    # id_train = [
    #     'IRD_RPE65_19', 'IRD_RPE65_11', 'IRD_RPE65_17', '52458', '52065', 'IRD_RPE65_15', '48731', '35281', '30518',
    #     '15313', '49885', '49883', 'IRD_RPE65_04', '52374', '20952', 'IRD_RPE65_03', '23028', 'IRD_RPE65_08', '49031',
    #     '35328', '35280', '28008', '52025', '52037', 'IRD_RPE65_21', '51939', '34571', '16831', 'IRD_RPE65_10', '40300',
    #     '49699', '35277', '35897', '51021', '44872', 'IRD_RPE65_12', '49759', '29331', '52051', '52372', 'IRD_RPE65_01',
    #     '26472', 'IRD_RPE65_13', '51812', 'IRD_RPE65_22', '35821', '35282',  'IRD_RPE65_18', 'IRD_RPE65_05',
    #     '51886', '52113', 'IRD_RPE65_20', 'IRD_RPE65_16', 'IRD_RPE65_07', '16834', 'IRD_RPE65_06', '49919', '43781',
    #     '48782', '52386', '10162', 'IRD_RPE65_23', 'IRD_RPE65_02', 'IRD_RPE65_09',
    # ]

    # id_val = ['42979', '52009', '8837', '33996', '51904', '51038', '21509', '51155', '49905',
    #           '45954', '52484', '51208', '6627', '51013', '51870', '48104', ]

    id_train = [
        'IRD_RPE65_19', 'IRD_RPE65_11', 'IRD_RPE65_15', 
        'IRD_RPE65_04', 'IRD_RPE65_03', 'IRD_RPE65_08', 'IRD_RPE65_21', 
        'IRD_RPE65_10', 'IRD_RPE65_05', 'IRD_RPE65_17', 
        'IRD_RPE65_13', 'IRD_RPE65_22',  'IRD_RPE65_18', 'IRD_RPE65_06',
        'IRD_RPE65_20', 'IRD_RPE65_16', 'IRD_RPE65_07', 'IRD_RPE65_09',
        'IRD_RPE65_23', 
    ]

    id_val = ['IRD_RPE65_02', 'IRD_RPE65_12', 'IRD_RPE65_01', ]

    # id_train = [
    #     '52458', '52065',  '35281', '30518', '45954', '52484', '51208', '6627',  '51013', '51870',
    #     '15313', '49885', '49883', '52374',  '21509', '48782', '52386', '10162',
    #     '35328', '35280', '28008', '52025', '52037', '51939', '34571', '16831', '40300', '48104', '49905',
    #     '49699', '35277', '35897', '51021', '44872', '49759', '29331',  '49031','48731',
    #     '49919', '43781','42979', '52009', '8837', '33996',   '35282', '51886', '52113', '51904', '51038',
    #     ]

    # id_val = ['26472', '51812', '35821', '52051', '52372', '51155', '16834', '20952', '23028',]

    # id_train = ["IRD_RPE65_19", "IRD_RPE65_17", "IRD_RPE65_15", "IRD_RPE65_11", "52458", "52386", "52374", "52372",
    #             "52113", "52065", "52051", "52037", '52025', "51939", "51812", "51021", "49919", "49885", "49883", "49759",
    #             "49699", "49031", "48731", "44872", '40300', "35897", "35328", "35281", "35280", "35277", "34571", "30518",
    #             "28008", "23028", '15313', "IRD_RPE65_04", "20952", "IRD_RPE65_21", "IRD_RPE65_20", "IRD_RPE65_18",
    #              'IRD_RPE65_08', 'IRD_RPE65_03', '48782', '52484', '29331',"IRD_RPE65_10", 'IRD_RPE65_16', 
    #             'IRD_RPE65_22', 'IRD_RPE65_13', 'IRD_RPE65_07', 'IRD_RPE65_06', 'IRD_RPE65_05', 'IRD_RPE65_01',
    #             '51886', '43781', '16831', '10162', 'IRD_RPE65_23', 'IRD_RPE65_09',
    #             '51904', '51038', '49905',  "IRD_RPE65_12", '42979', '52009', '8837', 
    #             ]

    # id_val = ['35821', '35282', '26472', '16834', '33996', '51155', '45954',
    #           '51208', '6627', '51013', '51870', '48104', '21509', 'IRD_RPE65_02',  ]

    print(set(id_train) & set(id_val))
    df1 = pd.DataFrame({'Train IDs': id_train,
                        })
    df1.to_csv(base_path + 'train_patient_ids.csv')

    df1 = pd.DataFrame({'Val IDs': id_val,
                        })
    df1.to_csv(base_path + 'val_patient_ids.csv')

    print('Factor train_size: ', len(id_train) / (len(id_train) + len(id_val)))
    print(len(id_train), len(id_val))
    IDs = []
    filenames_oct = get_filenames('dataset/2_OCTAnnotated/', 'vol')
    sum_mean = 0.0
    squared_sum = 0.0
    items = 0
    train_items = 0
    val_items = 0
    N = 0
    mean_data = []
    std_data = []

    for idx, f1 in enumerate(tqdm(filenames_oct)):
        N += 1
        name = os.path.splitext(os.path.split(f1)[1])[0]
        oct_read = vol_files(name)
        meta = oct_read.meta
        IDs.append(str(meta['PatientID']))

        data = oct_read.bscans[0].scan
        mask, annotations = get_images_masks(f1)
        sum_mean += np.mean(data / 255.)
        squared_sum += np.mean((data/255.) **2)
        if str(meta['PatientID']) in id_train:
            train_items += 1
            path_img_save = train_path_images
            path_msk_save = train_path_masks
            path_ann_save = train_path_ann
            path_img_save_patches = patches_images_train
            path_msk_save_patches = patches_masks_train
            img = Image.fromarray(data)
            msk = Image.fromarray(mask)
            ann = Image.fromarray(annotations)
            img.save(path_img_save + name + f".png")
            msk.save(path_msk_save + name + f".png")
            ann.save(path_ann_save + name + f".png")
            # crop_overlap(
            #     file=f1, image=data, mask=mask, path_img=path_img_save_patches, path_msk=path_msk_save_patches,
            #     patient_id=name + '_' + IDs[-1] + '_' + str(idx) + str(np.random.randint(0, 1000)),
            #     size=128, shift=128)
            slicing(
                file=f1, image=data, mask=mask, path_img=path_img_save_patches, path_msk=path_msk_save_patches,
                patient_id=name + '_' + IDs[-1] + '_' + str(idx) + str(np.random.randint(0, 1000)),
                size=128, shift=64)
        elif str(meta['PatientID']) in id_val:
            val_items += 1
            path_img_save = val_path_images
            path_msk_save = val_path_masks
            path_ann_save = val_path_ann
            path_img_save_patches = patches_images_val
            path_msk_save_patches = patches_masks_val
            
            img = Image.fromarray(data)
            msk = Image.fromarray(mask)
            ann = Image.fromarray(annotations)
            img.save(path_img_save + name + f".png")
            msk.save(path_msk_save + name + f".png")
            ann.save(path_ann_save + name + f".png")
            # crop_overlap(
            #     file=f1, image=data, mask=mask, path_img=path_img_save_patches, path_msk=path_msk_save_patches,
            #     patient_id=name + '_' + IDs[-1] + '_' + str(idx) + str(np.random.randint(0, 1000)),
            #     size=128, shift=128)
            slicing(
                file=f1, image=data, mask=mask, path_img=path_img_save_patches, path_msk=path_msk_save_patches,
                patient_id=name + '_' + IDs[-1] + '_' + str(idx) + str(np.random.randint(0, 1000)),
                size=128, shift=64)
        else:
            print('error', str(IDs[-1]))

    print(train_items, val_items)
    id_unique = list(set(IDs))
    id_unique.sort()
    print(id_unique)
    print(len(set(IDs)))
    mean = sum_mean / N
    std = (squared_sum / N - mean**2)**0.5
    print(f'Mean: {mean}, Std: {std}')
    sep = train_test_split(id_unique, shuffle=True, test_size=0.2)
    print(sep[0].sort())
    print(sep[1].sort())


if __name__ == "__main__":
    main()
