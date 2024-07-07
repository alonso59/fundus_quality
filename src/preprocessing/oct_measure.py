import os
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.utils import get_filenames
from oct_library import OCTProcessing

def main():
    base_path = 'logs/MAIN/'
    model_path = os.path.join(base_path, 'checkpoints/model.pth')
    model = torch.load(model_path, map_location='cuda')
    with open(os.path.join(base_path, 'experiment_cfg.yaml'), "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    # oct_path = 'dataset/2_OCTAnnotated/'
    oct_path = 'dataset/1_bonn_dataset_test/Full_dataset'
    # oct_path = '../dataset/1_bonn_dataset_test/VolumeFilesControls/'
    # oct_path = '../dataset/RPE65_VolumeFiles/'
    # oct_path = 'dataset/2_OCTAnnotated'
    # oct_path = 'dataset/1_bonn_dataset_test/RPE65_VolumeFiles'
    oct_files = get_filenames(oct_path, 'vol')
    config_path = os.path.join(base_path, 'experiment_cfg.yaml')
    i = 0
    base_path1 = 'logs/BEST_LARGE/' #logs/BEST_0.94, logs/2023-01-22_02_39_26
    model_path1 = os.path.join(base_path1, 'checkpoints/model.pth')
    df = pd.DataFrame()
    diction = []
    for etl in ['0.5mm', '1mm', '2mm', '3mm', '6mm']:
        for oct_file in tqdm(oct_files):
            try:
                oct_process = OCTProcessing(oct_file=oct_file, config_path=config_path, 
                            model_path=model_path, gamma=1, alphaTV=None) # 125, 36, 10, 68, 15
                oct_process.fovea_forward(foveax_pos=None, ETDRS_loc=etl) #
                _, _, __ = oct_process.volume_forward(model_path1, gamma=1, alpha=None, interpolated=True, tv_smooth=False, plot=False, bscan_positions=False)
                diction.append(oct_process.results)
            except Exception as exc:
                print(exc)
    df = pd.DataFrame(diction)
    df.to_csv(f'Annotated_v3.csv')

if __name__ == '__main__':
    main()

