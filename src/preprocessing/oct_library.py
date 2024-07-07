import os
import yaml
import torch
import matplotlib 
import numpy as np
import eyepy as ep
import albumentations as T
import matplotlib.pyplot as plt
import torch.nn.functional as F

from scipy import ndimage
from PIL import Image
from matplotlib import cm
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from patchify import patchify, unpatchify
from skimage.restoration import denoise_tv_chambolle

def predict(model, x_image, device):   
    
    MEAN =  0.1338 # 0 # 0.1338  # 0.1338 0.13505013393330723
    STD =  0.1466 # 1 # 0.1466  # 0.1466 0.21162075769722669
    normalization = T.Normalize(mean=MEAN, std=STD)

    n_dimention = np.ndim(x_image)
    # image = np.repeat(image, 3, axis=-1)
    image = normalization(image=x_image)
    image = np.expand_dims(image['image'], axis=-1)
    if n_dimention == 2:
        image = image.transpose((2, 0, 1))
    elif n_dimention == 3:
        image = image.transpose((0, 3, 1, 2))
    image = torch.tensor(image, dtype=torch.float, device=device)
    
    if torch.Tensor.dim(image) == 3:
        image = image.unsqueeze(0)

    y_pred = model(image)
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    if n_dimention == 2:
        y_pred = y_pred.squeeze(0)
    elif n_dimention == 3:
        y_pred = y_pred.squeeze(1)

    y_pred = y_pred.detach().cpu().numpy()
    return y_pred


class OCTProcessing:
    def __init__(self, oct_file, config_path, model_path, gamma, alphaTV, per_batch=True, dataset='bonn', bscan_idx=None):
        
        with open(config_path, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
        self.imgh = cfg['general']['img_sizeh']
        self.imgw = cfg['general']['img_sizew']
        self.classes = cfg['general']['classes']
        self.mode = cfg['general']['img_type']
        self.oct_file = oct_file
        self.model = torch.load(model_path, map_location='cuda')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.gamma = gamma
        self.alphaTV = alphaTV
        self.per_batch = per_batch
        
        if dataset == 'bonn':
            self.oct_reader_spec(self.oct_file)
            if bscan_idx is not None:
                self.bscan_idx = bscan_idx
            else:
                self.bscan_idx = len(self.oct)//2
            self.bscan_get()
            self.oct_metadata()   
        elif dataset == 'duke':
            pass
        elif dataset == 'hms':
            pass
        
        self.pred_class_map, self.pred_rgb, self.overlay = self.get_segmentation(self.model, self.bscan_fovea, self.mode, self.gamma, alpha=self.alphaTV)

    def __len__(self):
        return len(self.oct)
    
    def oct_reader_spec(self, oct_file):
        self.oct = ep.import_heyex_vol(oct_file)

    def bscan_get(self):
        self.bscan_fovea = self.oct[self.bscan_idx].data

    def oct_metadata(self):
        self.patient = os.path.splitext(os.path.split(self.oct_file)[1])[0] 
        self.scale_y = self.oct.meta.as_dict()['scale_y']
        self.scale_x = self.oct.meta.as_dict()['scale_x']
        
        self.visit_date = self.oct.meta.as_dict()['visit_date']
        self.laterality = self.oct.meta.as_dict()['laterality']

        self.loc_fovea = self.oct.meta.as_dict()['bscan_meta'][self.bscan_idx]['start_pos'][1] // self.scale_x
        self.fovea_xstart = self.oct.meta.as_dict()['bscan_meta'][self.bscan_idx]['start_pos'][0] // self.scale_x
        self.fovea_xstop = self.oct.meta.as_dict()['bscan_meta'][self.bscan_idx]['end_pos'][0] // self.scale_x

        self.results = {}
        self.results['N_Bscans'] = self.__len__()
        self.results['Fovea_BScan'] = self.bscan_idx
        self.results['scale_y'] = self.scale_y
        self.results['scale_x'] = self.scale_x
        self.results['visit_date'] = str(self.visit_date).partition("T")[0]
        self.results['laterality'] = self.laterality
        self.results['Y_Fovea'] = self.loc_fovea
        self.results['Fovea_xstart'] = self.fovea_xstart
        self.results['Fovea_xstop'] = self.fovea_xstop

        self.pixel_2_mm2 = self.scale_x * self.scale_x

    def get_segmentation(self, model, img, mode, gamma=1, alpha=0.0001):
        img_in = gray_gamma(img, gamma=gamma)
        img_in = tv_denoising(img_in, alpha=alpha)
        pady = 0
        if self.imgh is not img_in.shape[0]:
            pady = np.abs(self.imgh - img_in.shape[0]) // 2
            img_in = np.pad(img_in, [(pady, ), (0, )], 'constant', constant_values=0)
        if mode == 'large':
            shape_image_x = img_in.shape
            image_x = F.interpolate(torch.from_numpy(img_in).unsqueeze(0).unsqueeze(0).float(), (self.imgh, self.imgw), mode='bilinear', align_corners=False).squeeze().numpy()
            pred = predict(model, image_x, self.device)
            preds = F.interpolate(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(), (shape_image_x[0], shape_image_x[1]), mode='nearest').squeeze().numpy()
            img_out = img_in
        
        if mode == 'patches':
            pady = 0
            padx = 0
            if img.shape[0] == 496:
                pady = 8
            if img.shape[1] == 1000:
                padx = 12
            large_image = np.pad(img_in, [(pady, ), (padx, )], 'constant', constant_values=0)
            patches_images = patchify(large_image, (self.imgh, self.imgw), step=self.imgw)
            predictions = []
            for i in range(patches_images.shape[0]):
                for j in range(patches_images.shape[1]):
                    image_x = patches_images[i, j, :, :]
                    pred = predict(model, image_x, self.device)
                    pred = Image.fromarray(pred.astype('uint8'))
                    predictions.append(np.array(pred))
            predictions = np.array(predictions)
            predictions = np.reshape(predictions, patches_images.shape)
            rec_img = unpatchify(patches=patches_images, imsize=(self.imgh * predictions.shape[0], self.imgw * predictions.shape[1]))
            preds = unpatchify(patches=predictions, imsize=(self.imgh * predictions.shape[0], self.imgw * predictions.shape[1]))
            preds = preds[pady:img.shape[0]+pady, padx:img.shape[1]+padx]
            img_out = rec_img[pady:img.shape[0]+pady, padx:img.shape[1]+padx]
        
        if mode == 'slices':
            predictions = []
            if not self.per_batch:
                for i in range(self.imgw, img_in.shape[1]+self.imgw, self.imgw):
                    image_x = img_in[:, i - self.imgw:i]
                    pred = predict(model, image_x, self.device)
                    predictions.append(pred.astype('uint8'))
            else:
                img_pred = []
                for i in range(self.imgw, img_in.shape[1]+self.imgw, self.imgw):
                    image_x = img_in[:, i - self.imgw:i]
                    img_pred.append(image_x)
                img_pred = np.array(img_pred)
                predictions = predict(self.model, img_pred, self.device)
            predictions = np.array(predictions)
            preds = np.hstack(predictions)
            img_out = img_in
        # print(type(preds))
        # preds = closing_opening(preds.type('uint8'), self.classes)
        # print(preds.max())
        if self.imgh is not img_in.shape[0]:
            preds = preds[pady:img_in.shape[0]-pady,:]
            img_out = img_out[pady:img_in.shape[0]-pady,:]
        # prepare output
        shape_1 = (preds.shape[0], preds.shape[1], 3)
        pred_rgb = np.zeros(shape=shape_1, dtype='uint8')

        norm = matplotlib.colors.Normalize(vmin=0, vmax=preds.max())
        for idx in range(1, int(preds.max())+1):
            pred_rgb[..., 0] = np.where(preds == idx, cm.hsv(norm(idx), bytes=True)[0], pred_rgb[..., 0])
            pred_rgb[..., 1] = np.where(preds == idx, cm.hsv(norm(idx), bytes=True)[1], pred_rgb[..., 1])
            pred_rgb[..., 2] = np.where(preds == idx, cm.hsv(norm(idx), bytes=True)[2], pred_rgb[..., 2])

        # output
        img_overlay = Image.fromarray(img_out)
        pred_overlay = Image.fromarray(pred_rgb)
        img_overlay = img_overlay.convert("RGBA")
        pred_overlay = pred_overlay.convert("RGBA")
        overlay = Image.blend(img_overlay, pred_overlay, 0.4)
        overlay = np.array(overlay)
        self.bscan_fovea = img_out
        return preds, pred_rgb, overlay

    def get_individual_layers_segmentation(self):
        self.binary_opl = get_layer_binary_mask(self.sample_pred, self.classes, layer='OPL', offset=0)
        self.binary_elm = get_layer_binary_mask(self.sample_pred, self.classes, layer='ELM', offset=0)
        self.binary_ez = get_layer_binary_mask(self.sample_pred, self.classes, layer='EZ', offset=0)
        self.binary_bm = get_layer_binary_mask(self.sample_pred, self.classes, layer='BM', offset=0)
        
        self.segmented_opl = np.multiply(self.binary_opl, self.sample_bscan)
        self.segmented_elm = np.multiply(self.binary_elm, self.sample_bscan)
        self.segmented_ez = np.multiply(self.binary_ez, self.sample_bscan)
        self.segmented_bm = np.multiply(self.binary_bm, self.sample_bscan)

    def etdrs_localizations(self, foveax_pos=None, ETDRS_loc='6mm'):
        # ETDRS zones
        pix_to_mm = 1 // self.scale_x
        
        self.outer_ring_max = foveax_pos + int(pix_to_mm * 3) // 1  # 3 mm ratio
        self.outer_ring_min = foveax_pos - int(pix_to_mm * 3) // 1 # 3 mm ratio = 6mm diameter

        self.inner_ring_max = (foveax_pos + pix_to_mm * 1.5) // 1  # 1.5 mm ratio
        self.inner_ring_min = (foveax_pos - pix_to_mm * 1.5) // 1  # 1.5 mm ratio = 3mm diameter

        self.ring_2mm_max = (foveax_pos + pix_to_mm * 1) // 1  # 1 mm ratio
        self.ring_2mm_min = (foveax_pos - pix_to_mm * 1) // 1  # 1 mm ratio = 2mm diameter

        self.center_fovea_max = (foveax_pos + pix_to_mm // 2) // 1 # 0.5 mm ratio
        self.center_fovea_min = (foveax_pos - pix_to_mm // 2) // 1 # 0.5 mm ratio = 1mm diameter

        self.um5_max = (foveax_pos + pix_to_mm // 4) // 1 # 0.25 mm ratio
        self.um5_min = (foveax_pos - pix_to_mm // 4) // 1 # 0.25 mm ratio = 0.5mm diameter

        ymin, ymax = get_ylimits_roi(self.classes, msk=self.pred_class_map, offset=2)
        
        if ETDRS_loc == '0.5mm':
            xmin, xmax = int(self.um5_min), int(self.um5_max)
        if ETDRS_loc == '1mm':
            xmin, xmax = int(self.center_fovea_min), int(self.center_fovea_max)
        if ETDRS_loc == '2mm':
            xmin, xmax = int(self.ring_2mm_min), int(self.ring_2mm_max)
        if ETDRS_loc == '3mm':
            xmin, xmax = int(self.inner_ring_min), int(self.inner_ring_max)
        if ETDRS_loc == '6mm':
            xmin, xmax = int(self.outer_ring_min), int(self.outer_ring_max)

        self.references = [xmin, xmax, ymin, ymax]

    def fovea_forward(self, foveax_pos=None, ETDRS_loc='6mm'):

        # assert self.bscan_fovea.shape[1] <= 1024, self.__del__()
            
        if foveax_pos is None:
            foveax_pos = (self.bscan_fovea.shape[1]) // 2
        
        self.etdrs_localizations(foveax_pos, ETDRS_loc)

        self.sample_bscan = self.bscan_fovea[self.references[2]:self.references[3],self.references[0]:self.references[1]]
        # p = np.percentile(self.sample_bscan, 95)
        # self.sample_bscan = self.sample_bscan / p
        self.sample_pred = self.pred_class_map[self.references[2]:self.references[3],self.references[0]:self.references[1]]
        self.sample_pred_rgb = self.pred_rgb[self.references[2]:self.references[3],self.references[0]:self.references[1]]
        self.sample_overlay = self.overlay[self.references[2]:self.references[3],self.references[0]:self.references[1]]

        self.get_individual_layers_segmentation()

        # getting full layer segmentations
        self.binary_total = np.where(np.logical_and(self.sample_pred.astype('float') <= 4, self.sample_pred.astype('float') > 0), 1, 0)
        self.segmented_total = np.multiply(self.binary_total, self.sample_bscan)

        # getting max peaks and ELM BM localization
        self.max_opl_x, self.max_opl_y = get_max_peak(self.segmented_opl)
        self.max_opl_x, self.max_opl_y = rm_zero_idx(self.max_opl_x, self.max_opl_y)
        # self.xnew_opl = np.linspace(self.max_opl_x[0], self.max_opl_x[-1], num=500)
        # f_interpol = interp1d(self.max_opl_x, self.max_opl_y, kind='linear', fill_value="extrapolate")
        # self.max_opl_y_tv = denoising_1D_TV(f_interpol(self.xnew_opl), 10)

        self.max_ez_x, self.max_ez_y = get_max_peak(self.segmented_ez)
        self.max_ez_x, self.max_ez_y = rm_zero_idx(self.max_ez_x, self.max_ez_y)
        # self.xnew_ez = np.linspace(self.max_ez_x[0], self.max_ez_x[-1], num=500)
        # f_interpol = interp1d(self.max_ez_x, self.max_ez_y, kind='linear', fill_value="extrapolate")
        # self.max_ez_y_tv = denoising_1D_TV(f_interpol(self.xnew_ez), 10)
        
        self.max_elm_x, self.max_elm_y = get_max_peak(self.segmented_elm)
        self.max_elm_x, self.max_elm_y = rm_zero_idx(self.max_elm_x, self.max_elm_y)
        # self.xnew_elm = np.linspace(self.max_elm_x[0], self.max_elm_x[-1], num=500)
        # f_interpol = interp1d(self.max_elm_y, self.max_elm_x, kind='linear', fill_value="extrapolate")
        # self.max_elm_y_tv = denoising_1D_TV(f_interpol(self.xnew_elm), 10)
        
        self.lim_elm = get_limit(self.binary_elm, side='min', offset=2)
        self.lim_bm = get_limit(self.binary_bm, side='max')

        # *********** compute bio-markers *********** 
        rezi_mean, rezi_std = get_rEZI(self.max_ez_y, self.max_opl_y) # RELATIVE EZ INSTENSITY

        ez_th_mean, ez_th_std = get_thickness(self.binary_ez, self.scale_y) # EZ THICKNESS
        
        opl_th_mean, opl_th_std = get_thickness(self.binary_opl, self.scale_y) # OPL THICKNESS
        
        elm_th_mean, elm_th_std = get_thickness(self.binary_elm, self.scale_y) # ELM THICKNESS
        
        rpe_th_mean, rpe_ths_std = get_thickness(self.binary_bm, self.scale_y) # IZ+RPE THICKNESS

        opl_2_elm_mean, opl_2_elm_std = get_distance_in_mm(self.max_opl_y, self.lim_bm, self.scale_y) # DISTANCE FROM OPL TO ELM PEAKS IN MM

        opl_2_ez_mean, opl_2_ez_std = get_distance_in_mm(self.max_opl_y, self.max_ez_y, self.scale_y) # DISTANCE FROM OPL TO EZ PEAKS IN MM

        elm_2_ez_mean, elm_2_ez_std = get_distance_in_mm(self.lim_bm, self.max_ez_y, self.scale_y) # DISTANCE FROM ELM TO EZ PEAKS IN MM

        ez_2_bm_mean, ez_2_bm_std = get_distance_in_mm(self.max_ez_y, self.lim_bm, self.scale_y) # DISTANCE FROM EZ TO BM PEAKS IN MM
        
        elm_2_bm_mean, elm_2_bm_std = get_distance_in_mm(self.lim_bm, self.lim_bm, self.scale_y) # DISTANCE FROM ELM TO BM PEAKS IN MM

        _, ez_tv_mean, ez_tv_std = get_total_variation(self.segmented_ez, 3) # TOTAL VARIATION

        ez_mask = get_layer_binary_mask(self.pred_class_map, self.classes, layer='EZ', offset=0)
        pos_ez = np.where(ez_mask)
        try:
            self.ez_xmin_fovea = np.min(pos_ez[1][np.nonzero(pos_ez[1])])
            self.ez_xmax_fovea = np.max(pos_ez[1][np.nonzero(pos_ez[1])])
        except:
            self.ez_xmin_fovea = 0
            self.ez_xmax_fovea = 0
        # print(ez_xmin, ez_xmax)
        self.ez_fovea_width = np.abs(self.ez_xmin_fovea - self.ez_xmax_fovea) * self.scale_x
        axial_fovea_ez_area = get_area(self.binary_ez) * self.pixel_2_mm2
        
        if 'Anonym' in self.patient or 'Kunzel' in self.patient:
            y = 'control'
        else:
            y = 'RPE65'
        
        self.results = {
            'Name_Patient':self.patient,
            'Visit_date': self.visit_date,
            'Laterality':self.laterality,
            'Scale x pixel/mm':self.scale_x,
            'Scale y pixel/mm':self.scale_y,
            'Y_Fovea':self.loc_fovea,

            'rEZI_mean':rezi_mean,
            'EZ_th_mean':ez_th_mean,
            'EZ_OPL_mean':opl_2_ez_mean,
            'EZ_ELM_mean':elm_2_ez_mean,
            'EZ_BM_mean':ez_2_bm_mean,
            'ELM_BM_mean':elm_2_bm_mean,
            'EZ_TV_mean':ez_tv_mean,

            'rEZI_std': rezi_std,
            'EZ_th_std':ez_th_std,
            'EZ_OPL_std':opl_2_ez_std,
            'EZ_ELM_std':elm_2_ez_std,
            'EZ_BM_std':ez_2_bm_std,
            'ELM_BM_std':elm_2_bm_std,
            'EZ_TV_std':ez_tv_std,

            'Fovea_EZ_Diameter': self.ez_fovea_width,
            'Fovea_BScan_EZ_Area': axial_fovea_ez_area,

            "Patient": y, 
            "ETDRS_loc": ETDRS_loc,
        }
    
    def volume_forward(self, big_model_path=None, gamma=1.5, alpha=0.05, interpolated=True, tv_smooth=False, plot=False, bscan_positions=True):
        X_MINS = []
        X_MAX = []
        Y_POS = []

        delta_ez_lim = []

        if big_model_path is not None:
            model = torch.load(big_model_path, map_location='cuda')
            mode = 'large'
        else:
            model = self.model
            mode = self.mode

        if len(self.oct) > 1:
            for idx in range(len(self.oct)):
                bscan = self.oct[idx].data
                xstart = self.oct.meta.as_dict()['bscan_meta'][idx]['start_pos'][0]//self.oct.meta.as_dict()['scale_x']
                y_position = self.oct.meta.as_dict()['bscan_meta'][idx]['start_pos'][1] // self.oct.meta.as_dict()['scale_x']

                # if y_position >= self.outer_ring_min and y_position <= self.outer_ring_max:
                try:
                    # print(bscan.shape)
                    pred_class_map, _, _ = self.get_segmentation(model, bscan, mode, gamma=gamma, alpha=alpha)
                    ez_mask = get_layer_binary_mask(pred_class_map, self.classes, layer='EZ', offset=0)
                    pos_ez = np.where(ez_mask)
                    try:
                        xmin = np.min(pos_ez[1][np.nonzero(pos_ez[1])])   
                    except:
                        xmin = 0
                    try:
                        xmax = np.max(pos_ez[1][np.nonzero(pos_ez[1])])
                    except:
                        xmax = 0

                    if (xmax - xmin) < 1:
                        xmin = 0
                        xmax = 1

                    if xmin != 0 and xmax != 1:
                        delta_ez_lim.append(np.abs(xmax * self.scale_x - xmin * self.scale_x))
                        X_MINS.append(xmin + xstart)
                        X_MAX.append(xmax + (xstart))
                        Y_POS.append(y_position)
                except NameError as exc:
                    print(exc)
            try:
                if interpolated and len(self.oct) > 1:
                    ynew1 = np.linspace(Y_POS[0], Y_POS[-1], num=int(Y_POS[0] - Y_POS[-1]))
                    f1 = interp1d(Y_POS, X_MAX, kind='linear', fill_value="extrapolate")
                    ynew2 = np.linspace(Y_POS[0], Y_POS[-1], num=int(Y_POS[0] - Y_POS[-1]))
                    f2 = interp1d(Y_POS, X_MINS, kind='linear', fill_value="extrapolate")
                    func1 = f1(ynew1)
                    func2 = f2(ynew2)
                    array_diff = np.array(func1) - np.array(func2)
                    volume_area = np.sum(array_diff) * self.pixel_2_mm2
                else:
                    volume_area = np.nan
            except:
                volume_area = np.nan
        else:
            Y_POS = self.oct.meta.as_dict()['bscan_meta'][0]['start_pos'][1] // self.oct.meta.as_dict()['scale_x']
            X_MINS = self.ez_xmin_fovea
            X_MAX = self.ez_xmax_fovea
            delta_ez_lim = self.ez_fovea_width
            volume_area = np.nan
            
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(8,8), frameon=False)
            self.oct.plot(localizer=True, bscan_positions=bscan_positions)
            ax.scatter(X_MINS, Y_POS, c='red', s=10)
            ax.scatter(X_MAX, Y_POS, c='red', s=10)
            if  interpolated:
                ax.plot(func1, ynew1, '-', c='cornflowerblue', linewidth=1)
                ax.plot(func2, ynew2, '-', c='cornflowerblue', linewidth=1)
            else:
                ax.plot(X_MINS, Y_POS, '-', c='orange', linewidth=1)
                ax.plot(X_MAX, Y_POS, '-', c='orange', linewidth=1)
            if tv_smooth and self.__len__() > 1:
                assert interpolated, 'Interpolation for Total variation 1-D smooth is needed -> arg* interpolated = True'
                YNEW1 = denoising_1D_TV(f1(ynew1), 100)
                ax.plot(YNEW1, ynew1, '-', c='cornflowerblue', linewidth=1)
                YNEW2 = denoising_1D_TV(f2(ynew2), 100)
                ax.plot(YNEW2, ynew2, '-', c='blue', linewidth=1)
                # plt.legend(loc='best')
            self.plot_etdrs_grid(ax)
            ax.tick_params(labelsize=12)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            # ax.legend(loc='best')
            fig.tight_layout()
        self.results['Volume_area'] = volume_area
        return Y_POS, delta_ez_lim, volume_area

    def plot_etdrs_grid(self, ax):
        pix_to_mm = 1 // self.scale_x
        r1 = int(pix_to_mm * 3)  # 3 mm ratio
        r2 = r1/2
        r3 = r2/2
        r4 = r3/2
        r5 = int(pix_to_mm * 1) 

        foveax_pos = (self.oct.shape[1]) // 2
        if self.oct.localizer.shape[0] == self.bscan_fovea.shape[1]:
            center = int(foveax_pos*1.55) 
        else:
            center = int(foveax_pos*3.141)
        theta = np.linspace(0, 2*np.pi, 100)
        d45 = np.pi / 4
        
        # radio * constant + x_point
        
        x1 = center + r1 * np.cos(theta)
        y1 = center + r1 * np.sin(theta)

        
        x2 = center + r2 * np.cos(theta)
        y2 = center + r2 * np.sin(theta)

        
        x3 = center + r3 * np.cos(theta)
        y3 = center + r3 * np.sin(theta)
        
        
        x4 = center + r4 * np.cos(theta)
        y4 = center + r4 * np.sin(theta)

        
        x5 = center + r5 * np.cos(theta)
        y5 = center + r5 * np.sin(theta)

        # Quad I
        x_pt1 = center + r1 * np.cos(d45)
        y_pt1 = center + r1 * -np.sin(d45)
        x_pt2 = center + r3 * np.cos(d45)
        y_pt2 = center + r3 * -np.sin(d45)

        # Quad II
        x_pt3 = center + r1 * -np.cos(d45)
        y_pt3 = center + r1 * -np.sin(d45)
        x_pt4 = center + r3 * -np.cos(d45)
        y_pt4 = center + r3 * -np.sin(d45)

        # Quad III
        x_pt5 = center + r1 * -np.cos(d45)
        y_pt5 = center + r1 * np.sin(d45)
        x_pt6 = center + r3 * -np.cos(d45)
        y_pt6 = center + r3 * np.sin(d45)

        # Quad IV
        x_pt7 = center + r1 * np.cos(d45)
        y_pt7 = center + r1 * np.sin(d45)
        x_pt8 = center + r3 * np.cos(d45)
        y_pt8 = center + r3 * np.sin(d45)
        ax.plot([x_pt1, x_pt2], [y_pt1, y_pt2], c='lime')
        ax.plot([x_pt3, x_pt4], [y_pt3, y_pt4], c='lime')
        ax.plot([x_pt5, x_pt6], [y_pt5, y_pt6], c='lime')
        ax.plot([x_pt7, x_pt8], [y_pt7, y_pt8], c='lime')
        ax.plot(x1,y1, c='lime')
        ax.plot(x2,y2, c='lime')
        ax.plot(x5,y5, '--',c='lime')
        ax.plot(x3,y3, c='lime')
        ax.plot(x4,y4, '--',c='lime')

    def plot_overlay_oct_segmentation(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(14,10), gridspec_kw={'width_ratios': [1]}, frameon=False)
        ax.set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax.set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax.tick_params(labelsize=12)
        ax.tick_params(labelsize=12)
        ax.imshow(self.overlay)

    def plot_slo_etdrs(self):
        pix_to_mm = 1 // self.scale_x
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(25,10), frameon=False) #gridspec_kw={'width_ratios': [1, 2]}
        self.oct.plot(localizer=True, bscan_positions=False, ax=ax)
        self.plot_etdrs_grid(ax)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.tick_params(labelsize=12)

    def plot_slo_fovea(self, etdrs_grid=False):
        pix_to_mm = 1 // self.scale_x

        fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(25,10), gridspec_kw={'width_ratios': [1, 1.8]}, frameon=False) #gridspec_kw={'width_ratios': [1, 2]}
        self.oct.plot(localizer=True, bscan_positions=True, ax=ax[0])
        ax[1].imshow(self.bscan_fovea, cmap='gray')
        ax[0].scatter(20+self.fovea_xstart, self.loc_fovea, c='red', s=50)
        ax[0].scatter(self.bscan_fovea.shape[1]-20+self.fovea_xstart, self.loc_fovea, c='red', s=50)
        if etdrs_grid:
            self.plot_etdrs_grid(ax[0])
        ax[0].legend(loc='best')
        ax[1].set_xlabel('B-Scan (X)', fontsize=24, weight="bold")
        ax[1].set_ylabel('A-Scan (Y)', fontsize=24, weight="bold")
        ax[0].set_xlabel('B-Scan (X)', fontsize=24, weight="bold")
        ax[0].set_ylabel('Volume (Z)', fontsize=24, weight="bold")
        ax[0].tick_params(labelsize=20)
        ax[0].tick_params(labelsize=20)
        ax[1].tick_params(labelsize=20)
        ax[1].tick_params(labelsize=20)

    def plot_segmentation_full(self):
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,8), dpi=200, frameon=False)
        ax.imshow(self.overlay, cmap='gray')
        ax.set_xticks([])
        figure.tight_layout()

    def plot_segmentation_localization(self):
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,8), dpi=200, frameon=False)
        ax.imshow(self.overlay, cmap='gray')

        ax.plot([self.um5_min, self.um5_min], [self.references[2], self.references[3]], '--', linewidth=1, color='lime', label='0.5mm')
        ax.plot([self.um5_max, self.um5_max], [self.references[2], self.references[3]], '--', linewidth=1, color='lime')
        ax.plot([self.um5_min, self.um5_max], [self.references[3], self.references[3]], '--', linewidth=1, color='lime')
        ax.plot([self.um5_min, self.um5_max], [self.references[2], self.references[2]], '--', linewidth=1, color='lime')

        ax.plot([self.center_fovea_min, self.center_fovea_min], [self.references[2], self.references[3]], linewidth=1, color='lime', label='1mm')
        ax.plot([self.center_fovea_max, self.center_fovea_max], [self.references[2], self.references[3]], linewidth=1, color='lime')
        ax.plot([self.center_fovea_min, self.center_fovea_max], [self.references[3], self.references[3]], linewidth=1, color='lime')
        ax.plot([self.center_fovea_min, self.center_fovea_max], [self.references[2], self.references[2]], linewidth=1, color='lime')

        ax.plot([self.ring_2mm_min, self.ring_2mm_min], [self.references[2], self.references[3]], '--', linewidth=1, color='lime', label='2mm')
        ax.plot([self.ring_2mm_max, self.ring_2mm_max], [self.references[2], self.references[3]], '--', linewidth=1, color='lime')
        ax.plot([self.ring_2mm_min, self.ring_2mm_max], [self.references[3], self.references[3]], '--', linewidth=1, color='lime')
        ax.plot([self.ring_2mm_min, self.ring_2mm_max], [self.references[2], self.references[2]], '--', linewidth=1, color='lime')

        ax.plot([self.inner_ring_min, self.inner_ring_min], [self.references[2], self.references[3]], linewidth=1, color='lime', label='3mm')
        ax.plot([self.inner_ring_max, self.inner_ring_max], [self.references[2], self.references[3]], linewidth=1, color='lime')
        ax.plot([self.inner_ring_min, self.inner_ring_max], [self.references[3], self.references[3]], linewidth=1, color='lime')
        ax.plot([self.inner_ring_min, self.inner_ring_max], [self.references[2], self.references[2]], linewidth=1, color='lime')

        # ax.plot([self.outer_ring_min, self.outer_ring_min], [self.references[2], self.references[3]], linewidth=1, color='g', label='6mm Outer Ring')
        # ax.plot([self.outer_ring_max, self.outer_ring_max], [self.references[2], self.references[3]], linewidth=1, color='g')
        # ax.plot([self.outer_ring_min, self.outer_ring_max], [self.references[3], self.references[3]], linewidth=1, color='g')
        # ax.plot([self.outer_ring_min, self.outer_ring_max], [self.references[2], self.references[2]], linewidth=1, color='g')
        # ax.set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        # ax.set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.tick_params(labelsize=12)
        # ax.legend(loc='best')
        figure.tight_layout()

    def plot_results(self):
        figure, ax = plt.subplots(nrows=3, ncols=1, figsize=(12,8), frameon=False, dpi=200)
        ax[0].imshow(self.sample_bscan, cmap='gray')
        ax[1].imshow(self.sample_overlay, cmap='gray')
        ax[2].imshow(self.sample_bscan, cmap='gray')
        ax[2].plot(self.max_opl_x, self.max_opl_y, c='cyan', label='Max Peaks OPL')
        ax[2].plot(self.max_ez_x, self.max_ez_y, linewidth=1.5,c='lime', label='Max Peaks EZ')
        # ax[2].plot(self.max_elm_x, self.max_elm_y, c='violet', label='ELM')
        ax[2].plot(self.lim_elm, c='violet', label='ELM')
        ax[2].plot(self.lim_bm, c='red', label='BM')
        ax[2].legend(loc='best')
        ax[0].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax[1].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax[2].set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax[2].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)
        ax[2].tick_params(labelsize=12)
        ax[2].tick_params(labelsize=12)
        figure.tight_layout()

    def plot_total_variation_alphas(self, alphas=[0.005, 0.05, 0.5], beta=3, xlabel=False):
        p = np.percentile(self.bscan_fovea[self.references[2]:self.references[3], self.references[0]:self.references[1]], 95)
        sample_bscan1 = self.bscan_fovea[self.references[2]:self.references[3], self.references[0]:self.references[1]] / p

        EZ_segmented1 = np.multiply(self.binary_ez, sample_bscan1)
        locales, mean_tv, _ = get_total_variation(EZ_segmented1, beta)

        plt.rc('font', size=30)

        plt.figure(figsize=(14,6), dpi=128, frameon=False) #, frameon=True
        
        plt.scatter(np.arange(locales.shape[0]), locales, s=5)
        plt.plot(locales, '-o', label=r'Original $\overline{LV}$: ' + format(mean_tv,'.1e'), linewidth=2)
        xlimit = [0, locales.shape[0]]
        for w in alphas:
            tv_denoised = denoise_tv_chambolle(sample_bscan1, weight=w)
            EZ_segmented1 = np.multiply(self.binary_ez, tv_denoised)
            locales, mean_tv, _ = get_total_variation(EZ_segmented1, beta)
            plt.plot(locales, '--*', linewidth=2, label=r'$\alpha$: ' +  format(w,'.1e') + r' $\overline{LV}$: '+format(mean_tv,'.1e'))
        if xlabel:
            plt.xlabel(r'N/(2$\beta$+1)', fontsize=32, weight="bold")
        plt.ylabel('LV', fontsize=34, weight="bold")
        plt.xticks(fontsize=34)
        plt.yticks(fontsize=34)
        plt.locator_params(axis='y', nbins=4)
        plt.locator_params(axis='x', nbins=5)
        plt.grid(True)
        plt.xlim(xlimit)
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0e')) 
        # plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.03f')) 
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc="lower left", fontsize="24", ncol=2)

    def plot_intensity_profiles(self, shift=1000):
        # p = np.percentile(self.bscan_fovea[self.references[2]:self.references[3], self.references[0]:self.references[1]], 95)
        sample_bscan1 = self.bscan_fovea[self.references[2]:self.references[3], self.references[0]:self.references[1]]  / 255
        segmented_total1 = np.multiply(self.binary_total, sample_bscan1)
        img = segmented_total1
        int_prof_x = []
        size = 1

        for i in range(size, img.shape[1], size):
            window = img[:, i - size:i]
            matrix_mean = np.zeros((img.shape[0]))
            for j in range(window.shape[0]):
                matrix_mean[j] = window[j, :].mean()
            int_prof_x.append(matrix_mean)

        # for t in range(240, np.array(int_prof_x).shape[0], shift):
        print(np.array(int_prof_x).shape)
        intensity = np.array(int_prof_x)[240, :]
        peaks, _ = find_peaks(intensity, height=0)
        y = np.arange(img.shape[0])
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figheight(4)
        fig.set_figwidth(10)
        ax.plot(intensity, y, 'k')
        ax.plot(intensity[peaks], peaks, "o", c='green')
        ax.set_xlabel('Grey value', fontsize=14, weight="bold")
        ax.set_ylabel(f'A-Scan \nDistance [Pixels]', fontsize=14, weight="bold")
        plt.gca().invert_yaxis()
        plt.show()
        # Z_INT = np.array(int_prof_x)
        # x_int = np.arange(Z_INT.shape[1])
        # y_int = np.arange(Z_INT.shape[0])
        # X_INT, Y_INT = np.meshgrid(x_int, y_int)
        # fig = plt.figure(figsize=(14, 8), dpi=200, frameon=False)
        # ax = plt.axes(projection='3d')
        # ax.set_aspect(aspect='auto', adjustable='datalim')
        # ax.contour3D(X_INT, Y_INT, Z_INT, 100, cmap='jet')
        # ax.set_xlabel('A-Scan(Y)', fontsize=14, weight="bold")
        # ax.set_ylabel('B-Scan(X)', fontsize=14, weight="bold")
        # ax.set_zlabel('Grey value', fontsize=14, weight="bold")
        # ax.view_init(35, -95)    

def closing_opening(pred, n_class):
    iter = 1
    one_hot = np.eye(5)[pred]
    
    

    one_hot[:, :, 3] = np.array(ndimage.binary_opening(one_hot[:, :, 3], iterations=1), dtype=type(pred))
    one_hot[:, :, 3] = np.array(ndimage.binary_closing(one_hot[:, :, 3], iterations=3), dtype=type(pred))

    one_hot[:, :, 4] = np.array(ndimage.binary_opening(one_hot[:, :, 4], iterations=1), dtype=type(pred))
    one_hot[:, :, 4] = np.array(ndimage.binary_closing(one_hot[:, :, 4], iterations=3), dtype=type(pred))

    one_hot[:, :, 2] = np.array(ndimage.binary_opening(one_hot[:, :, 2], iterations=1), dtype=type(pred))
    one_hot[:, :, 2] = np.array(ndimage.binary_closing(one_hot[:, :, 2], iterations=3), dtype=type(pred))

    one_hot[:, :, 1] = np.array(ndimage.binary_opening(one_hot[:, :, 1], iterations=1), dtype=type(pred))
    one_hot[:, :, 1] = np.array(ndimage.binary_closing(one_hot[:, :, 1], iterations=3), dtype=type(pred))

    amax_preds = np.argmax(one_hot, axis=2)

    return amax_preds

def rm_zero_idx(x_array, y_array):
    xold = np.array(x_array).copy()
    zero_idx = np.where(xold == np.nan)
    yold = np.delete(y_array, zero_idx)
    xold = xold[~np.isnan(xold)]
    return xold, yold

def get_ylimits_roi(clasees_list, msk, offset=5):
    mask_opl = np.where(msk == clasees_list.index('OPL'), 1, 0)
    pos_opl = np.where(mask_opl)
    mask_bm = np.where(msk == clasees_list.index('BM'), 1, 0)
    pos_bm = np.where(mask_bm)

    ymin = np.min(pos_opl[0][np.nonzero(pos_opl[0])])
    ymax = np.max(pos_bm[0][np.nonzero(pos_bm[0])])

    if ymin < offset:
        ymin = 0
    else:
        ymin = ymin-offset

    if (ymax + offset) > msk.shape[0]:
        ymax = msk.shape[0]
    else:
        ymax = ymax + offset
    return ymin, ymax

def get_max_peak(img, window_size=1):
    max1 = []
    int_prof_x = []
    size = window_size
    k = 0
    for i in range(size, img.shape[1], size):
        indices = (-img[:, i - size:i].reshape(-1)).argsort()[:1]
        row1 = (int)(indices[0] / size)
        col1 = indices[0] - (row1 * size)
        temp1 = row1, col1 + k
        k += size
        max1.append(temp1)
        window = img[:, i - size:i]
        matrix_mean = np.zeros((img.shape[0]))
        for j in range(window.shape[0]):
            matrix_mean[j] = window[j, :].mean()
        int_prof_x.append(matrix_mean)
    max1 = np.array(max1)
    try:
        x1 = max1[:, 1]
    except:
        x1 = 0
    try:
        y1 = max1[:, 0]
    except:
        y1 = 0
    y1 = [np.nan if x==0 else x for x in y1]
    return np.array(x1), np.array(y1)

def get_limit(binary_mask, side, offset=0):
    size = 1
    lim = []
    for i in range(size, binary_mask.shape[1], size):
        col = binary_mask[:, i - size:i]
        if 1 in col:
            if side == 'max':
                lim.append(np.max(np.where(col)[0]) + offset)
            if side == 'min':
                lim.append(np.min(np.where(col)[0]) + offset)
        else:
            lim.append(float('nan'))
    lim = np.array(lim)
    return lim

def get_layer_binary_mask(sample_pred, clasees_list, layer='EZ', offset=0):
    binary = np.where(sample_pred == clasees_list.index(layer), 1, 0)
    size = 1
    if offset > 0:
        for off in range(offset):
            for i in range(size, binary.shape[1], size):
                col = binary[:, i - size:i]
                if 1 in col:
                    place = np.max(np.where(col)[0])
                    binary[place, i - size:i] = 0
    return binary

def get_thickness(binary_image, scale): 
    size = 1
    thickness = []
    for i in range(size, binary_image.shape[1], size):
        col = binary_image[:, i - size:i]
        if 1 in col:
            thickness.append(np.max(np.where(col)[0]) * scale - np.min(np.where(col)[0]) * scale)
    
    thickness_nan = np.array(thickness).copy()

    # thickness_nan = thickness_nan[~np.isnan(thickness_nan)]
    # print(thickness_nan.shape, thickness_nan.max(0), thickness_nan.min())
    if not np.any(thickness_nan):
        thickness_mean = 0
        thickness_std = 0
    else:
        thickness_mean = np.nanmean(thickness_nan)
        thickness_std = np.nanstd(thickness_nan)
    # print(thickness_nan*1000)    
    return thickness_mean*1000, thickness_std*1000

def get_distance_in_mm(ref1, ref2, scale): 
    distance_in_mm = []
    for i in range(ref1.shape[0]):
        distance_in_mm.append(np.abs(ref1[i] * scale - ref2[i] * scale))
    
    distance_in_mm_nan = np.array(distance_in_mm).copy()

    distance_in_mm_nan = distance_in_mm_nan[~np.isnan(distance_in_mm_nan)]


    if not np.any(distance_in_mm_nan):
        distance_in_mm_mean = 0
        distance_in_mm_std = 0
    else:
        distance_in_mm_mean = np.mean(distance_in_mm_nan)
        distance_in_mm_std = np.std(distance_in_mm_nan)
        
    return distance_in_mm_mean*1000, distance_in_mm_std*1000

def get_area(binary_image):
    area_pixels = np.count_nonzero(binary_image == 1)
    return area_pixels

def get_rEZI(ref1, ref2):
    rezi = []
    for i in range(ref1.shape[0]):
        relative_diff = (np.abs((ref2[i] - ref1[i])) / ref1[i])
        rezi.append(relative_diff)
    
    rezi_nan = np.array(rezi).copy()

    rezi_nan = rezi_nan[~np.isnan(rezi_nan)]

    if not np.any(rezi_nan):
        rezi_mean = 0
        rezi_std = 0
    else:
        rezi_mean = np.mean(rezi_nan)
        rezi_std = np.std(rezi_nan)

    return rezi_mean, rezi_std

def get_total_variation(segmentation, beta):
    y1 = segmentation / 255.
    vari = 0.0
    local = 0.0
    locales = []
    for k in range(0, y1.shape[1], 2 * beta):
        sample = y1[:, k:k + 2 * beta]
        for j in range(sample.shape[1]):
            vari = np.abs(sample[1, j] - sample[0, j])
            for i in range(2, sample.shape[0]):
                dif = np.abs(sample[i, j] - sample[i - 1, j])
                vari += dif
            local = vari / sample.shape[0]
        locales.append(local)
    locales = np.array(locales)
    locales_nan = np.array(locales).copy()

    locales_nan = locales_nan[~np.isnan(locales_nan)]

    if not np.any(locales_nan):
        tv_mean = 0
        tv_std = 0
    else:
        tv_mean = np.mean(locales_nan)
        tv_std = np.std(locales_nan)
    return locales, tv_mean, tv_std 

def gray_gamma(img, gamma):
    gray = img / 255.
    out = np.array(gray ** gamma)
    out = 255*out
    return out.astype('uint8')

def tv_denoising(img, alpha):
    if alpha is not None:
        gray = img / 255.
        out = denoise_tv_chambolle(gray, weight=alpha)
        out = out * 255
    else:
        out = img
    return out.astype('uint8')

def denoising_1D_TV(Y, lamda):
    N = len(Y)
    X = np.zeros(N)

    k, k0, kz, kf = 0, 0, 0, 0
    vmin = Y[0] - lamda
    vmax = Y[0] + lamda
    umin = lamda
    umax = -lamda

    while k < N:
        
        if k == N - 1:
            X[k] = vmin + umin
            break
        
        if Y[k + 1] < vmin - lamda - umin:
            for i in range(k0, kf + 1):
                X[i] = vmin
            k, k0, kz, kf = kf + 1, kf + 1, kf + 1, kf + 1
            vmin = Y[k]
            vmax = Y[k] + 2 * lamda
            umin = lamda
            umax = -lamda
            
        elif Y[k + 1] > vmax + lamda - umax:
            for i in range(k0, kz + 1):
                X[i] = vmax
            k, k0, kz, kf = kz + 1, kz + 1, kz + 1, kz + 1
            vmin = Y[k] - 2 * lamda
            vmax = Y[k]
            umin = lamda
            umax = -lamda
            
        else:
            k += 1
            umin = umin + Y[k] - vmin
            umax = umax + Y[k] - vmax
            if umin >= lamda:
                vmin = vmin + (umin - lamda) * 1.0 / (k - k0 + 1)
                umin = lamda
                kf = k
            if umax <= -lamda:
                vmax = vmax + (umax + lamda) * 1.0 / (k - k0 + 1)
                umax = -lamda
                kz = k
                
        if k == N - 1:
            if umin < 0:
                for i in range(k0, kf + 1):
                    X[i] = vmin
                k, k0, kf = kf + 1, kf + 1, kf + 1
                vmin = Y[k]
                umin = lamda
                umax = Y[k] + lamda - vmax
                
            elif umax > 0:
                for i in range(k0, kz + 1):
                    X[i] = vmax
                k, k0, kz = kz + 1, kz + 1, kz + 1
                vmax = Y[k]
                umax = -lamda
                umin = Y[k] - lamda - vmin
                
            else:
                for i in range(k0, N):
                    X[i] = vmin + umin * 1.0 / (k - k0 + 1)
                break

    return X
