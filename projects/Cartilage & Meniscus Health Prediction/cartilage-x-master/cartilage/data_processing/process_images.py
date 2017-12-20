"""
Convert matlab files to tfrecord format with all metadata
"""

import os
import sys
import time
import itertools

import cv2
import numpy as np
import pandas as pd
import scipy.io as spio

from sklearn.utils import shuffle
from six.moves import cPickle as pickle


# ### Batch loading and saving
# We store the WORMS score rather than label {0, 1} since we might want to mix that up once the data's saved.


def create_sample_list(df_batch, save_dir, format_):
    """Return sample list with directories of processed files"""

    # some redundancy between scripts but this is the best approach
    worms_columns = [
        'file_id',
        'mriFile',
        'segFile',
        'mfcWorms',
        'lfcWorms',
        'patient'
    ]

    bme_columns = [
        'file_id',
        'mriFile',
        'segFile',
        'mfcBME',
        'lfcBME',
        'patient'
    ]

    demo_columns = [
        'patient',
        'GENDER',
        'AGE',
        'BMI'
    ]

    # split up worms and bme scoring
    df_worms = df_batch[worms_columns].rename(columns={'mfcWorms': 'mfc', 'lfcWorms': 'lfc'})
    df_bme = df_batch[bme_columns].rename(columns={'mfcBME': 'mfc', 'lfcBME': 'lfc'})
    df_demo = df_batch[demo_columns].drop_duplicates()

    # melting LFC/MFC into seperate rows
    df_worms_melt = pd.melt(df_worms,
                            id_vars=['patient', 'file_id', 'mriFile', 'segFile'],
                            value_vars=['lfc', 'mfc'],
                            var_name='type',
                            value_name='worms')
    df_bme_melt = pd.melt(df_bme,
                          id_vars=['patient', 'file_id', 'mriFile', 'segFile'],
                          value_vars=['lfc', 'mfc'],
                          var_name='type',
                          value_name='bme')

    # combining scores again
    df_scores = pd.merge(df_worms_melt, df_bme_melt,
                         how='inner',
                         on=['patient', 'file_id', 'type', 'mriFile', 'segFile'])

    # adding back demographic information
    df = pd.merge(df_scores, df_demo, how='left', on='patient')

    # adding a filename column
    df['sample_path'] = save_dir + '/' + df['file_id'].astype(str) + '_' + df['type'].astype(str) + '.' + format_

    return df


def save_pickle(data_dict, filepath):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print("Unable to save pickle file: ", filepath, ": ", e)
        raise


def bbox3d_coord(img):
    """Gets edge coordinates for the 3D area with non-zero elements"""
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    ymin, ymax = np.where(r)[0][[0, -1]]
    xmin, xmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return (ymin, ymax), (xmin, xmax), (zmin, zmax)


def crop_to_bbox(image, seg_img, c_type):
    """Crops MRI image to bounding box using segementation"""
    seg_img = np.squeeze(seg_img*(seg_img == 1)) if c_type == 'lfc' else np.squeeze(seg_img*(seg_img == 2))
    b = bbox3d_coord(seg_img)
    return image[b[0][0]:b[0][1], b[1][0]:b[1][1], b[2][0]:b[2][1]]


def resize_volume(image, target_dims):
    """Crops image to bounding box then resizes to target volume"""
    # TODO: using numpy `apply_along_axis` to (potentially) speed up

    # interpolate along depth (z-axis)
    xy_resized = np.zeros(shape=[target_dims[0], target_dims[1], image.shape[2]],
                          dtype=np.float32)
    for i in range(image.shape[2]):
        xy_resized[:, :, i] = cv2.resize(image[:, :, i],
                                         dsize=(target_dims[0], target_dims[1]),
                                         interpolation=cv2.INTER_AREA)
    # interpolate along x-axis
    resized = np.zeros(shape=[target_dims[0], target_dims[1], target_dims[2]],
                       dtype=np.float32)
    for i in range(target_dims[0]):
        resized[i, :, :] = cv2.resize(xy_resized[i, :, :],
                                      dsize=(target_dims[2], target_dims[1]),
                                      interpolation=cv2.INTER_AREA)
    return resized


def process_images(df_batch, save_dir, range_=(None, None), target_dims=None, format_='tfrecord'):
    """Load, crop, and resize matlab files in target folder in tfrecord format"""

    # get ranges
    start = range_[0] if range_[0] is not None else 0
    end = range_[1] if range_[1] is not None else df_batch.shape[0]
    num_samples = end - start

    df_range = df_batch[start:end]

    df_sample = create_sample_list(df_batch=df_range, save_dir=save_dir, format_=format_)

    print('\nRange {}:{} saving to: {}'.format(start, end, save_dir))
    t0 = time.time()

    for i, sample in enumerate(df_range.iterrows()):

        ti = time.time()

        # identifier
        file_id = sample[1]['file_id']

        # demographic
        age = sample[1]['AGE']
        gender = sample[1]['GENDER']
        bmi = sample[1]['BMI']

        # file directories
        mri_file = sample[1]['mriFile']
        seg_file = sample[1]['segFile']

        # labels
        worms_lfc = sample[1]['lfcWorms']
        bme_lfc = sample[1]['lfcBME']

        worms_mfc = sample[1]['mfcWorms']
        bme_mfc = sample[1]['mfcBME']

        # load matlab images
        t_mat0 = time.time()
        try:
            mat_data = spio.loadmat(mri_file, squeeze_me=True)

            # check for either possible key
            if 'im_store' in mat_data:
                mri_img = mat_data['im_store'].astype(np.float32)
            elif 'full_im' in mat_data:
                mri_img = mat_data['full_im'].astype(np.float32)
            else:
                raise KeyError('Keys "im_store" and "full_im" not in file {}'.format(mri_file))

            seg_img = spio.loadmat(seg_file, squeeze_me=True)['pred_con_vol'].astype(np.float32)

        # make sure it doesn't get trapped
        except (KeyboardInterrupt, SystemExit):
            raise

        except Exception:
            print('\tSkipping image {} of {}, error loading..                   '.format(i, num_samples), end='\r')
            with open('load_error_log_2.txt', 'a') as log:
                log.write(mri_file + '\n')
            continue

        # identify bad files by depth of less than 120
        if mri_img.shape[2] < 120:
            print('\tSkipping image {} of {}, low depth                     '.format(i, num_samples), end='\r')
            with open('depth_error_log_2.txt', 'a') as log:
                log.write(mri_file + '\n')
            continue
        t_mat = time.time() - t_mat0

        # crop and resize volume
        # LFC
        t_resize0 = time.time()
        try:
            image_lfc = crop_to_bbox(mri_img, seg_img, c_type='lfc')
            image_lfc = resize_volume(image_lfc, dims)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print('\tSkipping image {} of {}, crop/resize error                    '.format(i, num_samples),
                  end='\r')
            with open('resize_error_log.txt', 'a') as log:
                log.write(mri_file + '\n')
            continue

        # MFC
        try:
            image_mfc = crop_to_bbox(mri_img, seg_img, c_type='mfc')
            image_mfc = resize_volume(image_mfc, dims)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print('\tSkipping image {} of {}, crop/resize error lfc                    '.format(i, num_samples),
                  end='\r')
            with open('resize_error_log.txt', 'a') as log:
                log.write(mri_file + '\n')
            continue

        t_resize = time.time() - t_resize0

        # save array
        lfc_sample_path = str(df_sample[df_sample['type'] == 'lfc'].loc[df_sample['file_id'] == file_id, 'sample_path'].values[0])
        mfc_sample_path = str(df_sample[df_sample['type'] == 'mfc'].loc[df_sample['file_id'] == file_id, 'sample_path'].values[0])
     
        dat_lfc = {
            'image': image_lfc,
            'worms': worms_lfc,
            'bme': bme_lfc,
            'age': age,
            'gender': gender,
            'bmi': bmi
        }

        dat_mfc = {
            'image': image_mfc,
            'worms': worms_mfc,
            'bme': bme_mfc,
            'age': age,
            'gender': gender,
            'bmi': bmi
        }

        t_save0 = time.time()

        if format_ == 'pickle':
            save_pickle(dat_lfc, lfc_sample_path)
            save_pickle(dat_mfc, mfc_sample_path)
        else:
            pass

        t_save = time.time() - t_save0

        t1 = time.time()
        t_sample = t1 - ti
        t_dataset = t1 - t0
        # t1 = time.time()
        #         print('saved to {}'.format(save_filename), end='\r')#(t1-t0)*1000)
        print(
            '\tImage {} of {} saved, sample {:.0f}ms, load {:.0f}ms, resize {:.0f}ms, save {:.0f}ms, dataset: {:.2f}s'.
            format(i+1,
                   num_samples,
                   t_sample * 1000,
                   t_mat * 1000,
                   t_resize * 1000,
                   t_save * 1000,
                   t_dataset), end='\r')
    print('')
    return df_sample

if __name__ == '__main__':

    sample_start = int(sys.argv[1]) if len(sys.argv) == 3 else None
    sample_end = int(sys.argv[2]) if len(sys.argv) == 3 else None

    data_root = '/data/bigbone4/ciriondo/'

    save_dir = os.path.join(data_root, 'pickle_data_combined')
    sample_list_pre_csv = os.path.join(data_root, 'sample_list.csv')

    df_sample_pre = pd.read_csv(sample_list_pre_csv)

    # process actual images
    sample_range = (sample_start, sample_end)
    dims = [120, 120, 60]

    # run processing function for train and validation datasets
    df_sample_post = process_images(df_sample_pre, save_dir=save_dir, target_dims=dims, range_=sample_range, format_='pickle')

    sample_list_post_csv = os.path.join(data_root, 'sample_lists/sample_list_post_{}-{}.csv'.format(sample_range[0], sample_range[1]))
    df_sample_post.to_csv(sample_list_post_csv, index=False)
