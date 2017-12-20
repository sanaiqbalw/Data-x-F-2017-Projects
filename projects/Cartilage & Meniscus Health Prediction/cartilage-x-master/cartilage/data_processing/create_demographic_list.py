{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Combine original batchlist with demographic data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_root = '/data/bigbone4/ciriondo/'\n",
    "\n",
    "batchlist_csv = os.path.join(data_root, 'clean_all_path.csv')  # change for server\n",
    "demographic_csv = os.path.join(data_root, 'demographic_data.csv')  # change for server\n",
    "\n",
    "sample_list_csv = os.path.join(data_root, 'sample_list.csv')  # change for server"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_batch = pd.read_csv(batchlist_csv)\n",
    "df_demo = pd.read_csv(demographic_csv)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_batch['patient'] = df_batch['mriFile'].apply(lambda f: os.path.splitext(os.path.basename(f))[0].split('_')[0])\n",
    "df_batch['file_id'] = df_batch['mriFile'].apply(lambda f: os.path.splitext(os.path.basename(f))[0])\n",
    "df_batch = df_batch.loc[df_batch['mriFile'].apply(lambda p: os.path.exists(p))]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.merge(df_batch, df_demo, how='left', on='patient')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.to_csv(sample_list_csv, index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['mriFile',\n",
       " 'segFile',\n",
       " 'mfcWorms',\n",
       " 'lfcWorms',\n",
       " 'mfcBME',\n",
       " 'lfcBME',\n",
       " 'patient',\n",
       " 'file_id',\n",
       " 'GENDER',\n",
       " 'AGE',\n",
       " 'BMI']"
      ]
     },
     "execution_count": 150,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "list(df.columns.values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>mriFile</th>\n",
       "      <th>segFile</th>\n",
       "      <th>mfcWorms</th>\n",
       "      <th>lfcWorms</th>\n",
       "      <th>mfcBME</th>\n",
       "      <th>lfcBME</th>\n",
       "      <th>patient</th>\n",
       "      <th>file_id</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>P052</td>\n",
       "      <td>P052_0_loaded</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>P052</td>\n",
       "      <td>P052_1_loaded</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>P052</td>\n",
       "      <td>P052_2_loaded</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>P052</td>\n",
       "      <td>P052_3_loaded</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>2.5</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>P122</td>\n",
       "      <td>P122_0_loaded</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                             mriFile  \\\n",
       "0  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "1  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "2  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "3  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "4  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "\n",
       "                                             segFile  mfcWorms  lfcWorms  \\\n",
       "0  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...       0.0       0.0   \n",
       "1  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...       0.0       0.0   \n",
       "2  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...       0.0       0.0   \n",
       "3  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...       0.0       0.0   \n",
       "4  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...       2.5       1.0   \n",
       "\n",
       "   mfcBME  lfcBME patient        file_id  \n",
       "0     0.0     0.0    P052  P052_0_loaded  \n",
       "1     0.0     0.0    P052  P052_1_loaded  \n",
       "2     0.0     0.0    P052  P052_2_loaded  \n",
       "3     0.0     0.0    P052  P052_3_loaded  \n",
       "4     0.0     0.0    P122  P122_0_loaded  "
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_root = '/data/bigbone4/ciriondo/'\n",
    "#data_root = os.getcwd()\n",
    "\n",
    "batchlist_csv = os.path.join(data_root, 'clean_all_path.csv') \n",
    "demographic_csv = os.path.join(data_root, 'demographic_data.csv') \n",
    "\n",
    "sample_list_csv = os.path.join(data_root, 'sample_list.csv')  # change for server\n",
    "\n",
    "# Load in dataframe\n",
    "df_batch = pd.read_csv(batchlist_csv)\n",
    "\n",
    "df_batch['patient'] = df_batch['mriFile'].apply(lambda f: os.path.splitext(os.path.basename(f))[0].split('_')[0])\n",
    "df_batch['file_id'] = df_batch['mriFile'].apply(lambda f: os.path.splitext(os.path.basename(f))[0])\n",
    "df_batch = df_batch.loc[df_batch['mriFile'].apply(lambda p: os.path.exists(p))]\n",
    "\n",
    "df_batch.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [],
   "source": [
    "# drop any rows with files that don't exist\n",
    "df_batch = df_batch.loc[df_batch['mriFile'].apply(lambda p: os.path.exists(p))]\n",
    "# df_batch = df_batch.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_worms = df_batch.copy().rename(columns={'mfcWorms': 'mfc', 'lfcWorms': 'lfc'}).drop(['lfcBME', 'mfcBME'], axis=1)\n",
    "df_bme = df_batch.copy().rename(columns={'mfcBME': 'mfc', 'lfcBME': 'lfc'}).drop(['lfcWorms', 'mfcWorms'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [],
   "source": [
    "# melting LFC/MFC into seperate rows\n",
    "df_worms_melt = pd.melt(df_worms, \n",
    "                 id_vars=['patient', 'file_id', 'mriFile', 'segFile'], \n",
    "                 value_vars=['lfc', 'mfc'],\n",
    "                 var_name='type',\n",
    "                 value_name='worms')\n",
    "\n",
    "df_bme_melt = pd.melt(df_bme, \n",
    "                 id_vars=['patient', 'file_id', 'mriFile', 'segFile'], \n",
    "                 value_vars=['lfc', 'mfc'],\n",
    "                 var_name='type',\n",
    "                 value_name='bme')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3174, 6)"
      ]
     },
     "execution_count": 126,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_bme_melt.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3174, 6)"
      ]
     },
     "execution_count": 127,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_worms_melt.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3178, 7)"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_scores = pd.merge(df_worms_melt, df_bme_melt, \n",
    "                  how='inner', \n",
    "                  on=['patient', 'file_id', 'type', 'mriFile', 'segFile'])\n",
    "df_scores.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>patient</th>\n",
       "      <th>file_id</th>\n",
       "      <th>mriFile</th>\n",
       "      <th>segFile</th>\n",
       "      <th>type</th>\n",
       "      <th>worms</th>\n",
       "      <th>bme</th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>BMI</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>P052</td>\n",
       "      <td>P052_0_loaded</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>lfc</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>F</td>\n",
       "      <td>49.0</td>\n",
       "      <td>26.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>P052</td>\n",
       "      <td>P052_1_loaded</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>lfc</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>F</td>\n",
       "      <td>49.0</td>\n",
       "      <td>26.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>P052</td>\n",
       "      <td>P052_2_loaded</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>lfc</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>F</td>\n",
       "      <td>49.0</td>\n",
       "      <td>26.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>P052</td>\n",
       "      <td>P052_3_loaded</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>lfc</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>F</td>\n",
       "      <td>49.0</td>\n",
       "      <td>26.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>P122</td>\n",
       "      <td>P122_0_loaded</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>lfc</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>F</td>\n",
       "      <td>62.0</td>\n",
       "      <td>26.7</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  patient        file_id                                            mriFile  \\\n",
       "0    P052  P052_0_loaded  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "1    P052  P052_1_loaded  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "2    P052  P052_2_loaded  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "3    P052  P052_3_loaded  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "4    P122  P122_0_loaded  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "\n",
       "                                             segFile type  worms  bme GENDER  \\\n",
       "0  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...  lfc    0.0  0.0      F   \n",
       "1  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...  lfc    0.0  0.0      F   \n",
       "2  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...  lfc    0.0  0.0      F   \n",
       "3  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...  lfc    0.0  0.0      F   \n",
       "4  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...  lfc    1.0  0.0      F   \n",
       "\n",
       "    AGE   BMI  \n",
       "0  49.0  26.3  \n",
       "1  49.0  26.3  \n",
       "2  49.0  26.3  \n",
       "3  49.0  26.3  \n",
       "4  62.0  26.7  "
      ]
     },
     "execution_count": 129,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_demo = pd.read_csv(demographic_csv)\n",
    "df = pd.merge(df_scores, df_demo, how='left', on='patient')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "patient    465\n",
       "file_id    465\n",
       "mriFile    465\n",
       "segFile    465\n",
       "type       465\n",
       "worms      465\n",
       "bme        464\n",
       "GENDER     465\n",
       "AGE        465\n",
       "BMI        438\n",
       "dtype: int64"
      ]
     },
     "execution_count": 132,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df['worms'] > 1].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.to_csv(os.path.join(data_root, 'full_sample_list.csv'), index=False)\n",
    "# df_valid.to_csv(os.path.join(data_root, 'valid_batchlist.csv'), index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_root = '/data/bigbone4/ciriondo/'\n",
    "#data_root = os.getcwd()\n",
    "\n",
    "batchlist_csv = os.path.join(data_root, 'clean_all_path.csv') \n",
    "demographic_csv = os.path.join(data_root, 'demographic_data.csv') \n",
    "\n",
    "sample_list_csv = os.path.join(data_root, 'full_batchlist.csv')  # change for server\n",
    "\n",
    "# Load in dataframe\n",
    "df_batch = pd.read_csv(batchlist_csv)\n",
    "\n",
    "df_batch['patient'] = df_batch['mriFile'].apply(lambda f: os.path.splitext(os.path.basename(f))[0].split('_')[0])\n",
    "df_batch['file_id'] = df_batch['mriFile'].apply(lambda f: os.path.splitext(os.path.basename(f))[0])\n",
    "\n",
    "df_batch = df_batch.loc[df_batch['mriFile'].apply(lambda p: os.path.exists(p))]\n",
    "\n",
    "df_worms = df_batch.copy().rename(columns={'mfcWorms': 'mfc', 'lfcWorms': 'lfc'}).drop(['lfcBME', 'mfcBME'], axis=1)\n",
    "df_bme = df_batch.copy().rename(columns={'mfcBME': 'mfc', 'lfcBME': 'lfc'}).drop(['lfcWorms', 'mfcWorms'], axis=1)\n",
    "\n",
    "# melting LFC/MFC into seperate rows\n",
    "df_worms_melt = pd.melt(df_worms, \n",
    "                 id_vars=['patient', 'file_id', 'mriFile', 'segFile'], \n",
    "                 value_vars=['lfc', 'mfc'],\n",
    "                 var_name='type',\n",
    "                 value_name='worms')\n",
    "\n",
    "df_bme_melt = pd.melt(df_bme, \n",
    "                 id_vars=['patient', 'file_id', 'mriFile', 'segFile'], \n",
    "                 value_vars=['lfc', 'mfc'],\n",
    "                 var_name='type',\n",
    "                 value_name='bme')\n",
    "\n",
    "df_scores = pd.merge(df_worms_melt, df_bme_melt, \n",
    "                  how='inner', \n",
    "                  on=['patient', 'file_id', 'type', 'mriFile', 'segFile'])\n",
    "\n",
    "df_demo = pd.read_csv(demographic_csv)\n",
    "df = pd.merge(df_scores, df_demo, how='left', on='patient')\n",
    "\n",
    "df.to_csv(os.path.join(data_root, 'tfrecord_list.csv'), index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3178, 10)"
      ]
     },
     "execution_count": 113,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[None:None].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [],
   "source": [
    "save_dir = 'test/'\n",
    "df['sampleFile'] = save_dir + df['file_id'] + df['type'] + '.tfrecord'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>patient</th>\n",
       "      <th>file_id</th>\n",
       "      <th>mriFile</th>\n",
       "      <th>segFile</th>\n",
       "      <th>type</th>\n",
       "      <th>worms</th>\n",
       "      <th>bme</th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>BMI</th>\n",
       "      <th>sampleFile</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>P052</td>\n",
       "      <td>P052_0_loaded</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>lfc</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>F</td>\n",
       "      <td>49.0</td>\n",
       "      <td>26.3</td>\n",
       "      <td>test/P052_0_loadedlfc.tfrecord</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>P052</td>\n",
       "      <td>P052_1_loaded</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>lfc</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>F</td>\n",
       "      <td>49.0</td>\n",
       "      <td>26.3</td>\n",
       "      <td>test/P052_1_loadedlfc.tfrecord</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>P052</td>\n",
       "      <td>P052_2_loaded</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>lfc</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>F</td>\n",
       "      <td>49.0</td>\n",
       "      <td>26.3</td>\n",
       "      <td>test/P052_2_loadedlfc.tfrecord</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>P052</td>\n",
       "      <td>P052_3_loaded</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>lfc</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>F</td>\n",
       "      <td>49.0</td>\n",
       "      <td>26.3</td>\n",
       "      <td>test/P052_3_loadedlfc.tfrecord</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>P122</td>\n",
       "      <td>P122_0_loaded</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>/data/bigbone4/DeepLearning_temp/Data/all_CUBE...</td>\n",
       "      <td>lfc</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>F</td>\n",
       "      <td>62.0</td>\n",
       "      <td>26.7</td>\n",
       "      <td>test/P122_0_loadedlfc.tfrecord</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  patient        file_id                                            mriFile  \\\n",
       "0    P052  P052_0_loaded  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "1    P052  P052_1_loaded  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "2    P052  P052_2_loaded  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "3    P052  P052_3_loaded  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "4    P122  P122_0_loaded  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...   \n",
       "\n",
       "                                             segFile type  worms  bme GENDER  \\\n",
       "0  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...  lfc    0.0  0.0      F   \n",
       "1  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...  lfc    0.0  0.0      F   \n",
       "2  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...  lfc    0.0  0.0      F   \n",
       "3  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...  lfc    0.0  0.0      F   \n",
       "4  /data/bigbone4/DeepLearning_temp/Data/all_CUBE...  lfc    1.0  0.0      F   \n",
       "\n",
       "    AGE   BMI                      sampleFile  \n",
       "0  49.0  26.3  test/P052_0_loadedlfc.tfrecord  \n",
       "1  49.0  26.3  test/P052_1_loadedlfc.tfrecord  \n",
       "2  49.0  26.3  test/P052_2_loadedlfc.tfrecord  \n",
       "3  49.0  26.3  test/P052_3_loadedlfc.tfrecord  \n",
       "4  62.0  26.7  test/P122_0_loadedlfc.tfrecord  "
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
