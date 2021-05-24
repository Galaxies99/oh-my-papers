import os
import yaml
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'data_preparation.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

if 'data_folder' not in cfg_dict.keys():
    raise AttributeError('The data folder is left unspecified in configuration file.')
data_folder = cfg_dict['data_folder']

if 'data_name' not in cfg_dict.keys():
    raise AttributeError('The data name is left unspecified in configuration file.')
data_name = cfg_dict['data_name']

if 'tiny_data_name' not in cfg_dict.keys():
    raise AttributeError('The tiny data name is left unspecified in configuration file.')
tiny_data_name = cfg_dict['tiny_data_name']

if 'tiny_data_year_threshold' not in cfg_dict.keys():
    raise AttributeError('The tiny data year threshold is left unspecified in configuration file.')
tiny_data_year_threshold = cfg_dict['tiny_data_year_threshold']

df = pd.read_csv(os.path.join(data_folder, data_name + '.csv'), index_col = 'index', engine = 'python')
tiny_df = df[df.src_year <= tiny_data_year_threshold]
tiny_df.to_csv(os.path.join(data_folder, tiny_data_name + '.csv'))

print('*** Summary Statistics ***')
print('# of reference items：{}'.format(tiny_df.shape[0]))