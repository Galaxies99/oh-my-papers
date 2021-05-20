import os
import yaml
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'data_preparation.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

if 'conf_list' not in cfg_dict.keys():
    raise AttributeError('The conference list is left unspecified in configuration file.')
conf_list = cfg_dict['conf_list']

if 'science_parse_jar_path' not in cfg_dict.keys():
    raise AttributeError('The path to the science parse jar file is left unspecified in configuration file.')
sp_jar = cfg_dict['science_parse_jar_path']

if 'raw_data_folder' not in cfg_dict.keys():
    raise AttributeError('The raw data folder is left unspecified in configuration file.')
raw_data_folder = cfg_dict['raw_data_folder']

if 'paper_data_folder' not in cfg_dict.keys():
    raise AttributeError('The paper data folder is left unspecified in configuration file.')
paper_data_folder = cfg_dict['paper_data_folder']


if os.path.exists(paper_data_folder) is False:
    raise AttributeError('Invalid paper data directory.')
if os.path.exists(raw_data_folder) is False:
    os.makedirs(raw_data_folder)


for conf in conf_list:
    paper_folder = os.path.join(paper_data_folder, conf)
    paper_folder = paper_folder + '/'
    raw_folder = os.path.join(raw_data_folder, conf)
    if os.path.exists(raw_folder) is False:
        os.makedirs(raw_folder)
    raw_folder = raw_folder + '/'
    os.system('java -Xmx6g -jar {} {} -o {}'.format(sp_jar, paper_folder, raw_folder))