import os
import json
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
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

if 'raw_data_folder' not in cfg_dict.keys():
    raise AttributeError('The raw data folder is left unspecified in configuration file.')
raw_data_folder = cfg_dict['raw_data_folder']

if 'data_folder' not in cfg_dict.keys():
    raise AttributeError('The data folder is left unspecified in configuration file.')
data_folder = cfg_dict['data_folder']

if 'data_name' not in cfg_dict.keys():
    raise AttributeError('The data name is left unspecified in configuration file.')
data_name = cfg_dict['data_name']


def conf_to_venue_year_simple(conf_name):
    venue = conf_name[:4]
    year = int(conf_name[4:]) + 2000
    return venue, year


def author_list_to_name(author_list):
    author_name = ''
    for author in author_list:
        if author_name is None:
            return ''
        if author_name != '':
            author_name = author_name + ', '
        author_name = author_name + author.encode('utf-8', 'replace').decode('utf-8')
    return author_name


def format_venue(venue):
    if venue is None:
        return ''
    
    venue = format_str(venue)
    venue_lower = venue.lower()
    if 'arxiv' in venue_lower:
        return 'arxiv'
    venue_cap = ''
    for c in venue:
        if 'A' <= c <= 'Z':
            venue_cap = venue_cap + c
    
    if 'CVPR' in venue_cap:
        return 'CVPR'
    if 'ICCV' in venue_cap:
        return 'ICCV'
    if 'ECCV' in venue_cap:
        return 'ECCV'
    if 'NIPS' in venue_cap:
        return 'NIPS'
    if 'AAAI' in venue_cap:
        return 'AAAI'
    if 'ICML' in venue_cap:
        return 'ICML'
    
    return ''
    

def unique_paper_identifier(paper_name, venue, year):
    id = ""
    for c in paper_name:
        if 'a' <= c <= 'z':
            id += c
        elif 'A' <= c <= 'Z':
            id += c.lower()
    id = id + venue + str(year)
    return id


def format_citations(pre_citation_list, pre_pd_citation_list, abstract_dict):
    citation_list = []
    pd_citation_list = []
    for i, c in enumerate(pre_citation_list):
        ref_id = c['ref_id']
        if ref_id in abstract_dict.keys():
            citation_dict = pre_citation_list[i]
            citation_info = pre_pd_citation_list[i]
            citation_dict['ref_abstract'] = abstract_dict[ref_id]
            citation_info[-1] = abstract_dict[ref_id]
            citation_list.append(citation_dict)
            pd_citation_list.append(citation_info)
    return citation_list, pd_citation_list


def format_str(s):
    s = str(s).lstrip().rstrip()
    return s.replace('\0', '').replace('\n', '').replace('\r', '').replace('\t', '')


def process_raw_citation(input_dir, conf_list, output_dir, dataset_name):
    if os.path.exists(input_dir) == False:
        raise AttributeError('Invalid input directory!')
    
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    paper_identifier_ID = {}
    paper_ID_cnt = 1
    citation_list = []
    pd_citation_list = []
    abstract_dict = {}

    for conf in conf_list:
        print('--> Processing {} papers'.format(conf))
        conf_folder = os.path.join(input_dir, conf)
        if os.path.exists(conf_folder) == False:
            raise AttributeError('Invalid conference directory!')
        papers = os.listdir(conf_folder)
        for paper in tqdm(papers):
            _, ext = os.path.splitext(paper)
            if ext != '.json':
                continue
            with open(os.path.join(conf_folder, paper), 'r') as f:
                paper_dict = json.load(f)
            meta_data = paper_dict.get('metadata', {})
            if meta_data == {}:
                continue
            
            # Source paper data: paper_name, author_name, venue, year, abstract
            paper_name = meta_data.get('title', '')
            author_name = author_list_to_name(meta_data.get('authors', []))
            venue, year = conf_to_venue_year_simple(conf)
            abstract = meta_data.get('abstractText', '')
            if paper_name is None or abstract is None:
                continue
            paper_name = paper_name.encode('utf-8', 'replace').decode('utf-8')
            abstract = abstract.encode('utf-8', 'replace').decode('utf-8')
            paper_name = format_str(paper_name)
            author_name = format_str(author_name)
            venue = format_str(venue)
            year = int(year)
            abstract = format_str(abstract)
            if paper_name == '' or author_name == '' or abstract == '':
                continue
            src_paper_id = unique_paper_identifier(paper_name, venue, year)
            src_ID = paper_identifier_ID.get(src_paper_id, paper_ID_cnt)
            src_ID = int(src_ID)
            if src_ID == paper_ID_cnt:
                paper_identifier_ID[src_paper_id] = src_ID
                paper_ID_cnt += 1
            if src_ID not in abstract_dict.keys():
                abstract_dict[src_ID] = abstract
            
            # Reference data: left_context, right_context, ref_paper_name, ref_author_name, ref_venue, ref_year
            references = meta_data.get('references', [])
            ref_mentions = meta_data.get('referenceMentions', [])
            for ref_info in ref_mentions:
                ref_id = ref_info.get('referenceID', -1)
                context = ref_info.get('context', '')
                start_offset = ref_info.get('startOffset', -1)
                end_offset = ref_info.get('endOffset', -1)
                if context is None or ref_id is None or start_offset is None or end_offset is None:
                    continue
                context = context.encode('utf-8', 'replace').decode('utf-8')
                context = format_str(context)
                if ref_id == -1 or context == '' or start_offset == -1 or end_offset == -1:
                    continue
                left_context = context[:start_offset]
                right_context = context[end_offset:]
                # Prevent NaN in DataFrame.
                if left_context == "":
                    left_context = " "
                if right_context == "":
                    right_context = " "
                if ref_id >= len(references):
                    continue
                ref_paper = references[ref_id]
                ref_paper_name = ref_paper.get('title', '')
                ref_author_name = author_list_to_name(ref_paper.get('author', []))
                ref_venue = format_venue(ref_paper.get('venue', ''))
                ref_year = ref_paper.get('year', -1)
                if ref_paper_name is None or ref_venue is None or ref_year is None:
                    continue
                ref_paper_name = ref_paper_name.encode('utf-8', 'replace').decode('utf-8')
                ref_paper_name = format_str(ref_paper_name)
                ref_author_name = format_str(ref_author_name)
                ref_venue = format_str(ref_venue)
                ref_year = int(ref_year)
                if ref_paper_name == '' or ref_author_name == '' or ref_venue == '' or ref_year == -1:
                    continue
                ref_paper_id = unique_paper_identifier(ref_paper_name, ref_venue, ref_year)
                ref_ID = paper_identifier_ID.get(ref_paper_id, paper_ID_cnt)
                if ref_ID == paper_ID_cnt:
                    paper_identifier_ID[ref_paper_id] = ref_ID
                    paper_ID_cnt += 1

                # Citation dict:
                citation_dict = {
                    'src_id': src_ID,
                    'src_title': paper_name,
                    'src_author': author_name,
                    'src_venue': venue,
                    'src_year': year,
                    'src_abstract': abstract,
                    'left_context': left_context,
                    'right_context': right_context,
                    'ref_id': ref_ID,
                    'ref_title': ref_paper_name,
                    'ref_author': ref_author_name,
                    'ref_venue': ref_venue,
                    'ref_year': ref_year,
                    'ref_abstract': ''
                }

                citation_list.append(citation_dict)
                pd_citation_list.append([
                    src_ID, paper_name, author_name, venue, year, abstract, left_context, right_context, 
                    ref_ID, ref_paper_name, ref_author_name, ref_venue, ref_year, ""
                ])
    
    citation_list, pd_citation_list = format_citations(citation_list, pd_citation_list, abstract_dict)

    full_citation_dict = {
        'citations': citation_list,
        'paper_identifier_dict': paper_identifier_ID,
        'paper_cnt': paper_ID_cnt - 1,
    }

    with open(os.path.join(output_dir, dataset_name + '.json'), 'w') as f:
        json.dump(full_citation_dict, f)
    
    column_name = [
        'src_id', 'src_title', 'src_author', 'src_venue', 'src_year', 'src_abstract', 'left_context', 
        'right_context', 'ref_id', 'ref_title', 'ref_author', 'ref_venue', 'ref_year', 'ref_abstract'
    ]
    df = pd.DataFrame(pd_citation_list, columns = column_name)
    df.index.name = 'index'
    df.to_csv(os.path.join(output_dir, dataset_name + '.csv'))
    print('*** Summary Statistics ***')
    print('# of papers: {}'.format(paper_ID_cnt - 1))
    print('# of reference items：{}'.format(df.shape[0]))


if __name__ == '__main__':
    process_raw_citation(raw_data_folder, conf_list, data_folder, data_name)
