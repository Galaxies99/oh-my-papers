import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

def split_process_dataset(file_path, seq_len, year, frequency=5):
    raw_data = pd.read_csv(file_path, index_col=False, dtype='object', engine='python')

    raw_data.rename(columns={
        'src_id': 'source_id',
        'src_title': 'source_title',
        'src_author': 'source_author',
        'src_venue': 'source_venue',
        'src_year': 'source_year',
        'src_abstract': 'source_abstract',
        'left_context': 'left_citated_text',
        'right_context': 'right_citated_text',
        'ref_id': 'target_id',
        'ref_title': 'target_title',
        'ref_author': 'target_author',
        'ref_venue': 'target_venue',
        'ref_year': 'target_year',
        'ref_abstract': 'target_abstract'
    }, inplace=True)

    data = _convert_df_type(raw_data, ['source_id', 'target_id', 'source_year', 'target_year'], type=int)
    data = _cut_off_dataset(data, year, frequency)
    data = _slicing_citation_text(data, seq_len)
    train_df, test_df = _split_dataset(data, year)

    return train_df, test_df

def construct_graph(df):
    source_ids = df['SourceID'].values.tolist()
    target_ids = df['TargetID'].values.tolist()
    ids = list(set(list(source_ids + target_ids)))
    source_titles = df['source_title'].values.tolist()
    target_titles = df['target_title'].values.tolist()
    source_abstracts = df['source_abstract'].values.tolist()
    target_abstracts = df['target_abstract'].values.tolist()

    edge_lists = [[src, tar] for src, tar in zip(source_ids, target_ids)]
    node_info = [{} for _ in range(len(ids))]
    for i, (src, tar) in enumerate(zip(source_ids, target_ids)):
        if not node_info[src]:
            node_info[src]['title'] = source_titles[i]
            node_info[src]['abstract'] = source_abstracts[i]
        if not node_info[tar]:
            node_info[tar]['title'] = target_titles[i]
            node_info[tar]['abstract'] = target_abstracts[i]

    return edge_lists, node_info

def _convert_df_type(df, columns, type):
    if not isinstance(columns, list):
        columns = [columns]
    for column in columns:
        df[[column]] = df[[column]].astype(type)

    return df

def _cut_off_dataset(df, year, frequency=5):
    # remove items with source_year > year
    idx_1 = set(df['source_year'][df['source_year'] < year].index)
    idx_2 = set(df['source_year'][df['source_year'] == year].index) & set(df['target_year'][df['target_year'] < year].index)
    df = df.loc[idx_1 | idx_2]

    # make sure every source paper has more than 5 out links
    target_cut_data = df[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
    target_cut = target_cut_data.target_id.value_counts()[(target_cut_data.target_id.value_counts() >= frequency)]
    target_id = np.sort(target_cut.keys())
    df = df.loc[df['target_id'].isin(target_id)]

    return df

def _slicing_citation_text(df, number):
    # slicing citaiton context
    df['LeftString'] = df['left_citated_text'].str[-number:]
    df['RightString'] = df['right_citated_text'].str[:number]

    return df

def _split_dataset(df, year):

    train_idx = df['source_year'][df['source_year'] < year].index
    test_idx = df['source_year'][df['source_year'] == year].index

    def _get_ids(df, idx):
        df = df.loc[idx]
        source_ids = df['source_id'].values.tolist()
        target_ids = df['target_id'].values.tolist()
        ids = set(source_ids + target_ids)

        return ids

    # renumber the id
    train_node_ids = _get_ids(df, train_idx)
    test_node_ids = _get_ids(df, test_idx)
    additional_node_ids = test_node_ids - train_node_ids
    train_nodes_mapping = dict(zip(list(train_node_ids), range(len(train_node_ids))))
    additional_nodes_mapping = dict(zip(list(additional_node_ids), range(len(train_node_ids), len(train_node_ids) + len(additional_node_ids))))
    mapping = {**train_nodes_mapping, **additional_nodes_mapping}

    new_src_id, new_tar_id = [], []
    for src, tar in zip(df['source_id'], df['target_id']):
        new_src_id.append(mapping[src])
        new_tar_id.append(mapping[tar])
    df['SourceID'] = new_src_id
    df['TargetID'] = new_tar_id
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = split_process_dataset(file_path='../data/citation.csv', seq_len=50, year=2015, frequency=5)

    # construct graph
    edge_list, node_info = construct_graph(train_df)

    # construct dataset for bert



