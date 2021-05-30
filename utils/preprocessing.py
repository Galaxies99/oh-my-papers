import os
import numpy as np
import pandas as pd

def split_process_dataset(file_path, seq_len, year, frequency=5):
    '''
    Preprocess input data and split it into train_df and test_df

    Parameters
    ----------
    file_path: path of the whole data
    seq_len: maximal length of the citation context
    year: year of boundary (training and test)
    frequency: only articles that are referenced more than 'frequency' are retained
    '''
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
    train_df, test_df, whole_df, graph_node_id_threshold = _split_dataset(data, year)
    ground_truth = _get_ground_truth(whole_df, graph_node_id_threshold)

    return train_df, test_df, ground_truth, whole_df, graph_node_id_threshold

def construct_graph(train_df, whole_df, graph_node_id_threshold):
    '''
    Construct the graph based on the input dataframe

    Parameters
    ----------
    df: input dataframe

    Returns
    -------
    edge_lists (list) : the list of edges (size: n * 2)
    node_info (list of dict): contains every paper's title and abstract
    '''
    df = whole_df
    source_ids = df['SourceID'].values.tolist()
    target_ids = df['TargetID'].values.tolist()
    ids = list(set(list(source_ids + target_ids)))
    source_titles = df['source_title'].values.tolist()
    target_titles = df['target_title'].values.tolist()
    source_abstracts = df['source_abstract'].values.tolist()
    target_abstracts = df['target_abstract'].values.tolist()
    source_years = df['source_year'].values.tolist()
    target_years = df['target_year'].values.tolist()
    source_venues = df['source_venue'].values.tolist()
    target_venues = df['target_venue'].values.tolist()
    source_authors = df['source_author'].values.tolist()
    target_authors = df['target_author'].values.tolist()

    edge_lists = [[src, tar] for src, tar in zip(train_df['SourceID'].values.tolist(), train_df['TargetID'].values.tolist())]
    node_info = [{} for _ in range(graph_node_id_threshold)]
    for i, (src, tar) in enumerate(zip(source_ids, target_ids)):
        if src < graph_node_id_threshold and not node_info[src]:
            node_info[src]['title'] = source_titles[i]
            node_info[src]['abstract'] = source_abstracts[i]
            node_info[src]['venue'] = source_venues[i]
            node_info[src]['author'] = source_authors[i]
            node_info[src]['year'] = source_years[i]
        if tar < graph_node_id_threshold and not node_info[tar]:
            node_info[tar]['title'] = target_titles[i]
            node_info[tar]['abstract'] = target_abstracts[i]
            node_info[tar]['venue'] = target_venues[i]
            node_info[tar]['author'] = target_authors[i]
            node_info[tar]['year'] = target_years[i]

    return edge_lists, node_info

def _convert_df_type(df, columns, type):
    '''
    Convert the type of given columns

    Parameters
    ----------
    df: input dataframe
    columns: columns to be processed
    type: the type wanted to transform into
    '''
    if not isinstance(columns, list):
        columns = [columns]
    for column in columns:
        df[[column]] = df[[column]].astype(type)

    return df

def _cut_off_dataset(df, year, frequency=5):
    '''
    Remove items with (src_year > year or (src_year == year and tar_year == year)),
    remove items whose citations < frequency

    Parameters
    ----------
    df: input dataframe
    year: the year threshold to use
    frequency: the citation threshold
    '''
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
    '''
    Slicing the citation context to meet the length constraints

    Parameters
    ----------
    df: input dataframe
    number: maximal length of citation context
    '''
    df['LeftString'] = df['left_citated_text'].str[-number:]
    df['RightString'] = df['right_citated_text'].str[:number]

    return df

def _split_dataset(df, year):
    '''
    Split dataset and renumber the ids

    Parameters
    ----------
    df: input dataframe
    year: the year threshold
    '''
    # get the splited indices
    train_idx = df['source_year'][df['source_year'] < year].index
    test_idx = df['source_year'][df['source_year'] == year].index

    def _get_ids(df, idx, only_tar=False):
        df = df.loc[idx]
        source_ids = df['source_id'].values.tolist() if not only_tar else []
        target_ids = df['target_id'].values.tolist()
        ids = set(source_ids + target_ids)

        return ids

    # renumber the id
    train_node_ids = _get_ids(df, train_idx)
    test_node_ids = _get_ids(df, test_idx)
    test_target_node_ids = _get_ids(df, test_idx, only_tar=True) - train_node_ids
    additional_node_ids = test_node_ids - train_node_ids - test_target_node_ids

    train_nodes_mapping = dict(zip(list(train_node_ids), range(len(train_node_ids))))
    test_target_nodes_mapping = dict(zip(list(test_target_node_ids), range(len(train_node_ids), len(train_node_ids) + len(test_target_node_ids))))
    additional_nodes_mapping = dict(zip(list(additional_node_ids), range(len(train_node_ids) + len(test_target_node_ids), len(train_node_ids) + len(test_target_node_ids) + len(additional_node_ids))))
    mapping = {**train_nodes_mapping, **test_target_nodes_mapping, **additional_nodes_mapping}

    new_src_id, new_tar_id = [], []
    for src, tar in zip(df['source_id'], df['target_id']):
        new_src_id.append(mapping[src])
        new_tar_id.append(mapping[tar])
    df['SourceID'] = new_src_id
    df['TargetID'] = new_tar_id
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]
    graph_node_id_threshold = len(train_node_ids) + len(test_target_node_ids)

    return train_df, test_df, df, graph_node_id_threshold

def _get_ground_truth(df, graph_node_id_threshold):
    groud_truth = [{} for _ in range(graph_node_id_threshold)]

    data = df[['SourceID', 'TargetID', 'source_title', 'source_abstract', 'source_venue', 'source_year', 'source_author',
             'target_title', 'target_abstract', 'target_venue', 'target_year', 'target_author']].to_numpy()
    keys = ['title', 'abstract', 'venue', 'year', 'author']
    for _, record in enumerate(data):
        src_id, tar_id = record[0], record[1]
        if src_id < graph_node_id_threshold and not groud_truth[src_id]:
            values = record[2:7].tolist()
            groud_truth[src_id] = dict(zip(keys, values))
        if tar_id < graph_node_id_threshold and not groud_truth[tar_id]:
            values = record[7:].tolist()
            groud_truth[tar_id] = dict(zip(keys, values))
    return groud_truth

if __name__ == "__main__":
    train_df, test_df, ground_truth, whole_df, graph_node_id_threshold = split_process_dataset(file_path='../data/citation.csv', seq_len=50, year=2015, frequency=5)

    # construct graph
    edge_list, node_info = construct_graph(train_df, whole_df, graph_node_id_threshold)

