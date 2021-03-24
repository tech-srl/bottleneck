import functools
import json
import pickle
import sys
from argparse import ArgumentParser
import gzip
import re
import os
import random
import multiprocessing as mp

ignore_list = ['openlivewriter', 'botbuilder']
CANDIDATE_BEGIN = '<CANDIDATE>'
CANDIDATE_END = '</CANDIDATE>'
SLOT = '<SLOT>'
project_name_map = {
    'akka.net': 'akka'
}
filename_mapping = {
    'C:\\Users\\t-mialla\\Documents\\sampleProjects\\SignalR\\src\\Microsoft.AspNet.SignalR.Core\\Messaging\\Cursor.cs': 'Core\\Messaging\\Cursor.cs'
}

RE_WORDS = re.compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ |  # All upper case
    \d+ | # Numbers
    _ |
    \" |
    .+
''', re.VERBOSE)

def split_subtokens(str):
    return [subtok for subtok in RE_WORDS.findall(str) if not subtok == '_']

def get_immediate_subdirectories(a_dir):
    return [(os.path.join(a_dir, name)) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(dir):
    return [(os.path.join(dir, name)) for name in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, name))]

def collect_tokens(path):
    projects = get_immediate_subdirectories(path)
    
    tokens_dict = {}
    for proj in projects:
        proj_name = proj.split('/')[-1]
        if proj_name in project_name_map:
            proj_name = project_name_map[proj_name]
        tokens_file_name = f'{proj}/{proj_name}-tokens.json.gz'
        if proj_name in ignore_list:
            continue
        #if os.path.isfile(tokens_file_name):
        with gzip.open(tokens_file_name, 'r') as file:
            lines = file.readlines()
            objs = json.loads(lines[0])
            for o in objs:
                tokens_dict[o['Provenance']] = o['Tokens']
    return tokens_dict

def create_sequences(path, tokens_dict, out_path):
    subsets = get_immediate_subdirectories(path) # train, valid, test, testonly
    process_gz_file_func = functools.partial(process_gz_file, tokens_dict)
    for dir in subsets:
        dir_name = dir.split('/')[-1]
        out_dir_path = f'{out_path}/{dir_name}'
        if os.path.isdir(out_dir_path):
            raise ValueError(f'{out_path}/{dir_name} already exists')
        os.mkdir(out_dir_path)
        files = get_immediate_files(dir)
        with open(f'{out_dir_path}/source.txt', 'w') as out_source_file, \
                open(f'{out_dir_path}/target.txt', 'w') as out_target_file:
            with mp.Pool(64) as pool:
                #results = [process_gz_file(file, tokens_dict) for file in files]
                results = pool.imap_unordered(process_gz_file_func, files)
                for example in results:
                    for source, target in zip(*example):
                        out_source_file.write(source)
                        out_target_file.write(target)

def process_gz_file(tokens_dict, gz_file_name):
    sources, targets = [], []
    with gzip.open(gz_file_name, 'r') as gz_file:
        lines = gz_file.readlines()
        objs = [json.loads(l) for l in lines]
        for o in objs:
            filename = o['filename']
            if filename in tokens_dict:
                tokens = tokens_dict[filename]
            elif filename in filename_mapping:
                tokens = tokens_dict[filename_mapping[filename]]
            else:
                found_filenames = [name for name in tokens_dict.keys() if filename.endswith(name)]
                if len(found_filenames) != 1:
                    found_filenames = [name for name in found_filenames if name != 's']
                    if len(found_filenames) != 1:
                        raise ValueError(
                            f'Looking for filename: {filename}, but found in tokens_dict: {found_filenames}')

                tokens = tokens_dict[found_filenames[0]]
                print(f'Taking {found_filenames[0]} instead of {filename}')
            slot_token_index = o['slotTokenIdx']
            tokens[slot_token_index] = SLOT
            subtokens = [' '.join(split_subtokens(tok)) for tok in tokens]
            candidates = [' '.join(split_subtokens(candi['SymbolName'])) for candi in o['SymbolCandidates']]
            # Important to shuffle, because the first one is always the correct one
            random.shuffle(candidates)
            label = [' '.join(split_subtokens(candi['SymbolName'])) for candi in o['SymbolCandidates'] if
                     candi['IsCorrect'] == True]
            if len(label) is not 1:
                raise ValueError(f'Found {len(label)} correct labels in {gz_file_name}, example {o["filename"]}')
            label = label[0]
            outline = ' '.join(subtokens) + ' ' + ' '.join(
                [CANDIDATE_BEGIN + ' ' + candi + ' ' + CANDIDATE_END for candi in candidates]) + '\n'
            sources.append(outline)
            targets.append(label + '\n')
    return sources, targets

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--raw", dest="raw_path", required=True)
    parser.add_argument("--reorg", dest="reorg_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)

    args = parser.parse_args()
    
    #tokens_dict = collect_tokens(args.raw_path)
    with open('tokens.pkl', 'rb') as file:
        tokens_dict = pickle.load(file)
    sequences = create_sequences(args.reorg_path, tokens_dict, args.out_path)
