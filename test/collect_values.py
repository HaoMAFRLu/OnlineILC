"""Collect variables
"""
import os, sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

def is_folder_empty(folder_path):
    contents = os.listdir(folder_path)
    return len(contents) == 0

def list_files_in_folder(folder_path):
    contents = os.listdir(folder_path)
    files = [f for f in contents if os.path.isfile(os.path.join(folder_path, f))]
    return files

def read_marker(file):
    with open(file, 'rb') as file:
        yref = pickle.load(file)
        yout = pickle.load(file)
        d = pickle.load(file)
        u = pickle.load(file)
        loss = pickle.load(file)
    return yref, yout, d, u, loss

def save_variable_marker(path, folder):
    subfolder = 'loss_marker'
    variables = ['yref', 'yout', 'd', 'u', 'loss']
    path_folder = os.path.join(path, folder, subfolder)

    for var in variables:
        globals()[var] = []
    
    if os.path.isdir(path_folder):
        if is_folder_empty(path_folder):
            print('Empty!')
        else:
            files = list_files_in_folder(path_folder)
            files = sorted(files, key=int)

            for i in files:
                path_file = os.path.join(path_folder, i)
                data = read_marker(path_file)
            
                for i_var in range(len(variables)):
                    var = variables[i_var]
                    lst = globals()[var]
                    lst.append(data[i_var])
            
            for var in variables:
                path_save = os.path.join(path, folder, subfolder+'_'+var)
                lst = globals()[var]
                with open(path_save, 'wb') as file:
                    pickle.dump(lst, file)
    else:
        print('Not a directory!')

def read_data(file):
    with open(file, 'rb') as file:
        data = pickle.load(file)
    return data

def save_variable_data(path, folder):
    subfolder = 'data'
    variables = ['hidden_states', 'u', 'yref', 'd', 'yout', 'loss', 'gradient']
    is_exist = [1] * len(variables)

    path_folder = os.path.join(path, folder, subfolder)

    for var in variables:
        globals()[var] = []
    
    if os.path.isdir(path_folder):
        if is_folder_empty(path_folder):
            print('Empty!')
        else:
            files = list_files_in_folder(path_folder)
            files = sorted(files, key=int)

            for i in files:
                path_file = os.path.join(path_folder, i)
                data = read_data(path_file)

                for i_var in range(len(variables)):
                    var = variables[i_var]
                    if var in data:
                        lst = globals()[var]
                        lst.append(data[var])
                    else:
                        is_exist[i_var] = 0

            for var in variables:
                if is_exist[variables.index(var)] == 1:
                    path_save = os.path.join(path, folder, subfolder+'_'+var)
                    lst = globals()[var]
                    with open(path_save, 'wb') as file:
                        pickle.dump(lst, file)
    else:
        print('Not a directory!')

def save_variable(path, folder, subfolder):
    if subfolder == 'loss_marker':
        save_variable_marker(path, folder)
    elif subfolder == 'data':
        save_variable_data(path, folder)

if __name__ == '__main__':
    root = fcs.get_parent_path(lvl=1)
    folder1 = 'test'
    path = os.path.join(root, 'data', folder1)
    folders = ['20240906_144508']

    if len(folders) == 0:
        folders = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    
    for folder in folders:
        save_variable(path, folder, 'loss_marker')
        save_variable(path, folder, 'data')

