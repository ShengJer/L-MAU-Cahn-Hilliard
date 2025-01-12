import os
import shutil

def clean_directory(config):
    if os.path.exists(config.results_path):
        shutil.rmtree(config.results_path)
    if os.path.exists(config.ckpt_path): 
        shutil.rmtree(config.ckpt_path)
    if os.path.exists(config.graph_path): 
        shutil.rmtree(config.graph_path)
    os.makedirs(config.results_path)
    os.makedirs(config.ckpt_path)
    os.makedirs(config.graph_path)

def save_direct(direct_name):
    Current_directory = os.getcwd()
    directory = os.path.join(Current_directory, direct_name)
    if not os.path.exists(directory):
          os.makedirs(directory)