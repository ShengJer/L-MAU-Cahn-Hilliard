import os
import sys
import shutil
import argparse
import numpy as np
import pynvml
sys.path.append('./data_provider')
sys.path.append('./model')
from data_provider import datasets_factory
from model_factory import Model
import trainer as trainer
import pickle as pk
import matplotlib.pyplot as plt
import torch

pynvml.nvmlInit() # initialize GPU version message
np.random.seed(1234)
torch.manual_seed(1234)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='lMAU parameters')

parser.add_argument('-train_data_paths', type=str, default='./data/PCA_data/train_data_50.npz')
parser.add_argument('-valid_data_paths', type=str, default='./data/PCA_data/valid_data_50.npz')
parser.add_argument('-test_data_paths', type=str, default='./data/PCA_data/test_data_50.npz')

parser.add_argument('-PCA_dir', type=str, default='./data/PCA_model')
parser.add_argument('-PCA_name', type=str, default='pca_50.pkl')
parser.add_argument('-gen_frm_dir', type=str, default='results')
parser.add_argument('-test_frm_dir', type=str, default='test_results')
parser.add_argument('-save_dir', type=str, default='checkpoints')
parser.add_argument('-cplot_dir', type=str, default='cplot')
parser.add_argument('-Graph_dir', type=str, default='Graph')
parser.add_argument('-dataset_name', type=str, default='phase_field')
parser.add_argument('-save_modelname', type=str, default='model.pt.tar-58000')

parser.add_argument('-batch_size', type=int, default=10)
parser.add_argument('-in_features', type=int, default=50)
parser.add_argument('-img_width', type=int, default=256)
parser.add_argument('-img_height', type=int, default=256)
parser.add_argument('-img_channel', type=int, default=1)
parser.add_argument('-total_length', type=int, default=80)
parser.add_argument('-input_length', type=int, default=10)
parser.add_argument('-output_length', type=int, default=70)
parser.add_argument('-display_interval', type=int, default=100)
parser.add_argument('-max_iterations', type=int, default=80000)
parser.add_argument('-plt_num_PCs', type=int, default=20)
## model parameters
parser.add_argument('-model_name', type=str, default='lmau')
parser.add_argument('-num_layers', type=int, default=4)
parser.add_argument('-num_hidden', type=int, default=128) # 64 or 128
parser.add_argument('-tau', type=int, default=40)
parser.add_argument('-cell_mode', type=str, default='residual')
parser.add_argument('-model_mode', type=str, default='recall')
## optimizer and scheduler:
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-lr_decay', type=float, default=0.85)
parser.add_argument('-step_size', type=int, default=4000) # consider interation
parser.add_argument('-loss_type', type=str, default='L1+L2')

## test
parser.add_argument('-test_interval', type=int, default=4000)
parser.add_argument('-num_save_samples', type=int, default=11)
parser.add_argument('-is_training', type=int, default=1)
parser.add_argument('-load_model', type=int, default=0)
parser.add_argument('-device', type=str, default='cuda:0')
# schedule sampling parameters
parser.add_argument('-scheduled_sampling', type=int, default=1)
parser.add_argument('-sampling_stop_iter', type=int, default=50000)
parser.add_argument('-sampling_start_value', type=float, default=1.0)
parser.add_argument('-sampling_changing_rate', type=float, default=0.00002)

args = parser.parse_args()
args.tied = True




def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.in_features))
    #(B, TL-IL-1, Fin)
    
    if not args.scheduled_sampling: # if scheduled_sampling = 0 then execute eta = 0.0, all zeros
        return 0.0, zeros 

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate  # probability eta decaying with sampling changing_rate
    else:
        eta = 0.0  # do not use true frame as input
    #print('eta: ', eta)
    
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    # Return random_flip (B,TL-IL-1) from continuous uniform distribution (randomly mask some time-length)
    
    true_token = (random_flip < eta) # true false matrix (B, TL-IL-1)
    
    ones = np.ones((args.in_features))
    # shape Fin
    zeros = np.zeros((args.in_features))
    # shape Fin
    
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag) 
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.in_features))
    # real_input_flag shape= #(B, TL-IL-1, in_features)
    return eta, real_input_flag


def Graph(criterion, idex):
    Current_directory = os.getcwd()
    graph_directory = os.path.join(Current_directory, args.Graph_dir)
    if not os.path.exists(graph_directory):
        os.mkdir(graph_directory)
          
    criterion_name = os.path.join(graph_directory, 'criterion.pkl')
    with open(criterion_name, 'wb') as f:
        pk.dump(criterion, f)
         
    plt.plot(range(1, len(criterion['train_epoch_loss'])+1), criterion['train_epoch_loss'], marker='.')
    plt.xlabel('epochs')
    plt.ylabel('train_epoch_loss')
    name = os.path.join(graph_directory, 'train_epoch_loss.png')
    plt.savefig(name)
    plt.close()
    plt.plot(range(1, len(criterion['train_avg_itrs_loss'])+1), criterion['train_avg_itrs_loss'], marker='.')
    plt.xlabel('epochs')
    plt.ylabel('average_train_itrs_loss')
    name = os.path.join(graph_directory, 'train_iter_loss.png')
    plt.savefig(name)
    plt.close()
    # the validation loss does not consider elementwide mean just square error/ how many samples
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(idex, criterion['average_valid_mse'], marker='.', color=color)
    ax1.set_xlabel('itrs')
    ax1.set_ylabel('average_valid_mse')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(idex, criterion['average_valid_mae'], marker='.', color=color)
    ax2.set_ylabel('average_valid_mae')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    name = os.path.join(graph_directory, 'average_valid_mse_mae.png')
    plt.savefig(name)
    plt.close()
def train_wrapper(model, pca_model):
    begin = 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo_begin = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    if args.load_model == 1:
        current = os.getcwd()
        save_path = os.path.join(current, args.configs.save_dir)
        model_filename = os.path.join(save_path, args.save_modelname)
        model.load(model_filename)
        begin = int(model_filename.split('-')[-1]) # get out the final save iteration
    
    # load the autocorrelation data from file
    train_input_handle, val_input_handle = datasets_factory.data_provider(
        dataset_name=args.dataset_name,
        train_data_paths=args.train_data_paths,
        valid_data_paths=args.valid_data_paths,
        batch_size=args.batch_size,
        img_width=args.img_width,
        seq_length=args.total_length,
        is_training=True,
        PCs=args.in_features)
    
    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)
    itrss = begin # the last save itr
    # real_input_flag = {}
    best_valid_mse = 1e+08
    iter_loss=0
    iter_store=0
    epoch_loss=[]
    average_itrs_loss = []
    valid_itr_id = []
    valid_avg_mse = []
    valid_avg_mae = []
    for itr in range(1, args.max_iterations + 1):
        if itrss > args.max_iterations: # (itr may be a certain iteration that is save)
            break
        if train_input_handle.no_batch_left():
            epoch_loss.append(iter_loss)
            average_itrs_loss.append(iter_loss/(itr-iter_store))
            iter_loss = 0
            iter_store = itr
            train_input_handle.begin(do_shuffle=True)
        itrss += 1
        ims = train_input_handle.get_batch() # (B, TL, PCs) dtype=np.float32 # autocorrelation data
        
        ## normalize the data
        
        max_value=np.max(ims, axis=1)[:,None,:]
        min_value=np.min(ims, axis=1)[:,None,:]
        denominator = (max_value - min_value)
        if (denominator == 0).any():
            denominator[denominator == 0] = 1.0
        norm_ims = (ims-min_value)/denominator
        
        ## no normalize
        #norm_ims = ims

        eta, real_input_flag = schedule_sampling(eta, itr)
        #(B, TL-IL-1, in_features)
        l2_loss=trainer.train(model, norm_ims, real_input_flag, args, itr)
        iter_loss += l2_loss
        
        if itr % args.test_interval == 0:
            print('Validate:')
            avg, frame_cri = trainer.test(model, pca_model, val_input_handle, args, itr)
            valid_avg_mse.append(avg['average_mse']) # average mse "per frame"
            valid_avg_mae.append(avg['average_mae']) # average mae "per frame"
            valid_itr_id.append(itr)
            if avg['average_mse'] < best_valid_mse:
                model.save(itr)
                best_valid_mse = avg['average_mse']
    
            meminfo_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print("GPU memory:%dM" % ((meminfo_end.used - meminfo_begin.used) / (1024 ** 2)))
        train_input_handle.next()
        
    # plot the training and validation loss
    criterion = {'train_epoch_loss': epoch_loss,
           'train_avg_itrs_loss': average_itrs_loss,
           'average_valid_mse': valid_avg_mse,
           'average_valid_mae': valid_avg_mae,}   
    idex = valid_itr_id
    Graph(criterion, idex)


def test_wrapper(model, pca_model):
    current = os.getcwd()
    save_path = os.path.join(current, args.save_dir)
    model_filename = os.path.join(save_path, args.save_modelname)
    model.load(model_filename)
    test_input_handle = datasets_factory.data_provider(
        dataset_name=args.dataset_name,
        train_data_paths=args.train_data_paths,
        valid_data_paths=args.test_data_paths,
        batch_size=args.batch_size,
        img_width=args.img_width,
        seq_length=args.total_length,
        is_training=False,
        PCs=args.in_features)
    
    trainer.test(model, pca_model, test_input_handle, args, 'test_result')


if __name__ == '__main__':

    print('Initializing models')
    # RNN model
    model = Model(args)
    # PCA model
    modelname = os.path.join(args.PCA_dir, args.PCA_name) 
    pca_model = pk.load(open(modelname,"rb"))
    
    if args.is_training:
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir) # remove the whole directory
        os.makedirs(args.save_dir)
        
        if os.path.exists(args.gen_frm_dir):
            shutil.rmtree(args.gen_frm_dir)
        os.makedirs(args.gen_frm_dir)
        train_wrapper(model, pca_model)
        
    else:
        test_wrapper(model, pca_model)

