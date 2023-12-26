import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pk
from numpy import linalg as LA
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA

def save_direct(direct_name):
    Current_directory = os.getcwd()
    directory = os.path.join(Current_directory, direct_name)
    if not os.path.exists(directory):
          os.makedirs(directory)
    return directory



datapath = r'/home/r10524004/auto_data'
train_datapath = os.path.join(datapath, r'auto_train_data_store')
train_filelist = os.listdir(train_datapath)
valid_datapath = os.path.join(datapath, r'auto_valid_data_store')
valid_filelist = os.listdir(valid_datapath)


samples=4500
Time = 80
width = 256
height = 256
auto_shape = (samples,
              Time,
              width,
              height)

auto = np.zeros(auto_shape, dtype=np.float32)
for i in range(1, samples+1):
    filename = os.path.join(train_datapath, train_filelist[i-1])
    data = np.load(filename)['data']
    auto[i-1, :, :, :]=data.astype(np.float32)        

#s = 500
#auto_valid_shape = (s,
                    #Time,
                    #width,
                    #height)
#auto_valid = np.zeros(auto_valid_shape, dtype=np.float32)
#for i in range(len(valid_filelist)):
    #filename = os.path.join(valid_datapath, valid_filelist[i])
    #data = np.load(filename)['data']
    #auto_valid[i, :, :, :]=data.astype(np.float32)


#auto = np.concatenate([auto, auto_valid])

#del auto_valid
       
PCA_components = 50

pca = PCA(
    svd_solver='auto',
    n_components=PCA_components)

PCA_trans = pca.fit_transform(auto.reshape((samples)*Time,width*height))

PC_plot = PCA_trans.reshape((samples, Time, PCA_components))

exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
print("variance={}".format(cum_sum_eigenvalues[-1]))

model_direct=save_direct(direct_name='PCA_model')
filename = os.path.join(model_direct, 'PCA_variance_phase.png')
plt.plot(range(1, len(cum_sum_eigenvalues)+1), cum_sum_eigenvalues, marker='.')
plt.xlabel("principle components")
plt.ylabel("variance")
plt.savefig(filename)
plt.close()
modelname = os.path.join(model_direct, 'pca_50.pkl')
pk.dump(pca, open(modelname,"wb"))
var_name = os.path.join(model_direct, 'cum_var.npy')
np.save(var_name, cum_sum_eigenvalues)




### plot the 3D plot
for i in np.arange(0, 360, 30):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    
    for j in range(len(PC_plot)):
        ax.scatter(PC_plot[j, :, 0], PC_plot[j, :, 1], PC_plot[j, :, 2], marker='.')
    
    ax.set_xlabel(r"$\varphi_{1}$", style='italic', fontsize=11)
    ax.set_ylabel(r"$\varphi_{2}$", style='italic', fontsize=11)
    ax.set_zlabel(r"$\varphi_{3}$", style='italic', fontsize=11)
    ax.tick_params(axis='both', which='both', labelsize=11)
    
    ax.view_init(azim=i, elev=23)
    
    filename = os.path.join(model_direct, "3D_Auto_azim={}.png".format(i))
    fig.savefig(filename)
    plt.close()
    

del auto

pca_reload = pk.load(open(modelname,"rb"))
GT_data = np.zeros((500, Time, width, height), dtype=np.float32)
for k in range(500):
    filename = os.path.join(train_datapath, train_filelist[k])
    data = np.load(filename)['data']
    GT_data [k, :, :, :]= data.astype(np.float32)

pc_trans = pca_reload.transform(GT_data.reshape(500*Time,width*height))

pc_projected = pca_reload.inverse_transform(pc_trans)

PD_data = pc_projected.reshape((GT_data.shape))

total_loss=LA.norm(GT_data-PD_data, None)

print('reconstruction loss = {:.4f}'.format(total_loss))

for i in range(0, 500, 70):
    model_direct=save_direct(direct_name='PCA_model')
    filename = os.path.join(model_direct, 'PCA_compare_phase={}.png'.format(i))
    fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize=(13,17))
    figure0=ax[0, 0].imshow(GT_data[i, 35, :, :])
    figure1=ax[0, 1].imshow(GT_data[i, 55, :, :])
    figure2=ax[1, 0].imshow(PD_data[i, 35, :, :])
    figure3=ax[1, 1].imshow(PD_data[i, 55, :, :])
    figure4=ax[2, 0].imshow(abs(GT_data[i, 35, :, :]-PD_data[i, 35, :, :]))
    figure5=ax[2, 1].imshow(abs(GT_data[i, 55, :, :]-PD_data[i, 55, :, :]))
    ax[0, 0].set_title('inverse target:35')
    ax[0, 1].set_title('inverse target:55')
    ax[1, 0].set_title('inverse prediction:35')
    ax[1, 1].set_title('inverse prediction:55')
    ax[2, 0].set_title('error:35')
    ax[2, 1].set_title('error:55')

    divider0 = make_axes_locatable(ax[0,0])
    divider1 = make_axes_locatable(ax[0,1])
    divider2 = make_axes_locatable(ax[1,0])
    divider3 = make_axes_locatable(ax[1,1])
    divider4 = make_axes_locatable(ax[2,0])
    divider5 = make_axes_locatable(ax[2,1])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cax5 = divider5.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(figure0, cax=cax0)
    plt.colorbar(figure1, cax=cax1)
    plt.colorbar(figure2, cax=cax2)
    plt.colorbar(figure3, cax=cax3)
    plt.colorbar(figure4, cax=cax4)
    plt.colorbar(figure5, cax=cax5)
    plt.savefig(filename)
    plt.close()
