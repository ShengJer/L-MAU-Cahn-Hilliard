import os.path
import numpy as np
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import pickle as pk

def train(model, ims, real_input_flag, configs, itr):
    # ims = [B, TL, Fin]  np.float32 with normalization
    # real_input_flag = (B, TL-IL-1, Fin)  np.float32
    _, loss_l1, loss_l2 = model.train(ims, real_input_flag, itr)
    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('itr: ' + str(itr),
              'training L1 loss: ' + str(loss_l1), 'training L2 loss: ' + str(loss_l2))
    
    return loss_l2

def test(model, Automodel, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    if configs.is_training == 0:
        res_path = os.path.join(configs.test_frm_dir, str(itr))
    else:
        res_path = os.path.join(configs.gen_frm_dir, str(itr))
        
    test_input_handle.begin(do_shuffle=False)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    avg_mse = 0
    avg_mae = 0
    batch_id = 0
    img_mse, img_mae = [], []
    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        img_mae.append(0)
    
    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         configs.in_features))
    

    while (test_input_handle.no_batch_left() == False or batch_id ==0):
        batch_id = batch_id + 1
        if configs.is_training == 0:
            test_ims, input_ms = test_input_handle.get_batch()
        else:
            test_ims = test_input_handle.get_batch()
        #test_ims # (B, TL, PCs)
        #batch_deno (B, TL, 1, 1, 1)
        #batch_min (B, TL, 1, 1, 1)
        #input_ms (B,TL, 1, 256, 256)
        ## normalize the data
        
        test_max_value=np.max(test_ims, axis=1)[:,None,:]
        test_min_value=np.min(test_ims, axis=1)[:,None,:]
        denominator = (test_max_value-test_min_value)
        if (denominator == 0).any():
            denominator[denominator == 0] = 1.0
        norm_test_ims = (test_ims-test_min_value)/(denominator)
        
        img_gen, _, _ = model.test(norm_test_ims, real_input_flag) # (B,TL-1,Fin)
        
        output_length = configs.total_length - configs.input_length # TL-IL
        img_out = img_gen[:, -output_length:]
        # takeout img_gen i=IL-1 ~ TL-2 which is the prediction of test_ims i=IL~TL-1

        #MSE per frame
        for i in range(output_length): # i=0~TL-IL-1
            x = test_ims[:, i + configs.input_length, :] # index=IL ~ TL-1
            gx = img_out[:, i, :] # prediction of test_ims i=IL~TL-1
            mse = np.square(x - gx).sum()# calculate the square error
            mae = np.abs(x - gx).sum()
            img_mae[i] += mae
            img_mse[i] += mse
            avg_mse += mse
            avg_mae += mae
            ## ssim should be (H, W, C) ...
            #x_ssim = x.transpose(0,2,3,1)
            #gx_ssim = gx.transpose(0,2,3,1)
        
        
        # norm_test_ims (B,TL,Fin)
        # img_out (B,OL,Fin)
        # save prediction examples
        
        orig_scalems_np = norm_test_ims * (denominator) + test_min_value
        pred_scale_np= img_out * (denominator) + test_min_value
        
        if configs.is_training == 0:
            ### plot the latent space
            orig_ims_np=orig_scalems_np.reshape(configs.batch_size, configs.total_length, configs.in_features) #(BxTL, Fin)
            pred_np=pred_scale_np.reshape(configs.batch_size, configs.output_length, configs.in_features) # (BxOL, Fin)
            
            
            orig_tensor=torch.from_numpy(orig_ims_np).to(torch.float32).to(configs.device)
            pred_tensor=torch.from_numpy(pred_np).to(torch.float32).to(configs.device)

            orig_tensor = orig_tensor.reshape(configs.batch_size*configs.total_length, configs.latent_channel, configs.latent_height, configs.latent_width)
            pred_tensor = pred_tensor.reshape(configs.batch_size*configs.output_length, configs.latent_channel, configs.latent_height, configs.latent_width)

            orig_imag=Automodel.decoder(orig_tensor)
            pred_imag=Automodel.decoder(pred_tensor)
            
            orig_imag = orig_imag.reshape(configs.batch_size, configs.total_length, configs.img_channel, configs.img_height, configs.img_width)
            pred_imag = pred_imag.reshape(configs.batch_size, configs.output_length, configs.img_channel, configs.img_height, configs.img_width)

            orig_imag = orig_imag.detach().cpu().numpy()
            pred_imag = pred_imag.detach().cpu().numpy()
            
            orig_imag = orig_imag.reshape(configs.batch_size, configs.total_length, configs.img_height, configs.img_width)
            pred_imag = pred_imag.reshape(configs.batch_size, configs.output_length, configs.img_height, configs.img_width)
            
            for k in range(len(orig_imag)): 
                path = os.path.join(res_path, str(k+1))
                os.makedirs(path)
                
                
                for j in range(configs.plt_num_PCs):
                    name = 'PC='+str(j + 1) + '.png'
                    file_name = os.path.join(path, name)
                    plt.plot(range(1, configs.total_length+1), orig_scalems_np[0, :, j], marker='.', label='GT')
                    plt.plot(range(configs.input_length, configs.input_length+configs.output_length), pred_scale_np[0, :, j], marker='.', label='PD')
                    plt.legend()
                    plt.savefig(file_name)
                    plt.close()
                
                x = np.arange(0, input_ms.shape[3])
                y = np.arange(0, input_ms.shape[3])
                X, Y = np.meshgrid(x, y)
                rangeGT = [10, 20 ,30 ,40 ,50, 60, 70, 79]
                rangePD = [0, 10, 20, 30, 40, 50, 60, 69]
                
                
                GT_latentname = os.path.join(path, 'latent_GT.npz')
                PD_latentname = os.path.join(path, 'latent_PD.npz')
                GT_storename = os.path.join(path, 'No_iter.npz')
                PD_storename = os.path.join(path, 'PD.npz')
                np.savez_compressed(GT_storename, data=orig_imag[k, :, :, :])
                np.savez_compressed(PD_storename, data=pred_imag[k, :, :, :])
                np.savez_compressed(GT_latentname, data=orig_scalems_np)
                np.savez_compressed(PD_latentname, data=pred_scale_np)
                
                for i in rangeGT:
                    name = 'gt' + str(i) + '.png'
                    file_name = os.path.join(path, name)
                    counter_set = plt.contourf(X, Y, input_ms[k, i, 0, :, :], levels=np.linspace(0, 1, 30))
                    plt.colorbar(counter_set, label='$\phi_{p}$')
                    plt.savefig(file_name)
                    plt.close()
                    
                for i in rangeGT:
                    name_noiter = 'no_iter' + str(i) + '.png'
                    file_name = os.path.join(path, name_noiter)
                    counter_set00 = plt.contourf(X, Y, orig_imag[k, i, :, :], levels=np.linspace(0, 1, 30))
                    plt.colorbar(counter_set00, label='$\phi_{p}$')
                    plt.savefig(file_name)
                    plt.close()
                    
                for i in rangePD:
                    name = 'pd' + str(i+configs.input_length) + '.png'
                    file_name = os.path.join(path, name)
                    counter_set = plt.contourf(X, Y, pred_imag[k, i, :, :], levels=np.linspace(0, 1, 30))
                    plt.colorbar(counter_set, label='$\phi_{p}$')
                    plt.savefig(file_name)
                    plt.close()
                    
                
        else:
            if batch_id <= configs.num_save_samples:
                path = os.path.join(res_path, str(batch_id))
                os.makedirs(path)
    
                for i in range(configs.plt_num_PCs):
                    name = 'PC='+str(i + 1) + '.png'
                    file_name = os.path.join(path, name)
                    plt.plot(range(1, configs.total_length+1), orig_scalems_np[0, :, i], marker='.', label='GT')
                    plt.plot(range(configs.input_length, configs.input_length+configs.output_length), pred_scale_np[0, :, i], marker='.', label='PD')
                    plt.legend()
                    plt.savefig(file_name)
                    plt.close()
                
        test_input_handle.next()
        
    avg_mse = avg_mse / (batch_id * configs.batch_size * configs.output_length)
    avg_mae = avg_mae / (batch_id * configs.batch_size *configs.output_length)

    print('average mse per frame: ' + str(avg_mse))

    print('average mae per frame: ' + str(avg_mae))

    avg = {'average_mse': avg_mse,
           'average_mae': avg_mae}
    
    frame_cri = {'Im_mse': img_mse,
                 'Im_mae': img_mae}
    file_name = os.path.join(res_path, 'Im_per_frame.npz')
    
    np.savez_compressed(file_name, sse=img_mse, sae=img_mae)
    
    return avg, frame_cri

