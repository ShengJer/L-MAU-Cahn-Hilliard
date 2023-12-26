import os.path
import numpy as np
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable


def train(model, ims, real_input_flag, configs, itr):
    # ims = [B, TL, Fin]  np.float32 with normalization
    # real_input_flag = (B, TL-IL-1, Fin)  np.float32
    _, loss_l1, loss_l2 = model.train(ims, real_input_flag, itr)
    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('itr: ' + str(itr),
              'training L1 loss: ' + str(loss_l1), 'training L2 loss: ' + str(loss_l2))
    
    return loss_l2

def test(model, pca_model, test_input_handle, configs, itr):
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
    
    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()  # (B, TL, PCs)
        
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
            x = norm_test_ims[:, i + configs.input_length, :] # index=IL ~ TL-1
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
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.makedirs(path)

            for i in range(configs.plt_num_PCs):
                name = 'PC='+str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                plt.plot(range(1, configs.total_length+1), norm_test_ims[0, :, i], marker='.', label='Normalized GT')
                plt.plot(range(configs.input_length, configs.input_length+configs.output_length), img_out[0, :, i], marker='.', label='PD')
                plt.legend()
                plt.savefig(file_name)
                plt.close()
                
            orig_ims_L_np = norm_test_ims * (denominator) + test_min_value
            pred_scale_L_np= img_out * (denominator) + test_min_value

            orig_ims=orig_ims_L_np.reshape(-1, configs.in_features)
            pred_scale=pred_scale_L_np.reshape(-1, configs.in_features
)
            orig_ims=pca_model.inverse_transform(orig_ims).reshape(configs.batch_size, configs.total_length, configs.img_height, configs.img_width)
            pred_scale=pca_model.inverse_transform(pred_scale).reshape(configs.batch_size, configs.output_length, configs.img_height, configs.img_width)
            if configs.is_training == 0:
                np.savez_compressed(os.path.join(path, 'auto_GT.npz'), data=orig_ims)
                np.savez_compressed(os.path.join(path, 'auto_PD.npz'), data=pred_scale)
                np.savez_compressed(os.path.join(path, 'latent_GT.npz'), data=orig_ims_L_np)
                np.savez_compressed(os.path.join(path, 'latent_PD.npz'), data=pred_scale_L_np)
            else:
                pass
                
            
            
            fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize=(13,17))
            file_name = os.path.join(path, 'compare.png')
            figure0=ax[0, 0].imshow(orig_ims[0, -10, :, :])
            figure1=ax[0, 1].imshow(orig_ims[0, -20, :, :])
            figure2=ax[1, 0].imshow(pred_scale[0, -10, :, :])
            figure3=ax[1, 1].imshow(pred_scale[0, -20, :, :])
            figure4=ax[2, 0].imshow(abs(orig_ims[0,-10,:,:]-pred_scale[0,-10,:,:]))
            figure5=ax[2, 1].imshow(abs(orig_ims[0,-20,:,:]-pred_scale[0,-20,:,:]))
            ax[0, 0].set_title('target_time:{}'.format(configs.total_length-10))
            ax[0, 1].set_title('target_time:{}'.format(configs.total_length-20))
            ax[1, 0].set_title('prediction_time:{}'.format(configs.total_length-10))
            ax[1, 1].set_title('prediction_time:{}'.format(configs.total_length-20))
            ax[2, 0].set_title('error_time:{}'.format(configs.total_length-10))
            ax[2, 1].set_title('error_time:{}'.format(configs.total_length-20))

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

