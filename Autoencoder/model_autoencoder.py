import torch
import torch.nn as nn
import numpy as np
### single layers construction
def conv2d_bn_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

def conv2d_bn_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

class LCA(torch.nn.Module):
    def __init__(self, in_channels):
        super(LCA,self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(in_channels,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        # input (Nin, Cin, Hin, Win)
        # output (Nin, 32, Hin/2, Win/2)
        
        
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        # input (Nin, 32, Hin/2, Win/2)
        # output (Nin, 32, Hin/4, Win/4)
        
        
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,4,stride=2),
            conv2d_bn_relu(64,64,3)
        )
        # input (Nin, 32, Hin/4, Win/4)
        # output (Nin, 64, Hin/8, Win/8)
        
        
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64,128,4,stride=2),
            conv2d_bn_relu(128,128,3),
        )
        # input (Nin, 64, Hin/8, Win/8)
        # output (Nin, 128, Hin/16, Win/16)
        
        
        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(128,128,4,stride=2),
            conv2d_bn_relu(128,128,3),
        )
        # input (Nin, 128, Hin/16, Win/16)
        # output (Nin, 128, Hin/32, Win/32)
        
        ## decoder layer
        self.deconv_5 = deconv_relu(128,64,4,stride=2)
        self.deconv_4 = deconv_relu(67,64,4,stride=2)
        self.deconv_3 = deconv_relu(67,32,4,stride=2)
        self.deconv_2 = deconv_relu(35,16,4,stride=2)
        self.deconv_1 = deconv_sigmoid(19,in_channels,4,stride=2)

        self.predict_5 = torch.nn.Conv2d(128,3,3,stride=1,padding=1)
        self.predict_4 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_3 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_2 = torch.nn.Conv2d(35,3,3,stride=1,padding=1)

        self.up_sample_5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )


    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        
        # input (Nin, Cin, Hin, Win)
        # output (Nin, 128, Hin/32, Win/32)
        
        return conv5_out

    def decoder(self, x):
        deconv5_out = self.deconv_5(x)
        #input (Nin, 128, Hin/32, Win/32)
        #output (Nin, 64, Hin/16, Win/16)
        
        predict_5_out = self.up_sample_5(self.predict_5(x))
        #input (Nin, 128, Hin/16, Win/32)
        # After self.predict_5()
            #output (Nin, 3, Hin/32, Win/32)
        # After self.up_sample_5()
            #output (Nin, 3, Hin/16, Win/16)
        concat_5 = torch.cat([deconv5_out, predict_5_out],dim=1)
        # shape = (Nin, 67, Hin/16, Win/16)
        
        deconv4_out = self.deconv_4(concat_5)
        #input (Nin, 67, Hin/16, Win/16)
        #output (Nin, 64, Hin/8, Win/8)
        
        predict_4_out = self.up_sample_4(self.predict_4(concat_5))
        #input (Nin, 67, Hin/16, Win/16)
        # After self.predict_4()
            #output (Nin, 3, Hin/16, Win/16)
        # After self.up_sample_4()
            #output (Nin, 3, Hin/8, Win/8)
        concat_4 = torch.cat([deconv4_out,predict_4_out],dim=1)
        # shape = (Nin, 67, Hin/8, Win/8)
        
        deconv3_out = self.deconv_3(concat_4)
        #input (Nin, 67, Hin/8, Win/8)
        #output (Nin, 32, Hin/4, Win/4)
        
        predict_3_out = self.up_sample_3(self.predict_3(concat_4))
        #input (Nin, 67, Hin/8, Win/8)
        # After self.predict_3()
            #output (Nin, 3, Hin/8, Win/8)
        # After self.up_sample_3()
            #output (Nin, 3, Hin/4, Win/4)
        
        concat2 = torch.cat([deconv3_out,predict_3_out],dim=1)
        # shape = (Nin, 35, Hin/4, Win/4)
        
        deconv2_out = self.deconv_2(concat2)
        #input (Nin, 35, Hin/4, Win/4)
        #output (Nin, 16, Hin/2, Win/2)
        
        predict_2_out = self.up_sample_2(self.predict_2(concat2))
         #input (Nin, 35, Hin/4, Win/4)
         # After self.predict_2()
             #output (Nin, 3, Hin/4, Win/4)
         # After self.up_sample_2()
             #output (Nin, 3, Hin/2, Win/2)

        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        # shape = (Nin, 19, Hin/2, Win/2)
        
        predict_out = self.deconv_1(concat1)
        #input (Nin, 19, Hin/2, Win/2)
        #output (Nin, Cin, Hin, Win)
        
        return predict_out
    
    def forward(self,x):
        # input (Nin, Cin, Hin, Win)
        # latent (Nin, 128, Hin/32, Win/32)
        # output (Nin, Cin, Hin, Win)
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out, latent

class HCA(torch.nn.Module):
    def __init__(self, in_channels):
        super(HCA,self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(in_channels,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,4,stride=2),
            conv2d_bn_relu(64,64,3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(64,128,4,stride=2),
            conv2d_bn_relu(128,128,3),
        )

        self.conv_stack6 = torch.nn.Sequential(
            conv2d_bn_relu(128,128,4,stride=2),
            conv2d_bn_relu(128,128,3),
        )

        self.conv_stack7 = torch.nn.Sequential(
            conv2d_bn_relu(128,256,4,stride=2),
            conv2d_bn_relu(256,256,3),
        )

        self.conv_stack8 = torch.nn.Sequential(
            conv2d_bn_relu(256, 256, 4, stride=2),
            conv2d_bn_relu(256, 256, 3),
        )
        
        self.deconv_8 = deconv_relu(256,256,4,stride=2)
        self.deconv_7 = deconv_relu(259,128,4,stride=2)
        self.deconv_6 = deconv_relu(131,128,4,stride=2)
        self.deconv_5 = deconv_relu(131,64,4,stride=2)
        self.deconv_4 = deconv_relu(67,64,4,stride=2)
        self.deconv_3 = deconv_relu(67,32,4,stride=2)
        self.deconv_2 = deconv_relu(35,16,4,stride=2)
        self.deconv_1 = deconv_sigmoid(19,in_channels,4,stride=2)

        self.predict_8 = torch.nn.Conv2d(256,3,3,stride=1,padding=1)
        self.predict_7 = torch.nn.Conv2d(259,3,3,stride=1,padding=1)
        self.predict_6 = torch.nn.Conv2d(131,3,3,stride=1,padding=1)
        self.predict_5 = torch.nn.Conv2d(131,3,3,stride=1,padding=1)
        self.predict_4 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_3 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_2 = torch.nn.Conv2d(35,3,3,stride=1,padding=1)

        self.up_sample_8 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_7 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )


    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        conv6_out = self.conv_stack6(conv5_out)
        conv7_out = self.conv_stack7(conv6_out)
        conv8_out = self.conv_stack8(conv7_out)
        
        # input (Nin, Cin, Hin, Win)
        # After conv_stack1 (Nin, 32, Hin/2, Win/2)
        # After conv_stack2 (Nin, 32, Hin/4, Win/4)
        # After conv_stack3 (Nin, 64, Hin/8, Win/8)
        # After conv_stack4 (Nin, 64, Hin/16, Win/16)
        # After conv_stack5 (Nin, 128, Hin/32, Win/32)
        # After conv_stack6 (Nin, 128, Hin/64, Win/64)
        # After conv_stack7 (Nin, 256, Hin/128, Win/128)
        # After conv_stack8 (Nin, 256, Hin/256, Win/256)
        
        return conv8_out

    def decoder(self, x):
        deconv8_out = self.deconv_8(x)
        #input (Nin, 256, Hin/256, Win/256)
        #output (Nin, 256, Hin/128, Win/128)
        
        predict_8_out = self.up_sample_8(self.predict_8(x))
        #input (Nin, 256, Hin/256, Win/256)
        #output (Nin, 3, Hin/128, Win/128)
        concat_7 = torch.cat([deconv8_out, predict_8_out], dim=1)
        # shape (Nin, 259, Hin/128, Win/128)
        
        
        deconv7_out = self.deconv_7(concat_7)
        #input (Nin, 259, Hin/128, Win/128)
        #output (Nin, 256, Hin/64, Win/64)
        
        predict_7_out = self.up_sample_7(self.predict_7(concat_7))
        #input (Nin, 259, Hin/128, Win/128)
        #output (Nin, 3, Hin/64, Win/64)
        concat_6 = torch.cat([deconv7_out,predict_7_out],dim=1)
        # shape (Nin, 259, Hin/64, Win/64)
        
        deconv6_out = self.deconv_6(concat_6)
        predict_6_out = self.up_sample_6(self.predict_6(concat_6))
        concat_5 = torch.cat([deconv6_out,predict_6_out],dim=1)
        # shape (Nin, 131, Hin/32, Win/32)
        
        deconv5_out = self.deconv_5(concat_5)
        #input (Nin, 131, Hin/32, Win/32)
        #output (Nin, 64, Hin/16, Win/16)
        predict_5_out = self.up_sample_5(self.predict_5(concat_5))
        #input (Nin, 131, Hin/32, Win/32)
        #output (Nin, 3, Hin/16, Win/16)
        concat_4 = torch.cat([deconv5_out,predict_5_out],dim=1)
        # # shape (Nin, 67, Hin/16, Win/16)
        
        deconv4_out = self.deconv_4(concat_4)
        predict_4_out = self.up_sample_4(self.predict_4(concat_4))
        concat_3 = torch.cat([deconv4_out,predict_4_out],dim=1)
        # # shape (Nin, 67, Hin/8, Win/8)
        
        
        deconv3_out = self.deconv_3(concat_3)
        predict_3_out = self.up_sample_3(self.predict_3(concat_3))
        concat2 = torch.cat([deconv3_out,predict_3_out],dim=1)
        # shape (Nin, 35, Hin/4, Win/4)
        
        
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))
        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        # shape (Nin, 19, Hin/2, Win/2)
        
        predict_out = self.deconv_1(concat1)
        # shape (Nin, in_channels, Hin, Win)
        
        return predict_out
        

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        # input (Nin, Cin, Hin, Win)
        # latent (Nin, 128, Hin/256, Win/256)
        # output (Nin, Cin, Hin, Win)
        return out, latent
    
    



















