import numpy as np
import torch
import torch.nn as nn
#from torchsummary import summary
from torchvision import transforms, io
from torch.utils.data import Dataset, DataLoader
#from PIL import Image
import os
#import pandas as pd
#import imageio
import argparse
#import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_images():
    # get manually collected data
    path = '/output/'
    image_path = os.path.join(path, './rgb/')
    seg_path   = os.path.join(path, './seg/')
    image_list = os.listdir(image_path)
    seg_list   = os.listdir(seg_path)
    image_list = [image_path+i for i in image_list]
    seg_list = [seg_path+i for i in seg_list]
    return image_list, seg_list


class Segmentation_Dataset(Dataset):
    def __init__(self, img_path, seg_path, h, w):
        # getting the data from the paths
        self.img_path = img_path
        self.seg_path = seg_path
        self.images = os.listdir(self.img_path)
        self.masks = os.listdir(self.seg_path)
        self.h = h
        self.w = w

        # transforming images into desired input size
        self.image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.h, self.w), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[0:3])
        ])

        self.mask_transforms = transforms.Compose([
            transforms.Resize((self.h, self.w), interpolation=transforms.InterpolationMode.NEAREST)
        ])

    # need these functions for pytorch dataset
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        # get rbg image
        img_file = os.path.join(self.img_path, img_name)
        img = io.read_file(img_file)
        img = io.decode_png(img)
        # and the segmentation mask
        seg_file = os.path.join(self.seg_path, img_name)
        mask = io.read_image(seg_file)

        # this collapses rgb channels into a single dimension mask 
        mask, _ = torch.max(mask[0:3], dim=0, keepdim=True)

        img, mask = self.image_transforms(img), self.mask_transforms(mask)
        return {"IMAGE": img, "MASK": mask}
    

# U-net encoder consists of several convolutional blocks that typically have the same layout:
# kernel size 3x3 with 1 padding to maintain input size with each filter
# reLU activation
# maxpool of kernel size 2x2 stride 2, to decrease resolution by 0.5 in both dimensions
# dropout
# i added batchnorm just in case we want to try that
class convStack(nn.Module):
    def __init__(self, in_channels, n_filters=32, p_dropout=0, maxpool=True, batchNorm=False):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, n_filters, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters) # batchnorm occurs after convolution
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters) # here too
        self.activation = nn.ReLU()
        self.maxpool = maxpool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if self.max_pooling else None
        self.p_dropout = p_dropout
        self.dropout = nn.Dropout(p=p_dropout)
        self.batchNorm = batchNorm


    def forward(self, x):
            # forward through the convolutional block
            out = self.conv_1(x)
            # do batchnorm if set to true
            if self.batchNorm:
                out = self.bn1(out)
            out = self.activation(out)
            out = self.conv_2(out)
            # do batchnorm if set to true
            if self.batchNorm:
                out = self.bn2(out)
            out = self.activation(out)
            # apply dropout if set
            if self.p_dropout > 0:
                out = self.dropout(out)

            # save the skip connection here before maxpooling (remember we wanna keep it the original size)
            skip_connection = out.clone()
            # ok now we can maxpool
            if self.maxpool:
                out = self.pool(out)
            next_layer = out
            # return both 
            return next_layer, skip_connection



# now we need the decoder
# similar to encoder block (kernel sizes, sampling rates) except we usually dont use dropout here since we want all the info we can obtain
# we upsample, then we add the skip inputs from the other side of the "U", and filter again
# then activation
# decoder usually doesnt have batchnorm because..
# we're not searching for the features anymore, we're building the mask
# added it in just in case i wanna experiment with it though
class decoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, n_filters=32, batchNorm=False):
        super().__init__()
        # the upsampler
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=(2, 2), stride=2, padding=0)
        # heres where the skip connection is added in
        self.conv_1 = nn.Conv2d(in_channels // 2 + skip_channels, n_filters, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.activation = nn.ReLU()
        self.batchNorm = batchNorm
    
    def forward(self, input, skip_input):

        out = self.upsample(input)
        #where we concatenate the skip connection
        out = torch.cat([out, skip_input], dim=1) 
        out = self.conv_1(out)

        if self.batchNorm:
                out = self.bn1(out)

        out = self.activation(out)
        out = self.conv_2(out)

        if self.batchNorm:
                out = self.bn2(out)

        out = self.activation(out)
        return out



# U-Net in action
# as seen in class its a series of convolutional blocks/stacks followed by a series of upsampling
# the number of filters increases alot towards the middle of the network
# 
class CARLA_UNet(nn.Module):
    def __init__(self, in_channels=3, n_filters=32, n_classes=23, batchNorm=False):
     
        # notice dropout is usually done in the denser middle part where we have lots of filters and low level features
        # not a good idea to do it in other places or in upsampling
        super().__init__()
        self.conv_layer_1 = convStack(in_channels, n_filters, batchNorm=batchNorm)
        self.conv_layer_2 = convStack(n_filters, n_filters*2, batchNorm=batchNorm)
        self.conv_layer_3 = convStack(n_filters*2,  n_filters*4, batchNorm=batchNorm)
        self.conv_layer_4 = convStack(n_filters*4, n_filters*8, p_dropout=0.3, batchNorm=batchNorm)
        # final encoder block doesnt maxpool again because now we're going back up in resolution
        self.conv_layer_5 = convStack(n_filters*8, n_filters*16, p_dropout=0.3, max_pooling=False, batchNorm=batchNorm)

        self.upsample_1 = decoderBlock(n_filters*16, n_filters*8, n_filters * 8, batchNorm=batchNorm)
        self.upsample_2 = decoderBlock(n_filters*8, n_filters*4, n_filters * 4, batchNorm=batchNorm)
        self.upsample_3 = decoderBlock(n_filters*4, n_filters*2, n_filters * 2, batchNorm=batchNorm)
        self.upsample_4 = decoderBlock(n_filters*2, n_filters*1, n_filters * 1, batchNorm=batchNorm)

        # and then one last set of sequential for good measure
        self.last_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters,  kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_classes,  kernel_size=(1, 1), padding=0),
        )

    def forward(self, x):
   
        # see now the skip connection is reintroduced at the proper resolution
        # input resolution
        conv_1_next, conv_1_skip = self.conv_layer_1(x)
        # after one maxpol
        conv_2_next, conv_2_skip = self.conv_layer_2(conv_1_next)
        # after two maxpools
        conv_3_next, conv_3_skip = self.conv_layer_3(conv_2_next)
        # after three maxpools
        conv_4_next, conv_4_skip = self.conv_layer_4(conv_3_next)
        # after four maxpools
        conv_5_next, conv_5_skip = self.conv_layer_5(conv_4_next)
        

        # combine the output with the previous skip corresponding to the right resolution
        out = self.upsample_layer_1(conv_5_next, conv_4_skip)
        # and again
        out = self.upsample_layer_2(out, conv_3_skip)
        # and again
        out = self.upsample_layer_3(out, conv_2_skip)
        # and again
        out = self.upsample_layer_4(out, conv_1_skip)
        # and one final convolution for good measure
        out = self.last_conv(out)
        return out
        


# som hyperparams

# EPOCHS = 30
# BATCH_SIZE = 32
# LR = 0.001
# B1 = 0.9
# B2 = 0.999

# main training function

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--b1", type=float, default=0.9, help="Adam beta1") # im probably not gonna mess with these
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2") # this neither

    args = parser.parse_args()

    # get paths for rbg images and segmentation masks
    path = '/output/'
    image_path = os.path.join(path, './rgb/')
    seg_path   = os.path.join(path, './seg/')

    # set up the dataloader
    dataloader = DataLoader(Segmentation_Dataset(image_path, seg_path), batch_size=args.batchSize, shuffle=True)

    # now set up the model
    unet = CARLA_UNet().to(device)
    criterion = nn.CrossEntropyLoss()
    # adam optimizer
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr, betas=(args.b1, args.b2))


    # training loop

    losses = []
    for epoch in range(args.epochs):
        epoch_losses = []
        for i, batch in enumerate(dataloader):
            images = batch['IMAGE'].to(device)
            masks = batch['MASK'].to(device)

            N, C, H, W = masks.shape
            masks = masks.reshape((N, H, W)).long()

            optimizer.zero_grad()
            
            outputs = unet(images)

            # compute loss
            loss = criterion(outputs, masks)
            epoch_losses.append(loss.item() * images.size(0))

            # backprop
            loss.backward()
            optimizer.step()


            # print how many samples its gone thru
            currSamples = min((i + 1) * args.batchSize, len(dataloader.dataset))
            print(f'EPOCH {epoch} ({currSamples}/{len(dataloader.dataset)})  \t Loss:{loss.item()}')
        losses.append(np.mean(epoch_losses) / len(dataloader.dataset))

        # save state dict after every epoch in case of crashes
        torch.save(unet.state_dict(), f"unet_{epoch}.pth")



    


    




















