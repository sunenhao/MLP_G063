import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConvolutionalNetwork(nn.Module):
    def __init__(self, input_shape, dim_reduction_type, num_output_classes, num_filters, num_layers, use_bias=False):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param dim_reduction_type: The type of dimensionality reduction to apply after each convolutional stage, should be one of ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        super(MyConvolutionalNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_conv_layers = 6
        self.num_full_layers = 2
        self.num_layers = num_layers
        self.dim_reduction_type = dim_reduction_type
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        self.dim_reduction_layer = [0]*self.num_conv_layers
        # build the network
        self.build_module()

    def conv_layer(self, in_channel, layer_i, feature_map, out):
            self.layer_dict['conv_{}'.format(layer_i)] = nn.Conv2d(in_channels=in_channel,
                                                             # add a conv layer in the module dict
                                                             kernel_size=3,
                                                             out_channels=feature_map, padding=0,
                                                             bias=self.use_bias)
            out = self.layer_dict['conv_{}'.format(layer_i)](out)  # use layer on inputs to get an output
            out = F.relu(out)  # apply relu
            print(out.shape)
            return out

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        print("Building basic block of ConvolutionalNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))  # create dummy inputs to be used to infer shapes of layers
        out = x
#------------------------conv layer-----------------------------------------------------------------
        out = self.conv_layer(out.shape[1], 0, 64, out)
        out = self.conv_layer(out.shape[1], 1, 64, out)
        self.layer_dict['dim_reduction_max_pool_{}'.format(1)] = nn.MaxPool2d(3, stride=2)
        out = self.layer_dict['dim_reduction_max_pool_{}'.format(1)](out)
        self.dim_reduction_layer[1] = 1
        print(out.shape)

        out = self.conv_layer(out.shape[1], 2, 128, out)
        out = self.conv_layer(out.shape[1], 3, 128, out)
        out = self.conv_layer(out.shape[1], 4, 128, out)
        self.layer_dict['dim_reduction_max_pool_{}'.format(4)] = nn.MaxPool2d(3, stride=2)
        out = self.layer_dict['dim_reduction_max_pool_{}'.format(4)](out)
        self.dim_reduction_layer[4] = 1
        print(out.shape)


        out = self.conv_layer(out.shape[1], 5, 256, out)
        self.layer_dict['dim_reduction_max_pool_{}'.format(5)] = nn.MaxPool2d(3, stride=2)
        out = self.layer_dict['dim_reduction_max_pool_{}'.format(5)](out)
        self.dim_reduction_layer[5] = 1
        print(out.shape)

        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out,
                                        2)  # apply adaptive pooling to make sure output of conv layers is always (2, 2) spacially (helps with comparisons).
        print('shape before final linear layer', out.shape)
        out = out.view(out.shape[0], -1)

#-----------------------fully connected layer-------------------------------------------------------
        self.layer_dict['fcc_{}'.format(6)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=128,
                                            bias=self.use_bias)
        out = self.layer_dict['fcc_{}'.format(6)](out)  # apply ith fcc layer to the previous layers outputs
        out = F.relu(out)  # apply a ReLU on the outputs

        self.layer_dict['fcc_{}'.format(7)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=64,
                                            bias=self.use_bias)
        out = self.layer_dict['fcc_{}'.format(7)](out)  # apply ith fcc layer to the previous layers outputs
        out = F.relu(out)  # apply a ReLU on the outputs

        self.softmax_layer = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.num_output_classes,
                                            bias=self.use_bias)
        out = self.softmax_layer(out)  # apply ith fcc layer to the previous layers outputs
        out = F.softmax(out, dim=0)  # apply a softmax on the outputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = x
        for i in range(self.num_conv_layers):  # for number of layers

            out = self.layer_dict['conv_{}'.format(i)](out)  # pass through conv layer indexed at i
            out = F.relu(out)  # pass conv outputs through ReLU

            if self.dim_reduction_layer[i] == 1:
                    if self.dim_reduction_type == 'strided_convolution':  # if strided convolution dim reduction then
                        out = self.layer_dict['dim_reduction_strided_conv_{}'.format(i)](out)  # pass previous outputs through a strided convolution indexed i
                        out = F.relu(out)  # pass strided conv outputs through ReLU
                    elif self.dim_reduction_type == 'dilated_convolution':
                        out = self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)](out)
                        out = F.relu(out)
                    elif self.dim_reduction_type == 'max_pooling':
                        out = self.layer_dict['dim_reduction_max_pool_{}'.format(i)](out)
                    elif self.dim_reduction_type == 'avg_pooling':
                        out = self.layer_dict['dim_reduction_avg_pool_{}'.format(i)](out)

        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out, 2)
        out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)

        for i in range(self.num_full_layers):
            j = i + self.num_conv_layers
            out = self.layer_dict['fcc_{}'.format(j)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        out = self.softmax_layer(out)  # pass through a linear layer to get logits/preds
        out = F.softmax(out, dim=0)
        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

        #self.logit_linear_layer.reset_parameters()
        self.softmax_layer.reset_parameters()
