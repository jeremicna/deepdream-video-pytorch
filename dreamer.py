import os
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from neural_dream.CaffeLoader import loadCaffemodel, ModelParallel
import neural_dream.dream_utils as dream_utils
import neural_dream.loss_layers as dream_loss_layers
import neural_dream.dream_image as dream_image
import neural_dream.dream_model as dream_model
import neural_dream.dream_tile as dream_tile
from neural_dream.dream_auto import auto_model_mode, auto_mean

import argparse
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument("-content_image", help="Content target image", default='examples/inputs/tubingen.jpg')
parser.add_argument("-image_size", help="Maximum height / width of generated image", type=int, default=512)
parser.add_argument("-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = c", default=0)

# Optimization options
parser.add_argument("-dream_weight", type=float, default=1000)
parser.add_argument("-normalize_weights", action='store_true')
parser.add_argument("-tv_weight", type=float, default=0)
parser.add_argument("-l2_weight", type=float, default=0)
parser.add_argument("-num_iterations", type=int, default=10)
parser.add_argument("-jitter", type=int, default=32)
parser.add_argument("-init", choices=['random', 'image'], default='image')
parser.add_argument("-optimizer", choices=['lbfgs', 'adam'], default='adam')
parser.add_argument("-learning_rate", type=float, default=1.5)
parser.add_argument("-lbfgs_num_correction", type=int, default=100)
parser.add_argument("-loss_mode", choices=['bce', 'mse', 'mean', 'norm', 'l2', 'abs_mean', 'abs_l2'], default='l2')

# Output options
parser.add_argument("-print_iter", type=int, default=1)
parser.add_argument("-print_octave_iter", type=int, default=0)
parser.add_argument("-save_iter", type=int, default=1)
parser.add_argument("-save_octave_iter", type=int, default=0)
parser.add_argument("-output_image", default='out.png')
parser.add_argument("-output_start_num", type=int, default=1)

# Octave options
parser.add_argument("-num_octaves", type=int, default=2)
parser.add_argument("-octave_scale", default='0.6')
parser.add_argument("-octave_iter", type=int, default=50)
parser.add_argument("-octave_mode", choices=['normal', 'advanced', 'manual_max', 'manual_min', 'manual'], default='normal')

# Channel options
parser.add_argument("-channels", type=str, help="channels for DeepDream", default='-1')
parser.add_argument("-channel_mode", choices=['all', 'strong', 'avg', 'weak', 'ignore'], default='all')
parser.add_argument("-channel_capture", choices=['once', 'iter'], default='once')

# Guassian Blur options
parser.add_argument("-layer_sigma", type=float, default=0)

# Laplacian pyramid options
parser.add_argument("-lap_scale", type=int, default=0)
parser.add_argument("-sigma", default='1')

# FFT options
parser.add_argument("-use_fft", action='store_true')
parser.add_argument("-fft_block", type=int, default=25)

# Zoom options
parser.add_argument("-zoom", type=int, default=0)
parser.add_argument("-zoom_mode", choices=['percent', 'pixel'], default='percent')
parser.add_argument("-leading_zeros", type=int, default=0)

# Tile options
parser.add_argument("-tile_size", type=int, default=0)
parser.add_argument("-overlap_percent", type=float, default=0.5)
parser.add_argument("-print_tile", type=int, default=0)
parser.add_argument("-disable_roll", action='store_true')
parser.add_argument("-print_tile_iter", type=int, default=0)
parser.add_argument("-image_capture_size", help="Image size for initial capture, and classification", type=int, default=512)

# Gif options
parser.add_argument("-create_gif", action='store_true')
parser.add_argument("-frame_duration", type=int, default=100)

# Other options
parser.add_argument("-original_colors", type=int, choices=[0, 1], default=0)
parser.add_argument("-pooling", choices=['avg', 'max'], default='max')
parser.add_argument("-model_file", type=str, default='models/bvlc_googlenet.pth')
parser.add_argument("-model_type", choices=['caffe', 'pytorch', 'keras', 'auto'], default='auto')
parser.add_argument("-model_mean", default='auto')
parser.add_argument("-label_file", type=str, default='')
parser.add_argument("-disable_check", action='store_true')
parser.add_argument("-backend", choices=['nn', 'cudnn', 'mkl', 'mkldnn', 'openmp', 'mkl,cudnn', 'cudnn,mkl'], default='nn')
parser.add_argument("-cudnn_autotune", action='store_true')
parser.add_argument("-seed", type=int, default=-1)
parser.add_argument("-clamp", action='store_true')
parser.add_argument("-random_transforms", choices=['none', 'all', 'flip', 'rotate'], default='none')
parser.add_argument("-adjust_contrast", type=float, help="try 99.98", default=-1)
parser.add_argument("-classify", type=int, default=0)

parser.add_argument("-dream_layers", help="layers for DeepDream", default='inception_4d_3x3_reduce')

parser.add_argument("-multidevice_strategy", default='4,7,29')

# Help options
parser.add_argument("-print_layers", action='store_true')
parser.add_argument("-print_channels", action='store_true')

# Experimental params
parser.add_argument("-norm_percent", type=float, default=0)
parser.add_argument("-abs_percent", type=float, default=0)
parser.add_argument("-mean_percent", type=float, default=0)
parser.add_argument("-percent_mode", choices=['slow', 'fast'], default='fast')
params = parser.parse_args()


Image.MAX_IMAGE_PIXELS = 1000000000
params = None

class DeepDreamer:
    def __init__(self, arg_list=None):
        global params
        if arg_list is not None:
            params = parser.parse_args(arg_list)
        else:
            params = parser.parse_args()
            
        self.params = params

        self.dtype, self.multidevice, self.backward_device = setup_gpu()

        cnn, layerList = loadCaffemodel(self.params.model_file, self.params.pooling, self.params.gpu, self.params.disable_check, True)
        self.has_inception = cnn.has_inception
        
        if self.params.print_layers:
            print_layers(layerList, self.params.model_file, self.has_inception)

        self.params.model_type = auto_model_mode(self.params.model_file) if self.params.model_type == 'auto' else self.params.model_type
        self.input_mean = auto_mean(self.params.model_file, self.params.model_type) if self.params.model_mean == 'auto' else self.params.model_mean
        if self.params.model_mean != 'auto':
            self.input_mean = [float(m) for m in self.input_mean.split(',')]

        self.clamp_val = 256 if self.params.model_type == 'caffe' else 1
        
        if self.params.label_file != '':
            labels = load_label_file(self.params.label_file)
            self.params.channels = channel_ids(labels, self.params.channels)
        if self.params.classify > 0:
             if not self.has_inception:
                 self.params.dream_layers += ',classifier'
             if self.params.label_file == '':
                 labels = list(range(0, 1000))

        dream_layers = self.params.dream_layers.split(',')
        start_params = (self.dtype, self.params.random_transforms, self.params.jitter, self.params.tv_weight, self.params.l2_weight, self.params.layer_sigma)
        primary_params = (self.params.loss_mode, self.params.dream_weight, self.params.channels, self.params.channel_mode)
        secondary_params = {'channel_capture': self.params.channel_capture, 'scale': self.params.lap_scale, 'sigma': self.params.sigma, \
        'use_fft': (self.params.use_fft, self.params.fft_block), 'r': self.clamp_val, 'p_mode': self.params.percent_mode, 'norm_p': self.params.norm_percent, \
        'abs_p': self.params.abs_percent, 'mean_p': self.params.mean_percent}

        self.net_base, self.dream_losses, self.tv_losses, self.l2_losses, self.lm_layer_names, self.loss_module_list = dream_model.build_net(cnn, dream_layers, \
        self.has_inception, layerList, self.params.classify, start_params, primary_params, secondary_params)
        
        self.net_base = self.net_base.to(self.backward_device)

        if self.params.classify > 0:
            self.classify_img = dream_utils.Classify(labels, self.params.classify)

        if self.multidevice and not self.has_inception:
            self.net_base = setup_multi_device(self.net_base)

        if not self.has_inception:
            print_torch(self.net_base, self.multidevice)

        for param in self.net_base.parameters():
            param.requires_grad = False

    def dream(self, input_image_path, output_image_path):
        self.params.output_image = output_image_path
        output_start_num = self.params.output_start_num - 1 if self.params.output_start_num > 0 else 0
        
        content_image = preprocess(input_image_path, self.params.image_size, self.params.model_type, self.input_mean).to(self.backward_device)

        if self.params.seed >= 0:
            torch.manual_seed(self.params.seed)
            torch.cuda.manual_seed_all(self.params.seed)
            torch.backends.cudnn.deterministic=True
            random.seed(self.params.seed)
            
        if self.params.init == 'random':
            base_img = torch.randn(content_image.size(), device=self.backward_device).mul(0.001)
        elif self.params.init == 'image':
            base_img = content_image.clone()

        for i in self.dream_losses:
            i.mode = 'capture'

        if self.params.image_capture_size == -1:
            self.net_base(base_img.clone())
        else:
            image_capture_size = tuple([int((float(self.params.image_capture_size) / max(base_img.size()))*x) for x in (base_img.size(2), base_img.size(3))])
            self.net_base(dream_image.resize_tensor(base_img.clone(), (image_capture_size)))

        if self.params.channels != '-1' or self.params.channel_mode != 'all' and self.params.channels != '-1':
            print_channels(self.dream_losses, self.params.dream_layers.split(','), self.params.print_channels)
            
        if self.params.classify > 0:
            pass

        for i in self.dream_losses:
            i.mode = 'None'

        current_img = base_img.clone()
        h, w = current_img.size(2), current_img.size(3)
        total_dream_losses, total_loss = [], [0]

        octave_list = octave_calc((h,w), self.params.octave_scale, self.params.num_octaves, self.params.octave_mode)

        for iter in range(1, self.params.num_iterations+1):
            for octave, octave_sizes in enumerate(octave_list, 1):
                net = self.net_base
                dream_losses, tv_losses, l2_losses = [], [], []
                
                if not self.has_inception:
                    for i, layer in enumerate(net):
                        if isinstance(layer, dream_loss_layers.TVLoss): tv_losses.append(layer)
                        if isinstance(layer, dream_loss_layers.L2Regularizer): l2_losses.append(layer)
                        if 'DreamLoss' in str(type(layer)): dream_losses.append(layer)
                elif self.has_inception:
                    net, dream_losses, tv_losses, l2_losses = dream_model.renew_net(
                        (self.dtype, self.params.random_transforms, self.params.jitter, self.params.tv_weight, self.params.l2_weight, self.params.layer_sigma), 
                        net, self.loss_module_list, self.lm_layer_names)

                img = new_img(current_img.clone(), octave_sizes)
                net(img)
                
                for i in dream_losses: i.mode = 'loss'
                if self.params.normalize_weights: normalize_weights(dream_losses)
                for param in net.parameters(): param.requires_grad = False

                num_calls = [0]
                def feval():
                    num_calls[0] += 1
                    optimizer.zero_grad()
                    net(img)
                    loss = 0
                    for mod in dream_losses: loss += -mod.loss.to(self.backward_device)
                    if self.params.tv_weight > 0:
                        for mod in tv_losses: loss += mod.loss.to(self.backward_device)
                    if self.params.l2_weight > 0:
                        for mod in l2_losses: loss += mod.loss.to(self.backward_device)
                    
                    if self.params.clamp: img.clamp(0, self.clamp_val)
                    if self.params.adjust_contrast > -1:
                        img.data = dream_image.adjust_contrast(img, r=self.clamp_val, p=self.params.adjust_contrast)
                    
                    total_loss[0] += loss.item()
                    loss.backward()

                    return loss

                optimizer, loopVal = setup_optimizer(img)
                while num_calls[0] <= self.params.octave_iter:
                    optimizer.step(feval)

                if octave == 1:
                     for mod in dream_losses: total_dream_losses.append(mod.loss.item())
                else:
                     for d_loss, mod in enumerate(dream_losses): total_dream_losses[d_loss] += mod.loss.item()

                if img.size(2) != h or img.size(3) != w:
                    current_img = dream_image.resize_tensor(img.clone(), (h,w))
                else:
                    current_img = img.clone()

            maybe_print(iter, total_loss[0], total_dream_losses)
            save_output(iter, current_img, content_image, "", self.input_mean, no_num=True)
            total_dream_losses, total_loss = [], [0]
            
            if self.params.zoom > 0:
                current_img = dream_image.zoom(current_img, self.params.zoom, self.params.zoom_mode)

def save_output(t, save_img, content_image, iter_name, model_mean, no_num=False):
    output_filename, file_extension = os.path.splitext(params.output_image)
    if t == params.num_iterations and not no_num:
        filename = output_filename + str(file_extension)
    else:
        filename = str(output_filename) + iter_name + str(file_extension)
    disp = deprocess(save_img.clone(), params.model_type, model_mean)

    if params.original_colors == 1:
        disp = original_colors(deprocess(content_image.clone(), params.model_type, model_mean), disp)

    disp.save(str(filename))

    if t == params.num_iterations and params.create_gif:
        dream_image.create_gif(output_filename, params.frame_duration)


def maybe_save(t, save_img, content_image, input_mean, start_num, leading_zeros):
    should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save:
        no_num = True if leading_zeros > 0 else False
        save_output(t, save_img, content_image, "_" + str(t+start_num).zfill(leading_zeros), input_mean, no_num)


def maybe_save_octave(t, n, o, save_img, content_image, input_mean):
    should_save = params.save_octave_iter > 0 and n % params.save_octave_iter == 0
    should_save = should_save or params.save_octave_iter > 0 and n == params.octave_iter
    if o == params.num_octaves:
        should_save = False if params.save_iter > 0 and t % params.save_iter == 0 or t == params.num_iterations else should_save
    if should_save:
        save_output(t, save_img, content_image, "_" + str(t) + "_" + str(o) + "_" + str(n), input_mean)


def maybe_print(t, loss, dream_losses):
    if params.print_iter > 0 and t % params.print_iter == 0:
        print("Iteration " + str(t) + " / "+ str(params.num_iterations))
        for i, loss_module in enumerate(dream_losses):
            print("  DeepDream " + str(i+1) + " loss: " + str(loss_module))
        print("  Total loss: " + str(abs(loss)))


def maybe_print_octave_iter(t, n, total, dream_losses):
    if params.print_octave_iter > 0 and t % params.print_octave_iter == 0:
        print("Octave iter "+str(n) +" iteration " + str(t) + " / "+ str(total))
        for i, loss_module in enumerate(dream_losses):
            print("  DeepDream " + str(i+1) + " loss: " + str(loss_module.loss.item()))


def maybe_print_octave_tiled(t, n, octaves, dream_losses):
    if params.print_octave_iter > 0 and t % params.print_octave_iter == 0:
        print("Octave "+str(n) + " / "+ str(octaves))
        for i, loss_module in enumerate(dream_losses):
            print("  DeepDream " + str(i+1) + " loss: " + str(loss_module))


def maybe_print_tile_iter(tile, num_tiles, t, n, total, dream_losses):
    if params.print_tile_iter > 0 and t % params.print_tile_iter == 0:
        print("Tile " +str(tile+1) + " / " + str(num_tiles) + " iteration " + str(t) + " / "+ str(total))
        for i, loss_module in enumerate(dream_losses):
            print("  DeepDream " + str(i+1) + " loss: " + str(loss_module.loss.item()))


def maybe_print_tile(tile_num, num_tiles):
    if params.print_tile > 0 and (tile_num + 1) % params.print_tile == 0:
        print('Processing tile: ' + str(tile_num+1) + ' of ' + str(num_tiles))


def print_channels(dream_losses, layers, print_all_channels=False):
    print('\nSelected layer channels:')
    if not print_all_channels:
        for i, l in enumerate(dream_losses):
            if len(l.dream.channels) > 25:
                ch = l.dream.channels[0:25] + ['and ' + str(len(l.dream.channels[25:])) + ' more...']
            else:
                ch = l.dream.channels
            print('  ' + layers[i] + ': ', ch)
    elif print_all_channels:
        for i, l in enumerate(dream_losses):
            ch = l.dream.channels
            print('  ' + layers[i] + ': ', ch)


def setup_optimizer(img):
    if params.optimizer == 'lbfgs':
        optim_state = {
            'max_iter': params.num_iterations,
            'tolerance_change': -1,
            'tolerance_grad': -1,
            'lr': params.learning_rate
        }
        if params.lbfgs_num_correction != 100:
            optim_state['history_size'] = params.lbfgs_num_correction
        optimizer = optim.LBFGS([img], **optim_state)
        loopVal = 1
    elif params.optimizer == 'adam':
        optimizer = optim.Adam([img], lr = params.learning_rate)
        loopVal = params.num_iterations - 1
    return optimizer, loopVal


def setup_gpu():
    def setup_cuda():
        if 'cudnn' in params.backend:
            torch.backends.cudnn.enabled = True
            if params.cudnn_autotune:
                torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.enabled = False

    def setup_cpu():
        if 'mkl' in params.backend and 'mkldnn' not in params.backend:
            torch.backends.mkl.enabled = True
        elif 'mkldnn' in params.backend:
            raise ValueError("MKL-DNN is not supported yet.")
        elif 'openmp' in params.backend:
            torch.backends.openmp.enabled = True

    multidevice = False
    
    # MPS
    if str(params.gpu).lower() == 'mps':
        if torch.backends.mps.is_available():
            dtype = torch.float32 
            backward_device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            print("MPS not available, falling back to CPU")
            setup_cpu()
            dtype, backward_device = torch.FloatTensor, torch.device("cpu")
    
    # Multi-GPU CUDA
    elif "," in str(params.gpu):
        devices = params.gpu.split(',')
        multidevice = True
        if 'c' in str(devices[0]).lower():
            backward_device = torch.device("cpu")
            setup_cuda(); setup_cpu()
        else:
            backward_device = torch.device(f"cuda:{devices[0]}")
            setup_cuda()
        dtype = torch.FloatTensor # Fallback for multi-device logic handling

    # Single-GPU CUDA
    elif "c" not in str(params.gpu).lower():
        setup_cuda()
        dtype = torch.cuda.FloatTensor
        backward_device = torch.device(f"cuda:{params.gpu}")
    
    # CPU
    else:
        setup_cpu()
        dtype = torch.FloatTensor
        backward_device = torch.device("cpu")
    
    return dtype, multidevice, backward_device


def setup_multi_device(net_base):
    assert len(params.gpu.split(',')) - 1 == len(params.multidevice_strategy.split(',')), \
      "The number of -multidevice_strategy layer indices minus 1, must be equal to the number of -gpu devices."

    new_net_base = ModelParallel(net_base, params.gpu, params.multidevice_strategy)
    return new_net_base


def preprocess(image_name, image_size, mode='caffe', input_mean=[103.939, 116.779, 123.68]):
    image = Image.open(image_name).convert('RGB')
    if type(image_size) is not tuple:
        image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    if mode == 'caffe':
        rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
        Normalize = transforms.Compose([transforms.Normalize(mean=input_mean, std=[1,1,1])])
        tensor = Normalize(rgb2bgr(Loader(image) * 256)).unsqueeze(0)
    elif mode == 'pytorch':
        Normalize = transforms.Compose([transforms.Normalize(mean=input_mean, std=[1,1,1])])
        tensor = Normalize(Loader(image)).unsqueeze(0)
    elif mode == 'keras':
        tensor = ((Loader(image) - 0.5) * 2.0).unsqueeze(0)
    return tensor


def deprocess(output_tensor, mode='caffe', input_mean=[-103.939, -116.779, -123.68]):
    input_mean = [n * -1 for n in input_mean]
    if mode == 'caffe':
        Normalize = transforms.Compose([transforms.Normalize(mean=input_mean, std=[1,1,1])])
        bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
        output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0).cpu())) / 256
    elif mode == 'pytorch':
        Normalize = transforms.Compose([transforms.Normalize(mean=input_mean, std=[1,1,1])])
        output_tensor = Normalize(output_tensor.squeeze(0).cpu())
    elif mode == 'keras':
        output_tensor = ((output_tensor + 1.0) / 2.0).squeeze(0).cpu()
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor.cpu())
    return image


def original_colors(content, generated):
    content_channels = list(content.convert('YCbCr').split())
    generated_channels = list(generated.convert('YCbCr').split())
    content_channels[0] = generated_channels[0]
    return Image.merge('YCbCr', content_channels).convert('RGB')


def print_torch(net_base, multidevice):
    if multidevice:
        return
    simplelist = ""
    for i, layer in enumerate(net_base, 1):
        simplelist = simplelist + "(" + str(i) + ") -> "
    print("nn.Sequential ( \n  [input -> " + simplelist + "output]")

    def strip(x):
        return str(x).replace(", ",',').replace("(",'').replace(")",'') + ", "
    def n():
        return "  (" + str(i) + "): " + "nn." + str(l).split("(", 1)[0]

    for i, l in enumerate(net_base, 1):
         if "2d" in str(l):
             if "AdaptiveAvgPool2d" not in str(l) and "AdaptiveMaxPool2d" not in str(l) and "BasicConv2d" not in str(l):
                 ks, st, pd = strip(l.kernel_size), strip(l.stride), strip(l.padding)
             if "BasicConv2d" in str(l):
                 print(n())
             elif "Conv2d" in str(l):
                 ch = str(l.in_channels) + " -> " + str(l.out_channels)
                 print(n() + "(" + ch + ", " + (ks).replace(",",'x', 1) + st + pd.replace(", ",')'))
             elif "AdaptiveAvgPool2d" in str(l) or "AdaptiveMaxPool2d" in str(l):
                 print(n())
             elif "Pool2d" in str(l):
                 st = st.replace("  ",' ') + st.replace(", ",')')
                 print(n() + "(" + ((ks).replace(",",'x' + ks, 1) + st).replace(", ",','))
         else:
             print(n())
    print(")")


def print_octave_sizes(octave_list):
    print('\nPerforming ' + str(len(octave_list)) + ' octaves with the following image sizes:')
    for o, octave in enumerate(octave_list):
        print('  Octave ' + str(o+1) + ' image size: ' + \
        str(octave[0]) +'x'+ str(octave[1]))
    print()


def octave_calc(image_size, octave_scale, num_octaves, mode='normal'):
    octave_list = []
    h_size, w_size = image_size[0], image_size[1]
    if len(octave_scale.split(',')) == 1 and 'manual' not in mode:
        octave_scale = float(octave_scale)
    else:
        octave_scale = [int(o) for o in octave_scale.split(',')]
        if mode == 'manual':
            octave_scale = [octave_scale[o:o+2] for o in range(0, len(octave_scale), 2)]
    if mode == 'normal' or mode == 'advanced':
        assert octave_scale is not list, \
            "'-octave_mode normal' and '-octave_mode advanced' require a single float value."
    if mode == 'manual_max' or mode == 'manual_min':
        if type(octave_scale) is not list:
            octave_scale = [octave_scale]
        assert len(octave_scale) + 1 == num_octaves, \
            "Exected " + str(num_octaves - 1) + " octave sizes, but got " + str(len(octave_scale)) + " containing: " + str(octave_scale)

    if mode == 'normal':
        for o in range(1, num_octaves+1):
            h_size *= octave_scale
            w_size *= octave_scale
            if o < num_octaves:
                octave_list.append((int(h_size), int(w_size)))
        octave_list.reverse()
        octave_list.append((image_size[0], image_size[1]))
    elif mode == 'advanced':
        for o in range(1, num_octaves+1):
            h_size = image_size[0] * (o * octave_scale)
            w_size = image_size[1] * (o * octave_scale)
            octave_list.append((int(h_size), int(w_size)))
    elif mode == 'manual_max':
        for o in octave_scale:
            new_size = tuple([int((float(o) / max(image_size))*x) for x in (h_size, w_size)])
            octave_list.append(new_size)
    elif mode == 'manual_min':
        for o in octave_scale:
            new_size = tuple([int((float(o) / min(image_size))*x) for x in (h_size, w_size)])
            octave_list.append(new_size)
    elif mode == 'manual':
        for o_size in octave_scale:
            assert len(o_size) % 2 == 0, "Manual octave sizes must be in pairs like: Height,Width,Height,Width..."
        assert len(octave_scale) == num_octaves - 1, \
            "Exected " + str(num_octaves - 1) + " octave size pairs, but got " + str(len(octave_scale)) + " pairs containing: " \
            + str(octave_scale)
        for size_pair in octave_scale:
            octave_list.append((size_pair[0], size_pair[1]))
    if mode == 'manual' or mode == 'manual_max' or mode == 'manual_min':
        octave_list.append(image_size)
    return octave_list


def normalize_weights(dream_losses):
    for n, i in enumerate(dream_losses):
        i.strength = i.strength / max(i.target_size)


def print_layers(layerList, model_name, has_inception):
    print()
    print("\nUsable Layers For '" + model_name + "':")
    if not has_inception:
        for l_names in layerList:
            if l_names == 'P':
                n = '  Pooling Layers:'
            if l_names == 'C':
                n = '  Conv Layers:'
            if l_names == 'R':
                n = '  ReLU Layers:'
            elif l_names == 'BC':
                n = '  BasicConv2d Layers:'
            elif l_names == 'L':
                n = '  Linear/FC layers:'
            if l_names == 'D':
                n = '  Dropout Layers:'
            elif l_names == 'IC':
                n = '  Inception Layers:'
            print(n, ', '.join(layerList[l_names]))
    elif has_inception:
        for l in layerList:
            print(l)
    quit()


def load_label_file(filename):
    with open(filename, 'r') as f:
        x = [l.rstrip('\n') for l in f.readlines()]
    return x


def channel_ids(l, channels):
    channels = channels.split(',')
    c_vals = ''
    for c in channels:
        if c.isdigit():
            c_vals += ',' + str(c)
        elif c.isalpha():
            v = ','.join(str(ch) for ch, n in enumerate(l) if c in n)
            v = ',' + v + ',' if len(v.split(',')) == 1 else v
            c_vals += v
    c_vals = '-1' if c_vals == '' else c_vals
    c_vals = c_vals.replace(',', '', 1) if c_vals[0] == ',' else c_vals
    return c_vals


def new_img(input_image, scale_factor=-1, mode='bilinear'):
    img = input_image.clone()
    if scale_factor != -1:
        img = dream_image.resize_tensor(img, scale_factor, mode)
    return nn.Parameter(img)


def main():
    dreamer = DeepDreamer() 
    dreamer.dream(dreamer.params.content_image, dreamer.params.output_image)

if __name__ == "__main__":
    main()
