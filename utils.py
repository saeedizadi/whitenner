import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.ndimage as ndimage
import os 
from PIL import Image
import glob
import io
from skimage.segmentation import felzenszwalb, slic
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.measure import compare_ssim as ssim
import torchvision
import logging


def nonlocal_denoise_batch(batch):
    if batch.is_cuda:
        batch = batch.to('cpu')
    for b in range(batch.size(0)):
        batch[b,:,:,:] = non_local_denoise(batch[b,:,:,:])
    return batch

def non_local_denoise(image):
    if isinstance(image, torch.Tensor):
        image = to_np_image(image)
        
    sigma_est = np.mean(estimate_sigma(image, multichannel=False))    
    patch_kw = dict(patch_size=5,  # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=False)
    
    image = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
    
    image = (image).astype(np.uint8)
    image = Image.fromarray(image, mode='L')
    tensor = torchvision.transforms.functional.to_tensor(image)     
    return tensor

    
def piecewise_constant_batch(batch, scale, sigma, min_size):      
    # check if tensor are on cuda
    if batch.is_cuda:
        batch = batch.to('cpu')
        
    for b in range(batch.size(0)):
        batch[b,:,:,:] = piecewise_constant(batch[b,:,:,:], scale, sigma, min_size)
    
    return batch

def piecewise_constant(image, scale, sigma, min_size):      
    if isinstance(image, torch.Tensor):
        image = torchvision.transforms.functional.to_pil_image(image)
        image = np.array(image)  / 255.

    # segments = slic(image.astype(np.float), n_segments=5000,
    #     compactness=0.1, max_iter=40, sigma=1.2, spacing=None, 
    #     multichannel=False, enforce_connectivity=True, 
    #     min_size_factor=0.1, max_size_factor=20)
        
    segments = felzenszwalb(image.astype(np.float), scale=scale, sigma=sigma, min_size=min_size, multichannel=True)
    # segments = felzenszwalb(image.astype(np.float), scale=10, sigma=1.2, min_size=40, multichannel=True)
    for l in range(segments.max()+1):
        indices = np.where(segments == l)
        xs, ys = indices[0], indices[1]

        part = image[xs, ys]
        image[xs, ys] = np.mean(part)

    image = (image * 255.0).astype(np.uint8)
    image = Image.fromarray(image, mode='L')
    tensor = torchvision.transforms.functional.to_tensor(image)

    return tensor
    
def add_noise(img, std):      
    noise = torch.randn(img.size()) * std
    return img + noise.to('cuda')

def get_model_path(run_dir, epoch):
    """Get the path to the saved model state_dict with the given epoch number.
    If epoch is 'latest', the latest model state dict path will be returned.
    Parameters
    ----------
    run_dir : str
        Path to run directory.
    epoch : type
        Epoch number of model to get.
    Returns
    -------
    str
        Path to model state_dict file.
    """
    if epoch == 'latest':
        return os.path.join(run_dir, 'latest.statedict.pkl')
    if epoch == 'best':
        return os.path.join(run_dir, 'best.statedict.pkl')

    filenames = os.listdir(run_dir)

    for filename in filenames:
        if 'statedict' not in filename:
            continue
        if filename.startswith('epoch'):
            number = int(filename[len('epoch'):].split('_')[0])
            if epoch == number:
                return os.path.join(run_dir, filename)

    raise ValueError("No statedict found with epoch number '{}'".format(epoch))

def getTimeName():
    """Return the current time in format <day>-<month>_<hour><minute>
    for use in filenames."""
    from datetime import datetime
    t = datetime.now()
    return "{:02d}-{:02d}_{:02d}{:02d}".format(t.day,t.month,t.hour,t.minute)


def psnr_score(output, target):
    """Return psnr. Assumes pixel values in [0,1]"""    
    output = output.cpu()
    target = target.cpu()
    with torch.no_grad():
        total = output.size()[0]        
        mse = nn.functional.mse_loss(output, target)        
        return 10 * np.log10(1. / mse.item())
    
def ssim_score(output, target):
    output = output.squeeze().cpu().data.numpy()
    target = target.squeeze().cpu().data.numpy()
    return ssim(output, target, data_range=output.max()-output.min(), multichannel=False)

def sem_score(psnrs):
    std = np.std(psnrs)
    return std / np.sqrt(len(psnrs))    


def save_model(model, epoch, directory, metrics, filename=None, _log=logging.getLogger()):
    """Save the state dict of the model in the directory,
    with the save name metrics at the given epoch.

    epoch: epoch number(<= 4 digits)
    directory: where to save statedict
    metrics: dictionary of metrics to append to save filename
    filename: if a name is given, it overrides the above created filename

    Returns the save file name
    """
    # save state dict
    postfix = ""
    if filename is None:
        filename = "epoch{0:04d}_{1}_".format(epoch,getTimeName())
        postfix = "_".join(["{0}{1:0.4f}".format(name, val) for name, val in metrics.items()])

    filename = os.path.join(directory, filename + postfix + ".statedict.pkl")

    if isinstance(model, nn.DataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    torch.save(state, filename)
    _log.info("Saved model at {}".format(filename))

    return filename

def psnr(prediction, target):
    mse = nn.functional.mse_loss(prediction, target)
    return 10 * torch.log10(1 / mse.item())


def expand(x, r):
    return np.repeat(np.repeat(x, r, axis = 0), r, axis = 1)





def random_noise(img, params):
    """Parameters for random noise include the mode and the type.

    mode: gaussian, poisson, or gaussian_poisson noise type
    std: std of gaussian
    photons_at_max: at image with intensity 1 has this many photons on average
    clamp: clamp result to [0,1]
    """

    noisy = img

    if params['mode'] == 'poisson' or params['mode'] == 'gaussian_poisson':
        noisy = torch.poisson(noisy * params['photons_at_max']) / params['photons_at_max']

    if params['mode'] == 'gaussian' or params['mode'] == 'gaussian_poisson':
        noise = torch.randn(img.size()).to(img.device) * params['std']
        noisy = noise + noisy

    if params['mode'] == 'bernoulli':
        noisy = noisy * torch.bernoulli(torch.ones(noisy.shape) * params['p'])

    if 'clamp' in params and params['clamp']:
        noisy = torch.clamp(noisy, 0, 1)

    return noisy


def test_bernoulli_noise():
    torch.manual_seed(2018)
    p = 0.2
    shape = (10, 1, 100, 100)
    n = 10 * 100 * 100
    img = torch.ones(shape)
    noisy = random_noise(img, {'mode': 'bernoulli', 'p': p})

    var = n * p * (1 - p)

    assert torch.abs(noisy.sum() - p * img.sum()) < 3 * (var ** 0.5)


def psnr(x, x_true, max_intensity=1.0, pad=None, rescale=False):
    '''A function computing the PSNR of a noisy tensor x approximating a tensor x_true.

    It vectorizes over the batch.

    PSNR := 10*log10 (MAX^2/MSE)

    where the MSE is the averaged squared error over all pixels and channels.
    '''

    return 10 * torch.log10((max_intensity ** 2) / mse(x, x_true, pad=pad, rescale=rescale))


def test_psnr():
    std = 0.1
    noise = torch.randn(10, 3, 100, 100) * std
    x_true = torch.ones(10, 3, 100, 100) / 2
    x = x_true + noise
    # MSE should be 0.01. PSNR should be 20.
    assert (torch.abs(psnr(x, x_true) - 20) < 0.1).all()

    x = 256 * x
    x_true = 256 * x_true
    assert (torch.abs(psnr(x, x_true, 256) - 20) < 0.2).all()


def test_mse_rescale():
    y = torch.randn(10, 3, 10, 10)
    x = 10 * y + 7
    assert (mse(x, y, rescale=True) < 1e-5).all()

    # Normalized values are (1, 1, 0, -2) and (1, 1, -1, -1)
    y = torch.Tensor([3, 3, 2, 0]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x = torch.Tensor([5, 5, 0, 0]).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    assert mse(x, y, rescale=True).sum() == 0.5


def mse(x, y, pad=None, rescale=False):
    if pad:
        x = x[:, :, pad:-pad, pad:-pad]
        y = y[:, :, pad:-pad, pad:-pad]

    def batchwise_mean(z):
        return z.reshape(z.shape[0], -1).mean(dim=1).reshape(-1, 1, 1, 1)

    if rescale:
        x = x - batchwise_mean(x)
        y = y - batchwise_mean(y)
        a = batchwise_mean(x * y) / batchwise_mean(x * x)
        x = a * x

    return batchwise_mean((x - y) ** 2).reshape(-1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)


# def ssim(img1, img2, window_size=11, size_average=True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)

#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)

#     return _ssim(img1, img2, window, window_size, channel, size_average)


def smooth(tensor):
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 2.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(tensor.device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)
    return filtered_tensor


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


class PercentileNormalizer():
    """Percentile-based image normalization.
    Parameters
    ----------
    pmin : float
        Low percentile.
    pmax : float
        High percentile.
    dtype : type
        Data type after normalization.
    kwargs : dict
        Keyword arguments for :func:`csbdeep.utils.normalize_mi_ma`.
    """

    def __init__(self, pmin=2, pmax=99.8, dtype=np.float32, **kwargs):
        if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100):
            raise ValueError
        self.pmin = pmin
        self.pmax = pmax
        self.dtype = dtype
        self.kwargs = kwargs

        self.mi = None
        self.ma = None

    def normalize(self, img, channel=1):
        """Percentile-based normalization of raw input image.
        Note that percentiles are computed individually for each channel (if present in `axes`).
        """
        axes = tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img, self.pmin, axis=axes, keepdims=True).astype(self.dtype, copy=False)
        self.ma = np.percentile(img, self.pmax, axis=axes, keepdims=True).astype(self.dtype, copy=False)
        return normalize_mi_ma(img, self.mi, self.ma, dtype=self.dtype, **self.kwargs)

    def denormalize(self, mean):
        """Undo percentile-based normalization to map restored image to similar range as input image.
        """
        alpha = self.ma - self.mi
        beta = self.mi
        return alpha * mean + beta


def test_percentile_normalizer():
    a = np.arange(1000).reshape(10, 1, 10, 10).astype(np.uint16)

    norm = PercentileNormalizer(pmin=0, pmax=100, dtype=np.float32, clip=False)
    assert norm.normalize(a).min() == 0 and norm.normalize(a).max() == 1

    # gap between 10th and 90th percentile is 100 to 900, so
    # the transform is (x - 100)/800
    norm = PercentileNormalizer(pmin=10, pmax=90, dtype=np.float32, clip=False)
    assert norm.normalize(a).max() == 1.125

    norm = PercentileNormalizer(pmin=2, pmax=99.8, dtype=np.float32, clip=True)
    assert norm.normalize(a).max() == 1.0


def gpuinfo(gpuid):
    import subprocess
    sp = subprocess.Popen(['nvidia-smi', '-q', '-i', str(gpuid), '-d', 'MEMORY'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split('BAR1', 1)[0].split('\n')
    out_dict = {}
    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except:
            pass
    return out_dict


def getfreegpumem(id):
    return int(gpuinfo(id)['Free'].replace('MiB', '').strip())


def getbestgpu():
    freememlist = []
    for id in range(4):
        freemem = getfreegpumem(id)
        print("GPU device %d has %d MiB left." % (id, freemem))
        freememlist.append(freemem)
    idbest = freememlist.index(max(freememlist))
    print("--> GPU device %d was chosen" % idbest)
    return idbest


def get_args():
    global args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_files",
                        help="configuration file for experiment.",
                        type=str,
                        nargs='+')
    parser.add_argument("--device",
                        help="cuda device",
                        type=str,
                        required=True)
    args = parser.parse_args()
def gradient(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().squeeze().cpu().numpy()
        
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ky = np.array([[-1,-2,-1] ,[0,0,0], [1,2,1]])    
    x=ndimage.convolve(img,kx) / 8.
    y=ndimage.convolve(img,ky) / 8.
    return x, y

def to_numpy(im):
    return im.squeeze().detach().cpu().numpy()

def build_crop_grid(source_size,target_size):
    k = float(target_size)/float(source_size)
    direct = torch.linspace(-k,k,target_size).unsqueeze(0).repeat(target_size,1).unsqueeze(-1)    
    full = torch.cat([direct,direct.transpose(1,0)],dim=2).unsqueeze(0)    
    return full

def build_full_grid(source_size):    
    direct = torch.linspace(-1,1,source_size).unsqueeze(0).repeat(source_size,1).unsqueeze(-1)    
    full = torch.cat([direct,direct.transpose(1,0)],dim=2).unsqueeze(0)
    return full

def shift_full_grid(x,grid, step):    
    grid = grid.repeat(x.size(0),1,1,1)        
    a = torch.FloatTensor(x.size(0)).random_(-step, step+1).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)            
    grid[:,:,:,0] += torch.FloatTensor(x.size(0)).random_(-step, step+1).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)        
    grid[:,:,:,1] += torch.FloatTensor(x.size(0)).random_(-step, step+1).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(3)            
    return grid

    
def save_tensor(input, filename, postfix, savedir, ext='png'):    
    if postfix != '':
        postfix = '_' + postfix
    
    output = input
    if not isinstance(output, Image.Image):
        convert_fn= to_np_array if ext == 'npy' else to_pil_image        
        output = convert_fn(input)
            
    filename = filename.split('.')[0] 
    filename = "{0}{1}.{2}".format(filename, postfix, ext)        
    filepath = os.path.join(savedir, filename)    
    if ext == 'png':
        output.save(filepath)
    else:
        np.save(filepath, output)




def get_axes_plot(images, titles):
    fig, axes = plt.subplots(1, len(titles), squeeze=True)
    for i in range(len(images)):
        im = to_np_image(images[i])
        im_temp = axes.ravel()[i].imshow(im, cmap=plt.gray())

        axes.ravel()[i].set_title(titles[i], fontdict={'fontsize': 5})
        axes.ravel()[i].set_axis_off()

    fig = plt.gcf()
    imgdata = io.BytesIO()
    fig.savefig(imgdata, format='png', dpi=400, bbox_inches='tight', pad_inches=0)
    imgdata.seek(0)
    im = Image.open(imgdata).convert('RGB')
    return im

def to_np_array(tensor):
    array = tensor.detach().cpu().numpy().squeeze()    
    return array

def to_np_image(tensor):
    array = to_np_array(tensor)
    array = 255.0 * array   
    array = np.clip(array, 0, 255)
    array = np.uint8(array)
    return array    
    
def to_pil_image(tensor):
    array = to_np_array(tensor)
    array = 255.0 * array   
    array = np.clip(array, 0, 255)
    array = np.uint8(array)
    im = Image.fromarray(array, mode='L')
    return im


    
        

