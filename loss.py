import torch
import torch.nn as nn
import torch.nn.functional as f
from utils import *

def np_gaussian_2d(width=5, sigma=-1):
    '''Truncated 2D Gaussian filter'''
    assert width % 2 == 1
    if sigma <= 0:
        sigma = float(width) / 4

    r = np.arange(-(width // 2), (width // 2) + 1, dtype=np.float32)
    gaussian_1d = np.exp(-0.5 * r * r / (sigma * sigma))
    gaussian_2d = gaussian_1d.reshape(-1, 1) * gaussian_1d
    gaussian_2d /= gaussian_2d.sum()

    return gaussian_2d

def get_d_l(x):
    a = np_gaussian_2d(7, sigma=3)        
    a = torch.tensor(a, requires_grad=False, dtype=torch.float)                       
    a = a.unsqueeze(0).unsqueeze(0)   

    G_x, G_y = gradient_operator(x)  
    G_x, G_y = torch.abs(G_x), torch.abs(G_y)
    Dx, Dy = F.conv2d(G_x, a, padding=3), F.conv2d(G_y, a, padding=3)    


    Lx = F.conv2d(G_x, a, padding=3)
    Ly = F.conv2d(G_y, a, padding=3)
    Lx, Ly = torch.sqrt(Lx), torch.abs(Ly)

    return Dx, Dy, Lx, Ly

def cross_entropy(predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions. 
        Input: predictions (N, k) ndarray
            targets (N, k) ndarray        
        Returns: scalar
        """
        predictions = torch.clamp(predictions, epsilon, 1. - epsilon)
        N = predictions.size(0)
        ce = -torch.sum(targets*torch.log(predictions+1e-9))/N
        return ce

def gradient_operator(x):
        a = torch.tensor([[-1,0,-1],[-1,0,1], [-1, 0,1]], requires_grad=False, dtype=torch.float)               
        a = a.view(1,1,3,3)
        
        b = torch.tensor([[-1,-1,-1],[0,0,0], [1, 1,1]], requires_grad=False, dtype=torch.float)
        b = b.view(1,1,3,3)

        G_x = F.conv2d(x, a, padding=1)
        G_y = F.conv2d(x, b, padding=1)            
        return G_x, G_y
    
def make_filter(size=3):
    f = -1 * torch.ones((size,size), requires_grad=False, dtype=torch.float)
    f[size//2, size//2] = size**2 - 1
    return f    
    
    

class ARLoss(nn.Module):
    def __init__(self, maxshift):
        super(ARLoss, self).__init__()
        self.maxshift = maxshift
        self.padding = (maxshift, maxshift, maxshift, maxshift)

    def _batch_shift(self, x):

        # Pads the x with refelction method.
        x = torch.nn.functional.pad(x, self.padding, 'reflect')
        x_org = x.clone()

        for b in range(x.size(0)):
            x[b, :, :, :] = self._shift(x[b])

        # Center Crop with grid_sample
        crop_grid = build_crop_grid(x.size(2), x.size(
            2)-2*self.maxshift).repeat(x.size(0), 1, 1, 1)
        x_org = F.grid_sample(x_org, crop_grid)
        x = F.grid_sample(x, crop_grid)

        return x_org, x

    def _shift(self, img):

        while True:
            hidx = np.random.randint(-self.maxshift, self.maxshift + 1)
            vidx = np.random.randint(-self.maxshift, self.maxshift + 1)
            if hidx != 0 or vidx != 0:
                break

        h_shift = img.clone()
        if hidx > 0:
            h_shift[:, :, hidx:] = img[:, :, :-hidx]
        elif hidx < 0:
            h_shift[:, :, :hidx] = img[:, :, -hidx:]
        v_shift = h_shift.clone()
        if vidx > 0:
            v_shift[:, vidx:, :] = h_shift[:, :-vidx, :]
        elif vidx < 0:
            v_shift[:, :vidx, :] = h_shift[:, -vidx:, :]

        return v_shift

    def forward(self, x):
        x, x_shifted = self._batch_shift(x)

        # Reshape
        a = x.contiguous().view(x.size(0), -1)
        b = x_shifted.contiguous().view(x.size(0), -1)

        # Subtract Mean
        m = torch.mean(x.contiguous().view(x.size(0), -1), dim=1)
        a -= m.unsqueeze(1).repeat(1, a.size(1))

        m = torch.mean(x_shifted.contiguous().view(
            x_shifted.size(0), -1), dim=1)
        b -= m.unsqueeze(1).repeat(1, b.size(1))

        acs = torch.diag(torch.matmul(a, torch.transpose(b, 0, 1)))
        acs = acs / (b.size(1) * torch.std(x.contiguous().view(x.size(0), -1), dim=1)
                     * torch.std(x_shifted.contiguous().view(x.size(0), -1), dim=1))

        return torch.abs(torch.mean(acs))


class StationaryLoss(nn.Module):
    def __init__(self, stat='std'):
        super(StationaryLoss, self).__init__()
        self.stat = stat

    def forward(self, x):

        # imgw = x.size(2)
        # bsize = np.random.randint(2, int(imgw/2))
        bsize = 4

        if self.stat == 'std':
            means = self.std_pool2d(
                x, bsize, bsize).view(x.size(0), -1)
        elif self.stat == 'mean':
            means = F.avg_pool2d(x, bsize, bsize).view(x.size(0), -1)

        n_elements = means.size(1)

        means_prob = torch.softmax(means, dim=1)
        gt_prob = (1./n_elements) * torch.ones_like(means)

        # out = F.kl_div(means_prob, gt_prob, reduction='mean')
        out = cross_entropy(means_prob, gt_prob)
        return out

    def std_pool2d(self, x, kernel_size, stride):
        means = F.avg_pool2d(x, kernel_size, stride)
        means = F.interpolate(means, size=x.size()[2:], mode='nearest')
        std = F.avg_pool2d((x - means) ** 2, kernel_size, stride)
        return std

    
class TVLoss(nn.Module):
        def __init__(self):
            super(TVLoss, self).__init__()
            
            self.a = torch.tensor([[-1,0,-1],[-2,0,2], [-1, 0,1]], requires_grad=False, dtype=torch.float)               
            self.a = self.a.view(1,1,3,3)
            
            self.b = torch.tensor([[-1,-2,-1],[0,0,0], [1, 2,1]], requires_grad=False, dtype=torch.float)
            self.b = self.b.view(1,1,3,3)
            
        def forward(self, signal):            
            G_x = F.conv2d(signal, self.a, padding=1)
            G_y = F.conv2d(signal, self.b, padding=1)
            TV = torch.sqrt(G_x**2 + G_y**2)
                       
            loss = torch.mean(TV)
            return loss

# class LORLoss(nn.Module):
#         def __init__(self, sigma = 1.1, size=5):
#             super(LORLoss, self).__init__()
#             self.sigma = sigma       
#             self.b = make_filter(size)            
#             # self.b = torch.ten.size()sor([[-1,-1,-1],[-1,8,-1], [-1, -1,-1]], requires_grad=False, dtype=torch.float)
#             self.b = self.b.view(1,1, size, size)

#         def forward(self, x):                        
#             G = F.conv2d(x, self.b, padding=1)
#             lx = torch.log10(1.0 + 0.5 * (G / self.sigma)**2)            
#             return torch.mean(lx)            

class LORLoss(nn.Module):
        def __init__(self, sigma = 1.1, size=5):
            super(LORLoss, self).__init__()
            self.sigma = sigma                   

        def forward(self, x):        
            Dx, Dy, Lx, Ly = get_d_l(x)                
            G = (Dx / Lx + 0.0001) + (Dy / Ly + 0.0001)            
            return torch.mean(G)       

class GRLoss(nn.Module):
    def __init__(self):
        super(GRLoss, self).__init__()
        pass

    def forward(self, output, target):
        o_x, o_y = gradient_operator(output)        
        t_x, t_y = gradient_operator(target)        
        return torch.nn.functional.mse_loss(o_x, t_x) + \
            torch.nn.functional.mse_loss(o_y, t_y)
