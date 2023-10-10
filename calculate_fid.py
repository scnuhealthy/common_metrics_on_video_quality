from scipy import linalg
from fvd.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
import numpy as np
import torch
from tqdm import tqdm

import os
from PIL import Image
from torchvision import transforms

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(videos1, videos2, device):
    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = {}

    # get FVD features
    feats1 = get_fvd_feats(videos1, i3d=i3d, device=device)
    feats2 = get_fvd_feats(videos2, i3d=i3d, device=device)

    # feats1 = feats1.cpu().numpy()
    # feats2 = feats2.cpu().numpy()

    m1 = np.mean(feats1, axis=0)
    s1 = np.cov(feats1, rowvar=False)

    m2 = np.mean(feats2, axis=0)
    s2 = np.cov(feats2, rowvar=False)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def load_video_data(root):

    transform = transforms.Compose([
        transforms.Resize((384,512)),
        transforms.ToTensor()
    ])

    # load data
    files = os.listdir(root)
    imgs = []
    for file in files:
        file_path = os.path.join(root, file)
        img = Image.open(file_path)
        img = transform(img)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs

def load_video_files(root):
    video_roots = os.listdir(root)
    videos_data = []
    for video_root in video_roots:
        video_data = load_video_data(os.path.join(root, video_root))
        videos_data.append(video_data)
    videos_data = torch.stack(videos_data)
    return videos_data

def main():

    videos1 = load_video_files('data/predict')
    videos2 = load_video_files('data/gt')
    # videos2 = torch.ones(4, 16, 3, 384, 512, requires_grad=False)
    device = torch.device("cuda")
    print(videos1.shape, videos2.shape)

    # NUMBER_OF_VIDEOS = 8
    # VIDEO_LENGTH = 16
    # CHANNEL = 3
    # SIZE = 64
    # videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    # videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    # device = torch.device("cuda")
    # # device = torch.device("cpu")

    import json
    result = calculate_fid(videos1, videos2, device)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()