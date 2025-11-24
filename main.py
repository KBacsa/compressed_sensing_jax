import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import ProximalGradient
from jaxopt.prox import prox_lasso
import matplotlib.pyplot as plt
import scipy.ndimage as spimg
import importlib
jw = importlib.import_module('jax-wavelets.jax_wavelets.jax_wavelets')


def psnr(img1, img2):
    mse = np.mean(np.square(np.subtract(img1.astype(int), img2.astype(int))))
    if mse == 0:
        return np.Inf
    PIXEL_MAX = 255.
    return 20 * np.log10(PIXEL_MAX) - 10 * np.log10(mse)


if __name__ == '__main__':

    np.random.seed(0)

    parser = argparse.ArgumentParser(prog='Compressed Senssing')
    parser.add_argument('--filename', type=str, default='escher.jpeg', help='Input image')
    parser.add_argument('--sub', type=int, default=2, help='Downsample factor')
    parser.add_argument('--mask-percent', type=float, default=0.25, help='Portion to mask')
    parser.add_argument('--method', type=str, default='wavelet', help='Use wavelet or dct')
    parser.add_argument('--wavelet-name', type=str, default='db3', help='Wavelet')
    parser.add_argument('--mode', type=str, default='wrap', help='Wavelet signal extension method')
    parser.add_argument('--level', type=int, default=3, help='Wavelet level')
    parser.add_argument('--step-size', type=float, default=0, help='Step size of descent (leave 0 for Line Search)')
    parser.add_argument('--max-ls', type=int, default=50, help='Max steps of Line Search')
    parser.add_argument('--l1-reg', type=float, default=1e-1, help='Weight of L1 regularization')
    parser.add_argument('--debug', type=bool, default=True, help='Show descent')

    args = parser.parse_args()

    # Load and downsample image
    image = plt.imread(args.filename)
    if len(image.shape) < 3:
        image = image[..., np.newaxis]
    image = spimg.zoom(image, (1/args.sub, 1/args.sub, 1))
    nx, ny, c = image.shape

    # Randomly mask image
    k = round(nx * ny * args.mask_percent)
    mask_index = np.sort(np.random.choice(nx * ny, k, replace=False))
    mask_array = np.zeros_like(image)
    mask_array.flat[mask_index] = 1 
    masked_image = image * mask_array 

    # Implement Compressed Sensing operators on 2D arrays
    if args.method == 'wavelet':
        filt = jw.get_filter_bank(args.wavelet_name)
        kernel_dec, kernel_rec = jw.make_kernels(filt, 1)

        def wavelet_decompose(x):
            return jw.wavelet_dec(x.reshape(1, x.shape[0], x.shape[1], 1), kernel_dec, levels=args.level, mode=args.mode)

        def wavelet_reconstruct(x_wav):
            return jw.wavelet_rec(x_wav, kernel_rec, levels=args.level, mode=args.mode).squeeze()

        def Psi(theta):
            x = wavelet_reconstruct(theta)
            return x

        def adj_Psi(x):
            theta = wavelet_decompose(x)
            return theta

    elif args.method == 'dct':

        def dct2(x):
            return jax.scipy.fft.dct(jax.scipy.fft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

        def idct2(x):
            return jax.scipy.fft.idct(jax.scipy.fft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
        
        def Psi(theta):
            x = idct2(theta)
            return x

        def adj_Psi(x):
            theta = dct2(x)
            return theta

    else:
        raise NotImplementedError('Unknown method.')

    def Phi(x, mask):
        return x.ravel()[mask]

    def forward(theta, mask):
        return Phi(Psi(theta), mask)

    def least_squares(x, data):
        mask, y = data
        residuals = forward(x, mask) - y
        return jnp.sum(residuals ** 2)


    if c < 3:
        colormap = 'gray'
    else:
        colormap = None

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap=colormap)
    axs[0].set_title('Original image')
    axs[1].imshow(masked_image, cmap=colormap)
    axs[1].set_title('Masked image')

    # Compressed sensing per channels
    image_channels = []
    for channel in range(c):
        image_channel = jnp.array(image[..., channel].astype(float))

        # Solving || A x - b||_2 + lambda ||x||_1
        b = Phi(image_channel, mask_index)

        theta_true = adj_Psi(image_channel)
        theta_init = jnp.ones_like(theta_true)

        pg = ProximalGradient(fun=least_squares, prox=prox_lasso, verbose=args.debug, stepsize=args.step_size, maxls=args.max_ls)
        pg_sol = pg.run(theta_init, hyperparams_prox=args.l1_reg, data=(mask_index, b)).params
        reconstruction = np.asarray(Psi(pg_sol))
        image_channels.append(reconstruction)

    reconstructed_image = np.stack(image_channels, axis=-1)
    reconstructed_psnr = psnr(image, reconstructed_image)

    axs[2].imshow(reconstructed_image, cmap=colormap)
    axs[2].set_title('Reconstructed image (PSNR: {:.2f} dB)'.format(reconstructed_psnr))
    plt.show()

