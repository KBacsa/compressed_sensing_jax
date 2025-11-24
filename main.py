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

    # TODO add argument parsing
    sub = 2 
    mask_percent = 0.25
    wavelet = 'db16'
    level = 3
    mode = 'wrap'
    l1_reg = 1e-1
    debug = True

    # Load and downsample image
    image = plt.imread('escher.jpeg')
    if len(image.shape) < 3:
        image = image[..., np.newaxis]
    image = spimg.zoom(image, (1/sub, 1/sub, 1))
    nx, ny, c = image.shape

    # Randomly mask image
    k = round(nx * ny * mask_percent)
    mask_index = np.sort(np.random.choice(nx * ny, k, replace=False))
    mask_array = np.zeros_like(image)
    mask_array.flat[mask_index] = 1 
    masked_image = image * mask_array 
    # Make wavelet filters
    filt = jw.get_filter_bank(wavelet)
    kernel_dec, kernel_rec = jw.make_kernels(filt, 1)

    def wavelet_decompose(x):
        return jw.wavelet_dec(x.reshape(1, x.shape[0], x.shape[1], 1), kernel_dec, levels=level, mode=mode)

    def wavelet_reconstruct(x_wav):
        return jw.wavelet_rec(x_wav, kernel_rec, levels=level, mode=mode).squeeze()

    def dct2(x):
        return jax.scipy.fft.dct(jax.scipy.fft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

    def idct2(x):
        return jax.scipy.fft.idct(jax.scipy.fft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
    
    # Define compressed sensing forward operators
    def Phi(x, mask):
        return x.ravel()[mask]

    def Psi(theta):
        x = wavelet_reconstruct(theta)
        return x

    def Psi_dct(theta):
        x = idct2(theta)
        return x

    def forward(theta, mask):
        return Phi(Psi(theta), mask)

    def least_squares(x, data):
        mask, y = data
        residuals = forward(x, mask) - y
        return jnp.sum(residuals ** 2)

    # Compressed sensing per channels
    mask_array = jnp.array(mask_array)
    image_channels = []

    if c < 3:
        colormap = 'gray'
    else:
        colormap = None

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap=colormap)
    axs[0].set_title('Original image')
    axs[1].imshow(masked_image, cmap=colormap)
    axs[1].set_title('Masked image')

    for channel in range(c):
        image_channel = jnp.array(image[..., channel].astype(float))

        # Solving || A x - b||_2 + lambda ||x||_1
        b = Phi(image_channel, mask_index)

        theta_true = wavelet_decompose(image_channel)
        print(theta_true.shape)
        #theta_true = dct2(image_channel)
        theta_init = jnp.ones_like(theta_true)

        pg = ProximalGradient(fun=least_squares, prox=prox_lasso, verbose=debug, stepsize=0, maxls=50)
        pg_sol = pg.run(theta_init, hyperparams_prox=l1_reg, data=(mask_index, b)).params
        reconstruction = np.array(wavelet_reconstruct(pg_sol))
        image_channels.append(reconstruction)

    reconstructed_image = np.stack(image_channels, axis=-1)

    reconstructed_psnr = psnr(image, reconstructed_image)

    axs[2].imshow(reconstructed_image, cmap=colormap)
    axs[2].set_title('Reconstructed image (PSNR: {:.2f} dB)'.format(reconstructed_psnr))
    #plt.show()
    fig.savefig('reconstruction_{}.png'.format(wavelet))

