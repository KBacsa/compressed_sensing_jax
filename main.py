import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import ProximalGradient
from jaxopt.prox import prox_lasso
import matplotlib.pyplot as plt
import scipy.ndimage as spimg
import importlib
jw = importlib.import_module('jax-wavelets.jax_wavelets.jax_wavelets')




if __name__ == '__main__':

    np.random.seed(0)

    # TODO add argument parsing
    sub = 2 
    mask_percent = 0.75
    wavelet = 'db6'
    level = 3
    mode = 'wrap'
    l1_reg = 1e-5
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
    for channel in range(c):
        image_channel = jnp.array(image[..., channel].astype(float))

        # Solving || A x - b||_2 + lambda ||x||_1
        b = Phi(image_channel, mask_array)

        theta_true = wavelet_decompose(image_channel)
        #theta_true = dct2(image_channel)
        theta_init = jnp.ones_like(theta_true)

        pg = ProximalGradient(fun=least_squares, prox=prox_lasso, verbose=debug)
        pg_sol = pg.run(theta_init, hyperparams_prox=l1_reg, data=(mask_array, b)).params
        reconstruction = wavelet_reconstruct(pg_sol)
        #reconstruction = idct2(pg_sol)

        image_channels.append(np.asarray(reconstruction))

    reconstructed_image = np.stack(image_channels, axis=-1)

    plt.imshow(reconstructed_image)
    plt.show()

