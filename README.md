# 2D Compressed Sensing in JAX

## Description

A simple implementation of 2D compressed sensing using JAX.
This implementation partially reproduces the tutorial from [Robert Taylor](https://humaticlabs.com/blog/compressed-sensing-python/), but relies entirely on ```jax``` and ```jaxopt``` rather than external libraries for computation of the gradients and the optimization.
The implementation of the wavelets is taken from [jax-wavelets](https://github.com/crowsonkb/jax-wavelets).

## Explanation

Given a sensing operator $$\Psi$$


## Requirements

The following packages are required:

```
jax==0.8.0
jaxopt==0.8.5
PyWavelets==1.9.0
```

## Examples

Example using Daubechies wavelet (6) with 25% of samples. 

![alt text](example_wavelet.png)

Example using Discrete Cosine Transform with 25% of samples. 

![alt text](example_dct.png)
