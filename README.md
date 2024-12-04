
# FFTTests
Benchmarking CUDA FFT implementation against optimized CPU based FFT implementations. The following libraries have been used:
1. [fftw3](https://www.fftw.org/) - Open source, runs well on most architectures
2. [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.ie2hyd) - Closed source, requires license for commercial usage, highly optimized **only for x86 architecture**
3. [cuFFT](https://developer.nvidia.com/cufft#ixmgsk) - Optimized and runs only on NVIDIA GPUs, can be used in commercial products

|Input Size | FFTW  |Intel MKL FFTW  | CUDA FFT |
|--|--|--|--|
| 1M | 8ms |3ms  |1ms  |
| 5M | 40ms |13ms  |4ms  |
| 10M | 166ms |34ms  |9ms  |
| 50M | 687ms |133ms  |39ms  |
| 100M | 1121ms |276ms  |75ms  |
| 500M | 14775ms |1660ms  |353ms  |
| 1B | 31386ms |11756ms  |707ms  |

Relative speed-up of GPU acceleration against CPU-based implementations

|Input Size | Speedup v FFTW  |Speedup v MKL FFTW
|--|--|--|
| 1M | 8x |3x
| 5M | 10x |3.25x|
| 10M | 18.4x |3.8x|
| 50M | 17.6x |3.41x|
| 100M | 14.94x |3.68x|
| 500M | 41.9x |4.7x|
| 1B | 44.3x |16.7x|
