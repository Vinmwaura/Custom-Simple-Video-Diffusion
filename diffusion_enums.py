from enum import Enum


# Diffusion Algorithms
class DiffusionAlg(Enum):
    DDPM = 0
    DDIM = 1
    COLD = 2


# Noise Schedulers
class NoiseScheduler(Enum):
    LINEAR = 0
    COSINE = 1

