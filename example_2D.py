from wavelet_noise import WaveletNoise2D
import matplotlib.pyplot as plt

# %% Single scale wavelet noise at increasing frequency

n = 256     # Size 
nBands = 1  # Number of octaves

# Generate noise with increasing frequency
noise1 = WaveletNoise2D(0.05, nBands, n)
noise2 = WaveletNoise2D(0.1, nBands, n)
noise3 = WaveletNoise2D(0.2, nBands, n)

# Normalise 
noise1 = (noise1 - noise1.min()) / (noise1.max() - noise1.min())
noise2 = (noise2 - noise2.min()) / (noise2.max() - noise2.min())
noise3 = (noise3 - noise3.min()) / (noise3.max() - noise3.min())

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(noise1, cmap='gray')
axes[1].imshow(noise2, cmap='gray')
axes[2].imshow(noise3, cmap='gray')
axes[0].axis('off')
axes[1].axis('off')
axes[2].axis('off')
plt.tight_layout()
plt.show()

# %% Multiscale noise

nBands = 4  
initial_freq = 0.05 # Starting frequency before scaling by lacunarity

multiscale_noise = WaveletNoise2D(initial_freq, nBands, n)

plt.figure()
plt.imshow(multiscale_noise, cmap = 'gray')
plt.axis('off')
plt.tight_layout()
plt.show()
