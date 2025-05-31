import numpy as np
from scipy.ndimage import convolve1d

down_coeffs = np.array([
    0.000334, -0.001528,  0.000410,  0.003545, -0.000938, -0.008233,  0.002172,  0.019120,
    -0.005040, -0.044412,  0.011655,  0.103311, -0.025936, -0.243780,  0.033979,  0.655340,
     0.655340,  0.033979, -0.243780, -0.025936,  0.103311,  0.011655, -0.044412, -0.005040,
     0.019120,  0.002172, -0.008233, -0.000938,  0.003546,  0.000410, -0.001528,  0.000334])

up_coeffs = np.array([0.25, 0.75, 0.75, 0.25])

def X_dim2D(input_matrix, n):

  X_axis = 1

  downsampled = convolve1d(input_matrix,
                            down_coeffs,
                            mode = 'wrap',
                            origin = -1,
                            axis = X_axis)[:,::2] 

  upsampled = np.zeros((n,n))
  upsampled[:,::2] = downsampled

  upsampled = convolve1d(upsampled,
                          up_coeffs,
                          mode = 'wrap',
                          origin = 0,
                          axis = X_axis)

  return upsampled


def X_dim3D(input_matrix, n):

  X_axis = 1

  downsampled = convolve1d(input_matrix,
                            down_coeffs,
                            mode = 'wrap',
                            origin = -1,
                            axis = X_axis)[:,::2,:] 

  upsampled = np.zeros((n,n,n))
  upsampled[:,::2,:] = downsampled

  upsampled = convolve1d(upsampled,
                          up_coeffs,
                          mode = 'wrap',
                          origin = 0,
                          axis = X_axis)

  return upsampled

def Y_dim2D(input_matrix, n):

  Y_axis = 0

  downsampled = convolve1d(input_matrix,
                            down_coeffs,
                            mode = 'wrap',
                            origin = -1,
                            axis = Y_axis)[::2,:] 

  upsampled = np.zeros((n,n))
  upsampled[::2,:] = downsampled

  upsampled = convolve1d(upsampled,
                          up_coeffs,
                          mode = 'wrap',
                          origin = 0,
                          axis = Y_axis)

  return upsampled


def Y_dim3D(input_matrix, n):

  Y_axis = 0

  downsampled = convolve1d(input_matrix,
                            down_coeffs,
                            mode = 'wrap',
                            origin = -1,
                            axis = Y_axis)[::2,:,:] 

  upsampled = np.zeros((n,n,n))
  upsampled[::2,:,:] = downsampled

  upsampled = convolve1d(upsampled,
                          up_coeffs,
                          mode = 'wrap',
                          origin = 0,
                          axis = Y_axis)

  return upsampled


def Z_dim3D(input_matrix, n):

  Z_axis = 2
  downsampled = convolve1d(input_matrix,
                            down_coeffs,
                            mode = 'wrap',
                            origin = -1,
                            axis = Z_axis)[:,:,::2]

  upsampled = np.zeros((n,n,n))
  upsampled[:,:,::2] = downsampled

  upsampled = convolve1d(upsampled,
                          up_coeffs,
                          mode = 'wrap',
                          origin = 0,
                          axis = Z_axis)

  return upsampled



def GenerateNoiseTile2D(input_texture, n):
    
    # Downsampling & upsampling per dimension
    temp1 = X_dim2D(input_texture, n)
    temp2 = Y_dim2D(temp1, n)

    # Subtract course-scale contribution
    noise_2 = input_texture - temp2

    # Correct even odd variance
    offset = n/2
    if (offset%2==0):
        offset += 1 # Add 1 to even offset to make it odd

    X_coord, Y_coord = np.meshgrid(range(n),
                                   range(n),
                                   indexing = 'xy')

    X_coord = X_coord + offset
    Y_coord = Y_coord + offset

    X_coord = X_coord.astype(int) & (n-1)
    Y_coord = Y_coord.astype(int) & (n-1)

    temp_variance = noise_2[Y_coord, X_coord]

    result = noise_2 + temp_variance

    return result

def GenerateNoiseTile3D(input_texture, n):
    
    # Downsampling & upsampling  
    temp1 = X_dim3D(input_texture, n)
    temp2 = Y_dim3D(temp1, n)
    temp3 = Z_dim3D(temp2, n)

    # Subtract course-scale contribution
    noise_3_cube = input_texture - temp3

    # Correct even odd variance
    offset = n/2
    if (offset%2==0):
        offset += 1 # Add 1 to even offset to make it odd

    X_coord, Y_coord, Z_coord = np.meshgrid(range(n),
                                            range(n),
                                            range(n),
                                            indexing = 'xy')

    X_coord = X_coord + offset
    Y_coord = Y_coord + offset
    Z_coord = Z_coord + offset

    X_coord = X_coord.astype(int) & (n-1)
    Y_coord = Y_coord.astype(int) & (n-1)
    Z_coord = Z_coord.astype(int) & (n-1)

    temp_variance = noise_3_cube[Y_coord, X_coord, Z_coord]

    result = noise_3_cube + temp_variance

    return result


def WNoise2D(NoiseTile,px,py,n):
            
    midx = np.ceil(px - 0.5).astype(int)
    midy = np.ceil(py - 0.5).astype(int)
    
    tx = midx - (px - 0.5)
    ty = midy - (py - 0.5)
    
    wx_n1 = 0.5 * tx**2
    wx_1  = 0.5 * (1-tx)**2
    wx_0  = 1 - wx_n1 - wx_1
    
    wy_n1 = 0.5 * ty**2
    wy_1  = 0.5 * (1-ty)**2
    wy_0  = 1 - wy_n1 - wy_1
    
    cx_n1 = (midx - 1) & (n - 1)
    cx_0  = (midx)  & (n - 1)
    cx_1  = (midx + 1) & (n - 1)  
    
    cy_n1 = (midy - 1) & (n - 1)
    cy_0  = (midy) & (n - 1)
    cy_1  = (midy + 1) & (n - 1) 

    weight1 = wx_n1 * wy_n1 
    weight2 = wx_0  * wy_n1   
    weight3 = wx_1  * wy_n1   
    weight4 = wx_n1 * wy_0  
    weight5 = wx_0  * wy_0   
    weight6 = wx_1  * wy_0   
    weight7 = wx_n1 * wy_1  
    weight8 = wx_0  * wy_1   
    weight9 = wx_1  * wy_1   
    
    index1 = NoiseTile[cy_n1,cx_n1]
    index2 = NoiseTile[cy_n1,cx_0]
    index3 = NoiseTile[cy_n1,cx_1]
    index4 = NoiseTile[cy_0,cx_n1]
    index5 = NoiseTile[cy_0,cx_0]
    index6 = NoiseTile[cy_0,cx_1]
    index7 = NoiseTile[cy_1,cx_n1]
    index8 = NoiseTile[cy_1,cx_0]
    index9 = NoiseTile[cy_1,cx_1]
                
    # Evaluate quadratic B-spline basis functions 
    result =    weight1 * index1 + \
                weight2 * index2 + \
                weight3 * index3 + \
                weight4 * index4 + \
                weight5 * index5 + \
                weight6 * index6 + \
                weight7 * index7 + \
                weight8 * index8 + \
                weight9 * index9 

    return result


def WNoise3D(NoiseTile,px,py,pz,n):
    
    midx = np.ceil(px - 0.5).astype(int)
    midy = np.ceil(py - 0.5).astype(int)
    midz = np.ceil(pz - 0.5).astype(int)
    
    tx = midx - (px - 0.5)
    ty = midy - (py - 0.5)
    tz = midz - (pz - 0.5)
    
    wx_n1 = 0.5 * tx**2
    wx_1  = 0.5 * (1-tx)**2
    wx_0  = 1 - wx_n1 - wx_1
    
    wy_n1 = 0.5 * ty**2
    wy_1  = 0.5 * (1-ty)**2
    wy_0  = 1 - wy_n1 - wy_1
    
    wz_n1 = 0.5 * tz**2
    wz_1  = 0.5 * (1-tz)**2
    wz_0  = 1 - wz_n1 - wz_1
    
    cx_n1 = (midx - 1) & (n - 1)
    cx_0  = (midx)  & (n - 1)
    cx_1  = (midx + 1) & (n - 1)  
    
    cy_n1 = (midy - 1) & (n - 1)
    cy_0  = (midy) & (n - 1)
    cy_1  = (midy + 1) & (n - 1) 
    
    cz_n1 = (midz - 1) & (n - 1) 
    cz_0  = (midz)  & (n - 1)
    cz_1  = (midz + 1) & (n - 1)
    

    weight1  = wx_n1 * wy_n1 * wz_n1 
    weight2  = wx_0  * wy_n1 * wz_n1  
    weight3  = wx_1  * wy_n1 * wz_n1 
    weight4  = wx_n1 * wy_0  * wz_n1 
    weight5  = wx_0  * wy_0  * wz_n1  
    weight6  = wx_1  * wy_0  * wz_n1  
    weight7  = wx_n1 * wy_1  * wz_n1 
    weight8  = wx_0  * wy_1  * wz_n1  
    weight9  = wx_1  * wy_1  * wz_n1 
    weight10 = wx_n1 * wy_n1 * wz_0 
    weight11 = wx_0  * wy_n1 * wz_0  
    weight12 = wx_1  * wy_n1 * wz_0  
    weight13 = wx_n1 * wy_0  * wz_0 
    weight14 = wx_0  * wy_0  * wz_0  
    weight15 = wx_1  * wy_0  * wz_0  
    weight16 = wx_n1 * wy_1  * wz_0 
    weight17 = wx_0  * wy_1  * wz_0  
    weight18 = wx_1  * wy_1  * wz_0  
    weight19 = wx_n1 * wy_n1 * wz_1 
    weight20 = wx_0  * wy_n1 * wz_1  
    weight21 = wx_1  * wy_n1 * wz_1  
    weight22 = wx_n1 * wy_0  * wz_1 
    weight23 = wx_0  * wy_0  * wz_1  
    weight24 = wx_1  * wy_0  * wz_1  
    weight25 = wx_n1 * wy_1  * wz_1 
    weight26 = wx_0  * wy_1  * wz_1  
    weight27 = wx_1  * wy_1  * wz_1  
    
    index1  = NoiseTile[cy_n1,cx_n1,cz_n1]
    index2  = NoiseTile[cy_n1,cx_0,cz_n1]
    index3  = NoiseTile[cy_n1,cx_1,cz_n1]
    index4  = NoiseTile[cy_0,cx_n1,cz_n1]
    index5  = NoiseTile[cy_0,cx_0,cz_n1]
    index6  = NoiseTile[cy_0,cx_1,cz_n1]
    index7  = NoiseTile[cy_1,cx_n1,cz_n1]
    index8  = NoiseTile[cy_1,cx_0,cz_n1]
    index9  = NoiseTile[cy_1,cx_1,cz_n1]
    index10 = NoiseTile[cy_n1,cx_n1,cz_0]
    index11 = NoiseTile[cy_n1,cx_0,cz_0]
    index12 = NoiseTile[cy_n1,cx_1,cz_0]
    index13 = NoiseTile[cy_0,cx_n1,cz_0]
    index14 = NoiseTile[cy_0,cx_0,cz_0]
    index15 = NoiseTile[cy_0,cx_1,cz_0]
    index16 = NoiseTile[cy_1,cx_n1,cz_0]
    index17 = NoiseTile[cy_1,cx_0,cz_0]
    index18 = NoiseTile[cy_1,cx_1,cz_0]
    index19 = NoiseTile[cy_n1,cx_n1,cz_1]
    index20 = NoiseTile[cy_n1,cx_0,cz_1]
    index21 = NoiseTile[cy_n1,cx_1,cz_1]
    index22 = NoiseTile[cy_0,cx_n1,cz_1]
    index23 = NoiseTile[cy_0,cx_0,cz_1]
    index24 = NoiseTile[cy_0,cx_1,cz_1]
    index25 = NoiseTile[cy_1,cx_n1,cz_1]
    index26 = NoiseTile[cy_1,cx_0,cz_1]
    index27 = NoiseTile[cy_1,cx_1,cz_1]
                
    # Evaluate quadratic B-spline basis functions 
    result =    weight1 * index1 + \
                weight2 * index2 + \
                weight3 * index3 + \
                weight4 * index4 + \
                weight5 * index5 + \
                weight6 * index6 + \
                weight7 * index7 + \
                weight8 * index8 + \
                weight9 * index9 + \
                weight10 * index10 + \
                weight11 * index11 + \
                weight12 * index12 + \
                weight13 * index13 + \
                weight14 * index14 + \
                weight15 * index15 + \
                weight16 * index16 + \
                weight17 * index17 + \
                weight18 * index18 + \
                weight19 * index19 + \
                weight20 * index20 + \
                weight21 * index21 + \
                weight22 * index22 + \
                weight23 * index23 + \
                weight24 * index24 + \
                weight25 * index25 + \
                weight26 * index26 + \
                weight27 * index27  
    
    return result


def WaveletNoise2D(initial_freq, 
                   nBands = 1,
                   n = 128,
                   initial_amp = 1,
                   persistence = 0.5, 
                   lacunarity = 2,
                   rnd_seed = 42):
    
    assert(nBands > 0)
    # Generate noise tile
    np.random.seed(rnd_seed)
    rnd_numbers = np.random.normal(0, 1, size = (n,n))
    noise_tile = GenerateNoiseTile2D(rnd_numbers,n)
    
    # Evaluate multiband noise
    px, py = np.meshgrid(range(n), range(n), indexing = 'xy')
    
    amplitude = initial_amp
    freq = initial_freq
    variance = np.zeros(nBands)
    result = np.zeros((n,n))
    
    for b in range(nBands):
    
        result += WNoise2D(noise_tile,
                           px * freq, 
                           py * freq, 
                           n) * amplitude
    
        variance[b] = amplitude**2
        
        amplitude *= persistence
        freq *= lacunarity
    
    variance = np.sum(variance)
    
    result /= np.sqrt(variance * 0.265) # 2D noise

    return result


def WaveletNoise3D(initial_freq, 
                   nBands,
                   n = 128,
                   initial_amp = 1,
                   persistence = 0.5, 
                   lacunarity = 2,
                   rnd_seed = 42):
    
    assert(nBands > 0)
    
    # Generate noise tile
    np.random.seed(rnd_seed)
    rnd_numbers = np.random.normal(0, 1, size = (n,n,n))
    noise_tile = GenerateNoiseTile3D(rnd_numbers, n)
    
    # Evaluate multiband noise
    px,py,pz = np.meshgrid(range(n), range(n), range(n), indexing = 'xy')
    
    amplitude = initial_amp
    freq = initial_freq
    variance = np.zeros(nBands)
    result = np.zeros((n,n,n))
    
    for b in range(nBands):
    
        result += WNoise3D(noise_tile,
                           px * freq, 
                           py * freq,
                           pz * freq, n) * amplitude
    
        variance[b] = amplitude**2
        
        amplitude *= persistence
        freq *= lacunarity
    
    variance = np.sum(variance)
    
    result /= np.sqrt(variance * 0.210) # 3D noise

    return result
