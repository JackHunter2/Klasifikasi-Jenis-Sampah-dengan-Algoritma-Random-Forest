import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

IMAGE_SIZE = (128, 128)
HIST_BINS = 32

def extract_features_from_image_pil(img_pil, resize=IMAGE_SIZE, hist_bins=HIST_BINS):
    img_res = img_pil.resize(resize)
    arr = np.array(img_res)

    # Mean & std RGB
    mean_r = np.mean(arr[:,:,0]); std_r = np.std(arr[:,:,0])
    mean_g = np.mean(arr[:,:,1]); std_g = np.std(arr[:,:,1])
    mean_b = np.mean(arr[:,:,2]); std_b = np.std(arr[:,:,2])

    # Histogram
    hist_r, _ = np.histogram(arr[:,:,0], bins=hist_bins, range=(0,255))
    hist_g, _ = np.histogram(arr[:,:,1], bins=hist_bins, range=(0,255))
    hist_b, _ = np.histogram(arr[:,:,2], bins=hist_bins, range=(0,255))

    # GLCM
    gray = img_res.convert('L')
    gray_arr = np.array(gray)
    glcm = graycomatrix(gray_arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]

    features = [
        mean_r, mean_g, mean_b,
        std_r, std_g, std_b,
        contrast, homogeneity, energy
    ]

    features.extend(hist_r.tolist())
    features.extend(hist_g.tolist())
    features.extend(hist_b.tolist())

    features.append(img_pil.size[0])
    features.append(img_pil.size[1])

    return np.array(features, dtype=np.float32)
