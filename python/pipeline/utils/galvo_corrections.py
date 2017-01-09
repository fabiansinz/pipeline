from .. import PipelineException
from scipy.interpolate import interp1d, interp2d
import numpy as np


def correct_motion(img, xymotion):
    """
    motion correction for 2P scans.
    :param img: 2D image [x, y]
    :param xymotion: x, y motion offsets
    :return: motion corrected image [x, y]
    """
    if not isinstance(img, np.ndarray) and len(xymotion) != 2:
        raise PipelineException('Cannot correct image. Only 2D images please.')
    sz = img.shape
    if not hasattr(correct_motion,'x1'):
        y1, x1 = np.ogrid[0: sz[0], 0: sz[1]]
        correct_motion.x1 = x1
        correct_motion.y1 = y1
    else:
        x1, y1 = correct_motion.x1, correct_motion.y1
    y2, x2 = np.arange(sz[0]) + xymotion[1], np.arange(sz[1]) + xymotion[0]

    interp = interp2d(x1, y1, img, kind='cubic')
    return interp(x2, y2)



def correct_raster(img, raster_phase, fill_fraction):
    """
    raster correction for resonant scanners.
    :param img: 2D image [x, y].
    :param raster_phase: phase difference beetween odd and even lines.
    :param fill_fraction: ratio between active acquisition and total length of the scan line. see scanimage.
    :return: raster-corrected image [x, y].
    """

    if not len(img.shape) == 2:
        raise PipelineException('Image must have two dimensions only')

    ix = np.arange(-img.shape[1] / 2 + 0.5, img.shape[1] / 2 + 0.5) / (img.shape[1] / 2)
    tx = np.arcsin(ix * fill_fraction)

    im = np.array(img).T
    img = img.T
    extrapval = np.mean(img)
    im[::2, :] = interp1d(ix, img[::2, :], kind='linear', bounds_error=False,
                                  fill_value=extrapval)(np.sin(tx + raster_phase) / fill_fraction)

    im[1::2, :] = interp1d(ix, img[1::2, :], kind='linear', bounds_error=-False,
                                   fill_value=extrapval)(np.sin(tx - raster_phase) / fill_fraction)
    return img.T


def plot_raster(filename, key):
    """
    plot origin frame, raster-corrected frame, and reversed raster-corrected frame.
    :param filename:  full file path for tiff file.
    :param key: scan key for the tiff file.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pipeline import preprocess, experiment
    from tiffreader import TIFFReader
    reader = TIFFReader(filename)
    img = reader[:, :, 0, 0, 100]
    raster_phase = (preprocess.Prepare.Galvo() & key).fetch1['raster_phase']
    newim = correct_raster(img, raster_phase, reader.fill_fraction)
    nnewim = correct_raster(newim, -raster_phase, reader.fill_fraction)
    print(np.mean(img - nnewim))

    plt.close()
    with sns.axes_style('white'):
        fig = plt.figure(figsize=(15, 8))
        gs = plt.GridSpec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img[:, :, 0, 0, 0], cmap=plt.cm.gray)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(newim[:, :, 0, 0, 0], cmap=plt.cm.gray)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(nnewim[:, :, 0, 0, 0], cmap=plt.cm.gray)
    plt.show()
