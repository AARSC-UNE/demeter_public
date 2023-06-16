#!/usr/bin/env python3
import geopandas
import os
from osgeo import gdal, gdalconst  
from tensorflow.keras.utils import Sequence
import rasterio
import math
import sys
import numpy as np
import uuid
import tensorflow as tf
from imgaug import augmenters as iaa

def calculatePatchSize(patchesdf, image):
    """_summary_

    Args:
        patchesdf (_type_): geopandas dataframe
        imageDir (_type_): A directory containing the images indicated by the image attribute table column in the dataframe

    Returns:
        _type_: _description_
    """
    patchesProj = patchesdf.crs.to_epsg()
    with rasterio.open(image) as img:
        patchGeom = geopandas.GeoSeries(patchesdf['geometry'][0])
        patchGeom.set_crs(epsg=patchesProj, inplace=True)
        patchGeom = patchGeom.to_crs(epsg=img.crs.to_epsg())
        area = float(patchGeom.area)
        patchSize = int(round(math.sqrt(area) / img.res[0], 0))

    return patchSize

def GDALTypeToNumpyType(gdaltype):
    """
    Given a gdal data type returns the matching
    numpy data type
    """
    dataTypeMapping = getDataTypeMapping()
    for (numpy_type, test_gdal_type) in dataTypeMapping:
        if test_gdal_type == gdaltype:
            return numpy_type
    print("Unknown GDAL datatype: %s" % gdaltype)
    sys.exit()

def getDataTypeMapping():
    return [
        (np.uint8, gdalconst.GDT_Byte),
        (np.bool, gdalconst.GDT_Byte),
        (np.int16, gdalconst.GDT_Int16),
        (np.uint16, gdalconst.GDT_UInt16),
        (np.int32, gdalconst.GDT_Int32),
        (np.uint32, gdalconst.GDT_UInt32),
        (np.single, gdalconst.GDT_Float32),
        (np.float, gdalconst.GDT_Float64)
    ]


def eastNorth2rowCol(transform, east, north):
    """
    Apply the GDAL transformation from (east, north) to (row, col) pixel coordinates.
    Will work the same on either scalars or arrays.

    Note that (east, north) is really just the world coordinate system of the file in question, and could actually refer
    to any projection.

    """
    col = (transform[0] * transform[5] -
           transform[2] * transform[3] + transform[2] * north -
           transform[5] * east) / (transform[2] * transform[4] - transform[1] * transform[5])

    row = (transform[1] * transform[3] - transform[0] * transform[4] -
           transform[1] * north + transform[4] * east) / (transform[2] * transform[4] - transform[1] * transform[5])
    return (row, col)


def arrayToImage(imageArray, outfile, proj=None, geotransform=None, GType=gdal.GDT_Byte, transpose=True):
    """
    Transposes (3d) or expands (2d) and export and array to a tiff file.
    imageArray is the array to be exported.
    proj is the spatial reference system to use and geotransform is the transformation
    to use.
    outfile must contain the .tif extension.
    verbose prints stuff out.

    """

    driver = gdal.GetDriverByName('GTiff')
    if transpose:
        # Reshape the array to be in gdal's order
        if len(imageArray.shape) == 3:
            imageArray = np.transpose(imageArray, (2, 0, 1))
        elif len(imageArray.shape) == 2:
            imageArray = np.expand_dims(imageArray, axis=0)

    # Read the diamentions of the file
    (nBands, nRows, nCols) = imageArray.shape

    # Create the output image file
    creationOptions = ['COMPRESS=DEFLATE', 'INTERLEAVE=BAND', 'TILED=YES',
                       'BIGTIFF=IF_SAFER']
    ds = driver.Create(outfile, nCols, nRows, nBands, GType,
                       creationOptions)

    # Write the 2d array to the band
    for i in range(nBands):
        b = ds.GetRasterBand(i + 1)
        b.WriteArray(imageArray[i, :, :])
        b = None

    # Set the projection
    if proj:
        ds.SetProjection(proj)
    if geotransform:
        ds.SetGeoTransform(geotransform)
    ds = None

    return outfile


def scaleImage_0_255_image(image):
    """

    :param image: Input image as an numpy array
    :param method: The scalling range which can be one of the following. '0-255', '-1-1', '0-1'.
    :return: a scalled numpy array
    """

    max = np.max(image)
    min = np.min(image)
    diff = max - min
    if diff <= 0:
        diff = 1
    image = np.round(255.0 * ((image - min) / (diff)), 0)
    return image.astype(np.uint8)


class myAffine(iaa.Affine):
    def __init__(self, scale=None, translate_percent=None, translate_px=None,
                 rotate=None, shear=None, order=1, cval=0, mode="constant",
                 fit_output=False, backend="auto",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(myAffine, self).__init__(scale=scale, translate_percent=translate_percent, translate_px=translate_px,
                                       rotate=rotate, shear=shear, order=order, cval=cval, mode=mode,
                                       fit_output=fit_output, backend=backend,
                                       seed=seed, name=name,
                                       random_state=random_state, deterministic=deterministic)
        self._mode_segmentation_maps = mode


def applyAugmentation(images, masks, seed=None):
    noiseList = [iaa.SaltAndPepper(p=(0, 0.005), per_channel=True),
                 iaa.SaltAndPepper(p=(0, 0.005), per_channel=False),
                 iaa.MultiplyElementwise(mul=(0.95, 1.05), per_channel=True),
                 iaa.MultiplyElementwise(mul=(0.95, 1.05), per_channel=False),
                 iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255), per_channel=False),
                 iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255), per_channel=True),
                 iaa.AdditivePoissonNoise(lam=(0, 10), per_channel=False),
                 iaa.AdditivePoissonNoise(lam=(0, 10), per_channel=True)]

    contrastList = [iaa.GammaContrast(gamma=(0.1, 3.0), per_channel=True),
                    iaa.GammaContrast(gamma=(0.1, 3.0), per_channel=False),
                    iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
                    iaa.AllChannelsCLAHE(clip_limit=(1, 5), tile_grid_size_px=(3, 12), tile_grid_size_px_min=3,
                                         per_channel=True),
                    iaa.LogContrast(gain=(0.6, 1.4), per_channel=True),
                    iaa.LinearContrast((0.4, 2.5), per_channel=True),
                    iaa.Multiply(mul=(0, 2)),
                    iaa.Multiply(mul=(0, 2), per_channel=True),
                    iaa.AllChannelsHistogramEqualization()]

    geomList = [myAffine(scale=(0.8, 1.2), translate_percent=None, translate_px=None, rotate=(-360, 360),
                         shear=(-45, 45), order=3, mode='symmetric', fit_output=False, seed=seed),
                # iaa.PerspectiveTransform(scale=(0.0, 0.1)), done by affine
                iaa.ElasticTransformation(alpha=(0, 1.0), sigma=(0.5, 1), seed=seed),
                # iaa.Rot90((0, 3)), done by affine
                iaa.Fliplr(0.5, seed=seed),
                iaa.Flipud(0.5, seed=seed)]

    otherList = [iaa.GaussianBlur(sigma=(0, 1.0)),
                 iaa.Clouds(),
                 iaa.Fog()]

    seq = iaa.Sequential([iaa.OneOf(contrastList), iaa.OneOf(noiseList), iaa.OneOf(geomList),
                          iaa.Sometimes(0.5, iaa.OneOf(otherList))], random_order=False)
    
    aug_images, aug_masks = seq(images=images, segmentation_maps=masks)

    return (aug_images, aug_masks)


class generator(Sequence):
    """A data generator to produces batches a image chips and labels based on a patches layer, training feature layers
    and imagery.

    Args:
        Sequence ([type]): [description]
    """

    def __init__(self, imageDir, database, numClasses, patches, batch_size, augment=True, scale=True, trainingDataDir='~/tmp'):
        """Initilisation function

        Args:
            imageDir ([str]): A directory where the imager can be found
            database ([type]): [description]
            numClasses ([type]): [description]
            patches ([type]): [description]
            batch_size ([type]): [description]
        """
        self.database = database
        self.numClasses = numClasses
        self.batch_size = batch_size
        self.augment = augment
        self.scale = scale
        self.patchesdf = geopandas.read_file(self.database, layer=patches)
        self.numPatches = len(self.patchesdf)
        self.imageDict = {}
        for image in self.patchesdf['image'].unique():
            self.imageDict[image] = os.path.join(imageDir, image)
        self.crs = self.patchesdf.crs.to_epsg()
        img = gdal.Open(self.imageDict[self.patchesdf['image'][0]])
        self.patchSize = calculatePatchSize(self.patchesdf, imageDir)
        self.bands = img.RasterCount
        self.dtype = GDALTypeToNumpyType(img.GetRasterBand(1).DataType)
        self.trainingDataDir = trainingDataDir
        del img
        

    def on_epoch_end(self):
        self.patchesdf = self.patchesdf.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):

        # Get the patches we need and define the arrays
        batches = self.patchesdf[index * self.batch_size:(index + 1) * self.batch_size]
        images = np.zeros((self.batch_size, self.patchSize, self.patchSize, self.bands), dtype=self.dtype)
        labels = np.zeros((self.batch_size, self.patchSize, self.patchSize, self.numClasses), dtype=np.uint8)

        # Loop through the patches
        for i, row in batches.iterrows():

            # Define the patch image and label names
            imageName = os.path.join(self.trainingDataDir, '{}_image.tif'.format(i))
            labelName = os.path.join(self.trainingDataDir, '{}_labels.tif'.format(i))

            # Get the patch information
            patchGeom = geopandas.GeoSeries(row['geometry'])
            patchGeom.set_crs(epsg=self.crs, inplace=True)
            proj_ref = int(row['epsg'])
            if self.crs != proj_ref:
                patchGeom = patchGeom.to_crs(epsg=proj_ref)
            imageds = gdal.Open(self.imageDict[row['image']])
            xmin = patchGeom.total_bounds[0]
            ymax = patchGeom.total_bounds[3]
            transform = imageds.GetGeoTransform()

            # Create or read the patch image
            if not os.path.exists(imageName):
                startrow, startcol = eastNorth2rowCol(transform, xmin, ymax)
                startrow = round(startrow, 0)
                startcol = round(startcol, 0)
                patch = imageds.ReadAsArray(xoff=startcol, yoff=startrow, xsize=self.patchSize, ysize=self.patchSize)
                arrayToImage(patch, imageName, imageds.GetProjection(), transform, imageds.GetRasterBand(1).DataType, transpose=False)
            else:
                imageds = gdal.Open(imageName)
                patch = imageds.ReadAsArray()

            # Get the patch image in the correct format
            patch = np.transpose(patch, (1, 2, 0))

            # Create or read the patch label
            if not os.path.exists(labelName):
                rasterizeOptions = gdal.RasterizeOptions(format='MEM', outputType=gdal.GDT_Byte,
                                                        outputSRS=imageds.GetProjection(),
                                                        outputBounds=patchGeom.total_bounds,
                                                        width=self.patchSize, height=self.patchSize, xRes=transform[1], yRes=transform[1],
                                                        noData=0, attribute='value', layers=row['layer'])
                
                label = gdal.Rasterize(str(uuid.uuid4()), self.database, options=rasterizeOptions).ReadAsArray()
                del rasterizeOptions
                arrayToImage(label, labelName, imageds.GetProjection(), transform)
            else:
                labelds = gdal.Open(labelName)
                label = labelds.ReadAsArray()
            
            # Get the label in the correct format
            if self.numClasses>1:
                label = tf.one_hot(label-1, self.numClasses, dtype=patch.dtype).numpy()
            else:
                label = np.expand_dims(label, axis=2)

            # Add to the array
            images[i-index*self.batch_size] = patch
            labels[i-index*self.batch_size] = label
            del imageds, patchGeom, proj_ref, label, patch, xmin, ymax, transform

        # Augment and return
        if self.scale:
            images = scaleImage_0_255_image(images)
        if self.augment:
            images, labels = applyAugmentation(images, labels)

        # Clean up
        del batches
        gc.collect()
        return (images.astype(np.float32), labels.astype(np.float32))

    def __len__(self):
        return self.numPatches//self.batch_size