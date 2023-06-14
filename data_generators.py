#!/usr/bin/env python3



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
        self.dtype = aarsc_utils.GDALTypeToNumpyType(img.GetRasterBand(1).DataType)
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
                startrow, startcol = aarsc_utils.eastNorth2rowCol(transform, xmin, ymax)
                startrow = round(startrow, 0)
                startcol = round(startcol, 0)
                patch = imageds.ReadAsArray(xoff=startcol, yoff=startrow, xsize=self.patchSize, ysize=self.patchSize)
                aarsc_utils.arrayToImage(patch, imageName, imageds.GetProjection(), transform, imageds.GetRasterBand(1).DataType, transpose=False)
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
                aarsc_utils.arrayToImage(label, labelName, imageds.GetProjection(), transform)
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
            images = aarsc_utils.scaleImage_0_255_image(images)
        if self.augment:
            images, labels = applyAugmentation(images, labels)

        # Clean up
        del batches
        gc.collect()
        return (images.astype(np.float32), labels.astype(np.float32))

    def __len__(self):
        return self.numPatches//self.batch_size