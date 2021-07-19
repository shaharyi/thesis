"""
Viewing MRI data with cut plane and iso surface.

This example downloads an MRI scan, turns it into a 3D numpy array and
visualizes it.

First we extract some internal structures of the brain by defining a
volume of interest around them, and using iso surfaces.

Then we display two cut planes to show the raw MRI data itself.

Finally we display the outer surface, but we restrict it to volume of
interest to leave a cut for the cut planes.

For an example of feature extraction from MRI data using Mayavi and vtk,
see :ref:`example_tvtk_segmentation`.
"""

### Read the data in a numpy 3D array ##########################################
import numpy as np
data = np.load('s3.npy')
#data.shape = (109, 256, 256)
print(data.shape)
print(data.dtype)
data = data.T

# Display the data #############################################################
from mayavi import mlab

mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))

src = mlab.pipeline.scalar_field(data)
# Our data is not equally spaced in all directions:
#src.spacing = [1, 1, 1.5]
#src.update_image_data = True

# Extract some inner structures: the ventricles and the inter-hemisphere
# fibers. We define a volume of interest (VOI) that restricts the
# iso-surfaces to the inner of the brain. We do this with the ExtractGrid
# filter.
blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
voi = mlab.pipeline.extract_grid(blur)
#voi.set(x_min=125, x_max=193, y_min=92, y_max=125, z_min=34, z_max=75)
#voi.set(x_min=101, x_max=101+86, y_min=77, y_max=77+57, z_min=86, z_max=120)
voi.set(x_min=101, x_max=101+86, y_min=77, y_max=77+57, z_min=86, z_max=86+68)

mlab.pipeline.iso_surface(voi, contours=[320,], #contours=[1610, 2480],colormap='Spectral')
                          color=(0.9, 0.1, 0.4))



# Add two cut planes to show the raw MRI data. We use a threshold filter
# to remove cut the planes outside the brain.
thr = mlab.pipeline.threshold(src, low=10) #112)
cut_plane = mlab.pipeline.scalar_cut_plane(thr,
                                plane_orientation='y_axes',
                                colormap='black-white')
                                #,vmin=140,vmax=260)

#cut_plane.implicit_plane.origin = (136, 111.5, 82)
cut_plane.implicit_plane.origin = (136, 140, 53*2.2)
#cut_plane.implicit_plane.normal = (-.5, 1, 0)
cut_plane.implicit_plane.widget.enabled = False

cut_plane2 = mlab.pipeline.scalar_cut_plane(thr,
                                plane_orientation='z_axes',
                                colormap='black-white')
                                #,vmin=140,vmax=260)
#cut_plane2.implicit_plane.origin = (136, 111.5, 82)
cut_plane2.implicit_plane.origin = (136, 140, 53*2.2)
cut_plane2.implicit_plane.widget.enabled = False


# Extract two views of the outside surface. We need to define VOIs in
# order to leave out a cut in the head.
voi2 = mlab.pipeline.extract_grid(src)
#voi2.set(y_min=112)
voi2.set(y_min=140)
outer = mlab.pipeline.iso_surface(voi2, #contours=[1776, ],
                                  contours=[40,],
                                        color=(0.8, 0.7, 0.6))

voi3 = mlab.pipeline.extract_grid(src)
#voi3.set(y_max=112, z_max=53)
voi3.set(y_max=140, z_max=53*2.2)
outer3 = mlab.pipeline.iso_surface(voi3, #contours=[1776, ],
                                   contours=[40,],
                                         color=(0.8, 0.7, 0.6))


#mlab.view(-125, 54, 326, (145.5, 138, 66.5))
#mlab.roll(-175)

mlab.show()
raise Exception('Done')
