import dicom, os

fnames = os.listdir('.')
for filename in fnames:
    print(filename)
    if filename.endswith('.dcm'):
        dcm = dicom.ReadFile(filename,force=True)
        a = dcm.pixel_array[190:190+105, 189:189+155]
        dcm.PixelData = a.tostring()
        dcm.Columns = 155
        dcm.Rows = 105
        dcm.save_as('crop_'+filename)

