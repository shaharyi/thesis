import dicom, os

fnames = os.listdir('.')
i=1
for filename in fnames:
    print(filename)
    if filename.endswith('.dcm'):
        dcm = dicom.ReadFile(filename,force=True)
        a = dcm.pixel_array[155:155+113, 202:202+171]
        dcm.PixelData = a.tostring()
        dcm.Columns = 171
        dcm.Rows = 113
        dcm.save_as('crop_'+filename)
