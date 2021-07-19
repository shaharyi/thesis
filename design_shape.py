from numpy import *
from numpy.linalg import *
from pdb import *
from mayavi import mlab
from scipy.ndimage import interpolation, morphology
from scipy.ndimage.filters import generic_gradient_magnitude, prewitt, sobel, laplace
import os

def flood_fill (data, node, target_color, replacement_color):
    print('Flood-fill started..')
    offsets = array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
    q = [node]
    while len(q)>0: 
        n = tuple(q.pop())
        if data[n] == target_color:
            data[n] = replacement_color
            q.extend(n+offsets)
    print('.. Flood-fill done')

def show_series(logFileName = None):
    if logFileName == None:
        logFileName=os.listdir('.')[0]
        if not logFileName.endswith('.log'):
            raise Exception('Log file not found')
    cube = load('rbf0.npy')
    X,Y,Z = cube.shape
    i=0
    triplets = []
    with open(logFileName,'rt') as f:
        for line in f:
            pos = line.find('array')
            if pos>-1:
                s = line[pos:] + f.next() + f.next()
                print(s)
                triplets.append( eval(s) )
    for i in [3,7,11]: #range(len(triplets)):
        rbf = 'rbf%d.npy' % i
        print(rbf)
        cube = load(rbf)
        src = mlab.pipeline.scalar_field(cube)
        iso = mlab.pipeline.iso_surface(src)
        iso.actor.property.backface_culling = True
        iso.actor.property.opacity = .6
        iso.contour.auto_contours = False
        iso.contour.contours[0:1] = [0.2]
        outline = mlab.outline(extent=[0,X,0,Y,0,Z],line_width=1)
        outline.outline_mode = 'cornered'
        for j in range(i):
            cut_plane = mlab.pipeline.cut_plane(src)
            surf = mlab.pipeline.surface(cut_plane)
            surf.enable_contours = True
            a,b,c,d = calc_plane_def(triplets[j])
            n = array([a,b,c]) #normal
            n = n / norm(n)    #normalize to size 1
            implicit_plane = cut_plane.filters[0]
            implicit_plane.widget.origin = triplets[j][0]
            implicit_plane.widget.normal = n
            implicit_plane.widget.enabled = False
        mlab.show()#stop=True)
    mlab.close(all=True)
    mlab.get_engine().stop()
                
def calc_plane_def((p1,p2,p3)):
    "return plane params (a,b,c,d)"
    normal = cross(p2-p1,p3-p1)
    a,b,c = normal.astype(float)
    d = -dot(normal, p1)
    return (a,b,c,d)

#show_series()
#raise Exception('exit_program')


def show_series_old(logFileName):
    cube = load('rbf0.npy')
    X,Y,Z = cube.shape
    i=0
    triplets = []
    with open(logFileName,'rt') as f:
        for line in f:
            pos = line.find('array')
            if pos>-1:
                s = line[pos:] + f.next() + f.next()
                print(s)
                triplets.append( eval(s) )
    for i in [0,1,2]: #range(len(triplets)):
        triplet = triplets[i]
        rbf = 'rbf%d.npy' % i
        print(rbf)
        cube = load(rbf)
        src = mlab.pipeline.scalar_field(cube)
        iso = mlab.pipeline.iso_surface(src)
        iso.actor.property.opacity = .3
        iso.contour.auto_contours = False
        iso.contour.contours[0:1] = [0.2]
        outline = mlab.outline(extent=[0,X,0,Y,0,Z],line_width=1)
        outline.outline_mode = 'cornered'
        cut_plane = mlab.pipeline.cut_plane(src)
        surf = mlab.pipeline.surface(cut_plane)
        surf.enable_contours = True
        a,b,c,d = calc_plane_def(triplet)
        n = array([a,b,c]) #normal
        n = n / norm(n)    #normalize to size 1
        implicit_plane = cut_plane.filters[0]
        implicit_plane.widget.origin = triplet[0]
        implicit_plane.widget.normal = n
        implicit_plane.widget.enabled = False
        mlab.show()#stop=True)
    mlab.close(all=True)
    mlab.get_engine().stop()
                
def trefoil_knot(N, r=.08):
    cube = zeros((N,N,N),dtype=uint8)
    trans = .5*ones(3)
    normalizer = 2.5*array([3,3,1])
    for t in linspace(0,2*pi,num=200):
##        x = (2+cos(3*t))*cos(2*t)
##        y = (2+cos(3*t))*sin(2*t)
##        z = sin(3*t)
        x = sin(t) + 2*sin(2*t)
        y = cos(t) - 2*cos(2*t)
        z = -sin(3*t)            
        x,y,z = (x,y,z) / normalizer + trans
        print((x,y,z))
        cube += sphere(N,(x,y,z),r)
    cube = clip(cube,0,1)
    return cube 

def shape_s(N):
    centers = (N * array([
                      [.70, .14, .70],
                      [.80, .18, .80],
                      [.84, .30, .84],
                      [.82, .40, .82],
                      [.68, .48, .68],                      
                      [.55, .54, .55],
                      [.48, .58, .48],
                      [.46, .70, .46],
                      [.52, .82, .52],
                      [.62, .86, .62],
                      ])).round().astype(int)
    radii   = N * array([.1,.11,.11,.12,.13,.12,.11,.11,.1,.1])
    return synth_data(N,centers,radii)

def shape_blimps(N, n):
    centers = (N * array([
        [.5, .5, .5],
        [.3, .3, .5],
        [.7, .3, .5],
        [.3, .7, .5],
                      ])).round().astype(int)
    radii   = N * array([.35, .2, .2, .2])
    return synth_data(N,centers[:n+1],radii[:n+1])

def shape_cross(N):
    t,r = tube(N,(.5,.5,.5),(.3,.4,.3),.1)
    cube = t.astype(bool)
    t,r = tube(N,(.5,.5,.5),(.3,.3,.7),.1)
    cube = cube + t.astype(bool)
    t,r = tube(N,(.5,.5,.5),(.6,.7,.3),.1)
    cube = cube + t.astype(bool)
    return 255*cube.astype(uint8)

def shape_cross2(N):
    t,r = tube(N,(.5,.5,.5),(.3,.3,.3),.1)
    cube = t.astype(bool)
    t,r = tube(N,(.5,.5,.5),(.3,.3,.7),.1)
    cube = cube + t.astype(bool)
    t,r = tube(N,(.5,.5,.5),(.7,.3,.5),.1)
    cube = cube + t.astype(bool)
    return 255*cube.astype(uint8)

def synth_data(N,centers,radii):
    x,y,z = mgrid[0:N,0:N,0:N]
    cube = zeros((N,N,N),bool)
    for i,c in enumerate(centers):
        sphere = (x-c[0])**2 + (y-c[1])**2 +(z-c[2])**2 <= radii[i]**2
        cube = cube + sphere
    return 255*cube.astype(uint8)
    
def torusY(N,c,R,r,xstretch=.8):
    x,y,z = mgrid[0:N,0:N,0:N]
    R,r = N*R, N*r
    c = N*array(c)
    cube = (R-sqrt((xstretch*(z-c[2]))**2+(y-c[1])**2))**2 + (x-c[0])**2 <= r**2
    return 255*cube.astype(uint8)

def torus(N,c,R,r,xstretch=.8):
    x,y,z = mgrid[0:N,0:N,0:N]
    R,r = N*R, N*r
    c = N*array(c)
    cube = (R-sqrt((xstretch*(x-c[0]))**2+(y-c[1])**2))**2 + (z-c[2])**2 <= r**2
    return 255*cube.astype(uint8)

def letterS(N):
    R, r = .175, .12
    cube = torus(N,(.5,.5,.5),R,r)
    start = round((.5+r)*N)
    end   = round((.5+R+r)*N)
    cube[:.5*N,:start,:] = cube[:.5*N,end-start:end,:]
    cube[:.5*N,start:,:] = 0
        
    start = round((.5-r)*N)
    end   = round((.5-R-r)*N)
    cube[.5*N:,start:,:] = cube[.5*N:,end:end+(N-start),:]
    cube[.5*N:,:start,:] = 0
    ##cube = interpolation.shift(cube,(0,N*.15,0))

    cube += sphere(N, (.5, .5-2*R, .5), r)
    cube += sphere(N, (.5, .5+2*R, .5), r)

    cube = interpolation.rotate(cube,angle=30,axes=(0,2),reshape=False)    
    cube = interpolation.rotate(cube,angle=60,axes=(1,2),reshape=False)    
    return cube.astype(uint8)

def torus_rot(N):
    cube = torus(N,(.5,.5,.3),.25,.12)
    cube = interpolation.rotate(cube,angle=45,axes=(0,2),reshape=False)
    return cube

def tube(N,c1,c2,r):
    cube = zeros((N,N,N))
    c1,c2 = map(array,[c1,c2])
    dist = N*norm(c2-c1)
    step = (c2-c1)/dist
    c = c1
    origr  = r
    for i in range(int(dist)):
        fluc = .5 if random.rand()<.5 else -.5
        fluc /= N
        if RANDOM and abs((r+fluc)/origr-1)<.35 and random.rand()<.5:
            r += fluc
        cube += sphere(N,c,r)
        c += step
    return cube,r

def xarc(N,c,R,r,pert=0,zdrift=0):
    c=array(c)
    cube = zeros((N,N,N),dtype=uint8)
    for rad in linspace(-pi/2,0,num=20):
        x,y,z = R*cos(rad), R*sin(rad), 0
        cube += sphere(N,c+(x,y,z),r)
        r += pert*(random.rand()-.5)
        c[2] += zdrift
    cube = clip(cube,0,1)
    return cube

def yarc(N,c,R,r):
    c=array(c)
    cube = zeros((N,N,N),dtype=uint8)
    for rad in linspace(-pi/2,0,num=20):
        x,y,z = 0,R*cos(rad), R*sin(rad)
        cube += sphere(N,c+(x,y,z),r)
    return cube

def ninja(N,margin=.23,radius=.1,arc=.15):
    m=margin
    r=radius
    a=arc
    cube = zeros((N,N,N),uint8)
    c, r = tube(N,(m,m,m),(1-m-a,m,m),r)
    cube += c
    cube += xarc(N,(1-m-a,m+a,m),a,r)
    c, r = tube(N,(1-m,m+a,m),(1-m,1-m-a,m),r)
    cube += c
    cube += yarc(N,(1-m,1-m-a,m+a),a,r)
    c, r = tube(N,(1-m,1-m,m+a),(1-m,1-m,1-m),r)
    cube += c
##    con = morphology.generate_binary_structure(3, 2)
##    cube = morphology.binary_opening(cube,
##                structure=con,iterations=1).astype(uint8)
    return cube

def ninja_twisted(N,margin=.23):
    m=margin
    cube = ninja(N,margin=m)
    cube = interpolation.rotate(cube,angle=30,axes=(0,1),reshape=False)
    c = zeros((N,N,N),uint8)
    c[:,:,N*m/2:] = cube[:,:,:-N*m/2]
    cube = c
    cube = interpolation.rotate(cube,angle=20,axes=(0,2),reshape=False)
    return cube

def moveX(N,cube,s):
    cube[:-s*N,:,:] = cube[s*N:,:,:]
    cube[-s*N:,:,:] = 0
    return cube

def torus_twisted(N):
    cube = torus(N,(.5,.5,.5),.26,.14,.82)
    cube = interpolation.rotate(cube,angle=45,axes=(0,2),reshape=False)
    cube = interpolation.rotate(cube,angle=45,axes=(0,1),reshape=False)
    return cube
    
def letterC(N):
    r=.14
    cube = torus(N,(.5,.5,.5),.26,r)
    cube[:.5*N,:,:] = 0
    cube += sphere(N,(.5,.24,.5),r)    
    cube += sphere(N,(.5,.76,.5),r)
    cube[cube>0]=255
    cube = moveX(N,cube,.2)
    return cube

def letterC_humble(N):
    cube = torus(N,(.5,.5,.2),.26,.12)
    xlen = cube.shape[0]
    cube[:.5*N,:,:] = 0
    cube += sphere(N,(.5,.24,.2),.12)    
    cube += sphere(N,(.5,.76,.2),.12)
    return cube

def horseshoe(N):
    cube = torus(N,(.5,.5,.3),.26,.12)
    xlen = cube.shape[0]
    cube[:xlen/2,:,:] = 0
    cube[.2*N:xlen/2,:,:] = cube[xlen/2,:,:]
    cube += sphere(N,(.2,.24,.3),.12)
    cube += sphere(N,(.2,.76,.3),.12)
##    con = morphology.generate_binary_structure(3, 1)
##    cube = morphology.binary_opening(cube,
##                structure=con,iterations=10).astype(uint8)
#    cube = interpolation.rotate(cube,angle=30,axes=(0,2),reshape=False)
#    cube[:-10,:,:] = cube[10:,:,:]
#    cube[-10:,:,:] = 0
    cube[cube>0] = 1
    return cube.astype(uint8)

def sphere(N,c=(.5,.5,.5),r=.5):
    x,y,z = mgrid[0:N,0:N,0:N]
    r = N*r
    c = N*array(c)
    cube = (x-c[0])**2 + (y-c[1])**2 +(z-c[2])**2 <= r**2
    return cube.astype(uint8)

def shape_s2(N):
    centers = (N * array([
       [70,200,40],
       [45,140,50],
       [90,85,60],
       [130,140,70],
       [170,190,80],
       [205,140,90],
##       [220,140,90],
       [200,80,100],
       ])/256
       ).round().astype(int)
    radii   = N * .15625* ones(len(centers))
    print(centers.max(1)+radii)
    return synth_data(N,centers,radii)

def random_sample(cube, n = 1000, thresh=1):
    pts = where(cube>=thresh)
    pts = vstack(pts).T
    idx = random.randint(len(pts),size=n)
    return pts[idx]

def earth_mover_dist(hist1,hist2):
    e = 0
    emd = 0
    for i in range(len(hist1)):
        e = hist1[i] + e - hist2[i]
        emd += abs(e)
    return emd

def MinkowskiL1_dist(hist1,hist2):
    return sum(abs(hist1 - hist2))

def D2(cube,thresh):
    N=len(cube)
    pts1 = random_sample(cube, N**3/2, thresh)
    pts2 = random_sample(cube, N**3/2, thresh)
    d = sqrt(sum((pts1-pts2)**2,axis=1))
    h,bins = histogram(d,bins=N, range=(0,N),density=True)
    return h

def bamba(N):
    cube = xarc(N,(.26,.75,.2),.5,.15,pert=.05,zdrift=.03)
    #cube = interpolation.rotate(cube,angle=30,axes=(0,2),reshape=False)
    return cube

N=128
RANDOM=True
#cube = torus_twisted(N)
#mlab.figure(bgcolor=(0, 0, 0), size=(640, 480))
#cube = torus(N,(.5,.5,.2),.25,.12)
#cube = horseshoe(64)
#cube = bamba(N)
#cube = load('s3_rbf_256.npy')


##cube = load('S3_ventricle2.npy').astype(float)
##c = laplace(cube)
##c = (c-c.mean())/c.std()
##c=abs(c)
##c[c<1] = 0
##cube = (c*255).astype(uint8)
##
##c[c<3] = 0
##c=morphology.distance_transform_edt(c==0)
###c = (c-c.mean())/c.std()
##c *=10
##cube = c.astype(uint8)

##shape = cube.shape #(49L, 113L, 171L)
##grads = calc_lap3d_singlecore(cube)
##grads = grads.reshape(prod(shape),3)
##c = array([norm(x) for x in grads])
##c = (c-c.mean())/c.std()
##c[c<1] = 0
##cube = (c*255).astype(uint8)
##cube = cube.reshape(shape)


##b = horseshoe(128)
##c = 1-b
##dtB = morphology.distance_transform_cdt(b)
##dtC = morphology.distance_transform_cdt(c)
##cube = dtB + dtC

#cube = letterC(N)

#cube = load('s3_rbf_256.npy')
#cube = load('S3_ventricle2_iso_gt.npy')

##cube = load('rbf0.npy')
##cube[cube>.1] = 1
##cube[cube<=.1] = 0
##cube[:8,:,:]=0
##cube[-8:,:,:] = 0
##cube=cube.astype(uint8)
##save('S3_ventricle2_iso_gt.npy',cube)

#cube = load('rbf0.npy')
#from sesame import *
#cube = compute_ground_truth((109L, 73L, 76L))
#cube = load('Cerebellum_box_gt.npy')
#cube = interpolation.rotate(cube,angle=45,axes=(0,2),reshape=True)
#cube = interpolation.rotate(cube,angle=45,axes=(1,2),reshape=True)

#cube = resample(cube, (.5,.5,.5))
#cube = compute_ground_truth((49,113,171))
    
###############
#groundTruth
#gt = load('data_boundary.npy')
#rbf = load('rbf3.npy')
##cube = abs(gt-rbf)
#cube = multiply(gt,rbf)
#gt[gt<.6]=0
#gt[gt>=.6]=1
#cube = rbf
################

#print(1-sum(cube)/sum(gt+rbf))

#histGT  = D2(gt, .7)
#histRBF = D2(rbf, .7)
#print(earth_mover_dist(histGT,histRBF))
#print(1-MinkowskiL1_dist(histGT,histRBF))

#cube = ninja_twisted(N)

#cube = logical_and(cube<=.1, cube>=0).astype(uint8)

#cube[cube<=0]=0
#cube[cube>0]=255
#cube = cube.astype(uint8)
#save('s3_rbf_iso.npy', cube)

#cube = load('ref_data.npy')
#cube = load('cached_data_s3_128.npy')
#cube = load('ninja128_worm.npy')
#cube = ninja(128)

#cube = ninja(N)
#cube = xarc(64,(0.5,0.5,0.5),.3,.1)
#cube = shape_s(128)
#cube = letterS(128)
#save('letterS',cube)
#con = morphology.generate_binary_structure(3, 1)
#cube = morphology.binary_closing(cube,structure=con,iterations=3).astype(uint8)
#print('closed')
#cube = sphere(128,(.5,.5,.5),.4)

R0=[[1,0],[0,-1]]
R1=[[0,1],[-1,0]]
#((ndimage.convolve(a,R0)**2 + ndimage.convolve(a,R1)**2)**.5).round().astype(int)
#cube= (ndimage.convolve(a,R0)**2 + ndimage.convolve(a,R1)**2)

#cube = shape_blimps(N,3)

def talus():
    #binvox.exe -d 120 -t raw -e "FJ3385_BP8033_FMA24482_Right talus.obj"
    ## "...
    ## area filled: 86 x 73 x 120
    ## integer bounding box: [0,0,0] - [85,72,119]
    ## ..."
    cube = fromfile('FJ3385_BP8033_FMA24482_Right talus.raw',uint8)
    cube = cube.reshape(120*ones(3))
    flood_fill(cube,array([20,20,20]),0,255)
    c=zeros((86+8,73+8,120+8),uint8)
    c[4:-4,4:-4,4:-4] = cube[:86,:73,:]
    save('FJ3385_BP8033_FMA24482_Right talus_Filled.npy',c)

def L1_vertebra():
    #binvox.exe -d 120 -t raw -e "FJ3157_BP8948_FMA13072_First lumbar vertebra.obj"
    ## "...
    ## area filled: 106 x 85 x 120
    ## integer bounding box: [0,0,0] - [105,84,119]
    ## ..."
    cube = fromfile('FJ3157_BP8948_FMA13072_First lumbar vertebra.raw',uint8)
    cube = cube.reshape(120*ones(3))
    #this point is set by looking at the model
    flood_fill(cube,array([60,20,90]),0,255)
    c=zeros((106+8,85+8,120+8),uint8)
    c[4:-4,4:-4,4:-4] = cube[:106,:85,:]
    save('FJ3157_BP8948_FMA13072_First lumbar vertebra_Filled.npy',c)
    return c

def calcaneus():
    #binvox.exe -d 120 -t raw -e "FJ3157_BP8948_FMA13072_First lumbar vertebra.obj"
    ## "...
    ## area filled: 96 x 85 x 120
    ## integer bounding box: [0,0,0] - [95,84,119]
    ## ..."
    cube = fromfile('FJ3360_BP9040_FMA24497_Right calcaneus.raw',uint8)
    cube = cube.reshape(120*ones(3))
    flood_fill(cube,array([50,50,60]),0,255)
    c=zeros((96+8,85+8,120+8),uint8)
    c[4:-4,4:-4,4:-4] = cube[:96,:85,:]
    save('FJ3360_BP9040_FMA24497_Right calcaneus_Filled.npy',c)
    return c

def hepatic_vein_120():
    #binvox.exe -d 120 -t raw -e "FJ2415_BP5820_FMA14339_Left hepatic vein.obj"
    ## "...
    ##     area filled: 120 x 72 x 103
    ##     integer bounding box: [0,0,0] - [119,71,102]
    ## ..."
    cube = fromfile('FJ2415_BP5820_FMA14339_Left hepatic vein_120.raw',uint8)
    cube = cube.reshape(120*ones(3))
    x,y,z=20,62,90
##    cube[cube==255]=100
##    cube[x-1:x+2,y-1:y+2,z-1:z+2] = 255
    flood_fill(cube,array([x,y,z]),0,255)
    c=zeros((120+8,72+8,103+8),uint8)
    c[4:-4,4:-4,4:-4] = cube[:,:72,:103]
    save('FJ2415_BP5820_FMA14339_Left hepatic vein_120_Filled.npy',c)
    return c

def hepatic_vein_148():
    #binvox.exe -d 148 -t raw -e "FJ2415_BP5820_FMA14339_Left hepatic vein.obj"
    ## "...
    ## area filled: 148 x 88 x 127
    ## integer bounding box: [0,0,0] - [147,87,126]
    ## ..."
    cube = fromfile('FJ2415_BP5820_FMA14339_Left hepatic vein.raw',uint8)
    cube = cube.reshape(148*ones(3))
    x,y,z=20,74,110
##    cube[cube==255]=100
##    cube[x-1:x+2,y-1:y+2,z-1:z+2] = 255
    flood_fill(cube,array([x,y,z]),0,255)
    c=zeros((148+8,88+8,127+8),uint8)
    c[4:-4,4:-4,4:-4] = cube[:,:88,:127]
    save('FJ2415_BP5820_FMA14339_Left hepatic vein_Filled.npy',c)
    return c

def hum_dia_crop(): 
    #from sesame import *
    refmarks = io.loadmat(REF_MARKS_MATFILE).values()[0]
    cube = compute_ground_truth((112L, 105L, 155L))
    #due to some error in enforce_clockwise
    cube[91,:,:] = 1-cube[91,:,:]
    cube = cube.astype(uint8)*255
    save('hum_dia',cube)
    #cube = load('hum_dia.npy')

#cube = load('FJ3360_BP9040_FMA24497_Right calcaneus_Filled.npy')
            
##c = fromfile('FJ2415_BP5820_FMA14339_Left hepatic vein.raw',uint8)
##c = c.reshape(120*ones(3))
##cube=zeros((120+8,72+8,103+8),uint8)
##cube[4:-4,4:-4,4:-4] = c[:,:72,:103]

#cube = load('Case21_segmentation.npy')
#print('cube created')
#from sesame import *
#cube = resample(cube, (.625,3.6,3.6))

#cube = trefoil_knot(128)

#cube = load('hum_dia_crop_pad.npy')
#cube = load('s3_ventricle2.npy')
cube = load('s3_ventricle2_iso_gt.npy')
cube = cube[::2,:,:]
#cube = (cube /2**4) .astype(uint8)
save('s3_ventricle2_gt.npy',cube)

src = mlab.pipeline.scalar_field(cube)
src.spacing = [2.1333, 1, 1]

#con = mlab.contour3d(cube)
#vol = mlab.pipeline.volume(src)
#print('vol created')

iso = mlab.pipeline.iso_surface(src)
iso.actor.property.opacity = .3

##cut_plane = mlab.pipeline.cut_plane(src)    
##surf = mlab.pipeline.surface(cut_plane)
##surf.enable_contours = True

X,Y,Z = array(cube.shape) * src.spacing
outline = mlab.outline(extent=[0,X,0,Y,0,Z],line_width=1)
outline.outline_mode = 'cornered'

print(cube.shape)
print(cube.dtype)
mlab.show()
