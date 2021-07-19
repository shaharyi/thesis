#adapted from
#http://en.wikipedia.org/wiki/Sobel_operator#Extension_to_other_dimensions

#smoothing operator
vs = [ 1,  4,  6,  4,  1]
#derivative operator
vd = [-1, -4, -5,  0,  5,  4,  1]

def Hx(x,y,z):
    return vd[x]*vs[y]*vs[z]
def Hy(x,y,z):
    return vd[y]*vs[x]*vs[z]
def Hz(x,y,z):
    return vd[z]*vs[x]*vs[y]

def GetZ():
    f=[]
    for z in range(len(vd)):
        f.append([])
        for y in range(len(vs)):
            f[z].append([])
            for x in range(len(vs)):
                f[z][y].append(Hz(x,y,z))
    return f
        
def GetX():
    f=[]
    for z in range(len(vs)):
        f.append([])
        for y in range(len(vs)):
            f[z].append([])
            for x in range(len(vd)):
                f[z][y].append(Hx(x,y,z))
    return f

def GetY():
    f=[]
    for z in range(len(vs)):
        f.append([])
        for y in range(len(vd)):
            f[z].append([])
            for x in range(len(vs)):
                f[z][y].append(Hy(x,y,z))
    return f

if __name__ == '__main__':
    from pprint import pprint
    pprint(GetY())
