import pdb
from itertools import count, izip 
from collections import defaultdict
from pprint import pprint

import pygame

##############
#
#http://en.wikipedia.org/wiki/Union_find
#
class Node:
    """label is a numeric value
    no make_set() since it's included here
    """
    def __init__ (self, label):
        self.label = label
        self.parent = self
        self.rank   = 0
    def __str__(self):
        return self.label

def union(x, y):
    xRoot = find(x)
    yRoot = find(y)
    if xRoot.rank > yRoot.rank:
        yRoot.parent = xRoot
    elif xRoot.rank < yRoot.rank:
        xRoot.parent = yRoot
    elif xRoot != yRoot: # Unless x and y are already in same set, merge them
        yRoot.parent = xRoot
        xRoot.rank = xRoot.rank + 1
    return x.parent

def find(x):
    if x.parent == x:
        return x
    else:
        x.parent = find(x.parent)
        return x.parent
#    
##############

def blob_extract(img):
     
    """en.wikipedia.org/wiki/Blob_extraction
    input:  Surface with "1" for foreground, "0" for background
    output: Surface with labels where no background
    """
    linked = {}
    M,N = img.get_size()

    labels_img = img.copy()
    labels_img = labels_img.convert(8)
    labels_img.fill(0)
    labels_px = pygame.PixelArray(labels_img)
    
    labels = {}
    next_label = 1
    
    #First pass
    px = pygame.PixelArray(img)
    for row in range(M):
        for col in range(N):
            pos = row,col
            if px[pos] != 0:
                #connected elements with the current element's value
                neighbors = {}
                if row > 0:
                    north = row-1, col
                    if px[north] == px[pos]:
                        neighbors[north] = px[north]
                if col > 0:
                    west = row, col-1
                    if px[west] == px[pos]:
                        neighbors[west] = px[west]
                if neighbors == {}:
                    new_node = Node(next_label)
                    linked[next_label] = new_node
                    labels[pos] = new_node
                    next_label += 1
                else:
                    #Find the smallest label                   
                    L = [labels[x] for x in neighbors.keys()]
                    seq = [x.label for x in L]
                    minvalue, minindex = min(izip(seq, count())) 
                    labels[pos] = L[minindex]
                    for labelx in L:
                        for labely in L:
                            u = union(linked[labelx.label], labely)
                        linked[labelx.label] = u
    #Second pass                        
    areas = defaultdict(int)
    for row in range(M):
        for col in range(N):
            pos = row,col
            if px[pos] != 0:
                labels[pos] = find(labels[pos])
                areas[labels[pos].label] += 1
                labels_px[pos] = 50*labels[pos].label
    del labels_px
    del px
    return labels_img, areas

def test():
    img = pygame.image.load('2blobs.jpg')
    screen = pygame.display.set_mode((100,100))
    img = img.convert(8)
    screen.blit(img,(0,0))
    pygame.display.flip()
    blobs,areas = blob_extract(img)
    pprint(areas)
    screen.blit(blobs,(0,50))
    pygame.display.flip()
    pygame.event.clear()
    pygame.event.wait()
    pygame.quit()
    
if __name__=='__main__':
    test()
