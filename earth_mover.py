def earth_mover_dist(hist1,hist2):
    x=0;
    for i in hist1.keys():
        x = x + abs(hist2[i]-hist1[i]);
        if hist2[i] > hist1[i]:
            hist2[i+1] = hist2[i+1]+(hist2[i]-hist1[i]);
            hist2[i] = hist1[i];
        else:
            hist1[i+1] = hist1[i+1]+(hist1[i]-hist2[i]);
            hist1[i] = hist2[i];
    return x

def histogram(px, (x,y),(w,h)):    
    d={}
    for i in range(x,x+w):
        for j in range(y,y+h):
            c=px[i,j]
            d[c] = d.get(c,0) + 1
    return d
