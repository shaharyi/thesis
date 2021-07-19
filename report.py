h = 'metric_0,metric_1,metric_2,metric_3,max avg,' + \
    'max 95th percentile,dice coeff,relative vol diff' + '\n'

import os
f=os.listdir('.')[0]
if not f.endswith('.log'):
    exit_program()
with open(f,'rt') as inp, open('dist.csv','wt') as out:
    out.write(h)
    for line in inp:
        if line=='* distances:\n':
            d=''
            for i in range(8):
                line = inp.next()
                d += line.split()[-1] 
                if i<7: d+=','
            d+='\n'
            out.write (d)
