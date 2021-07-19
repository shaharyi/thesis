#3-1.img -> 3-001.dcm

import os

print(os.getcwd())
files = os.listdir('.')

for f in files:
    if f.endswith('.img'):
        num_start = f.index('-')+1
        num_end   = f.index('.')
        fnew = f[:num_start] +              \
               '0'*(3-num_end+num_start) +  \
               f[num_start:num_end] +       \
               f[num_end:-3] +              \
               'dcm'
        print(f + ' -> ' + fnew)
        os.rename(f,fnew)
