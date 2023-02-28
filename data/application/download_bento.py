import requests
import os

if not os.path.isdir('bento/'):
    os.makedirs('bento/')

urls = {
    'seqfish.h5ad' : 'https://figshare.com/ndownloader/files/29046873',
    'seqfish+_processed.h5ad' : 'https://figshare.com/ndownloader/files/35596982'
    }

for file in urls.keys():
    r = requests.get(urls[file])
    
    with open('bento/' + file, 'wb') as f:
        f.write(r.content)