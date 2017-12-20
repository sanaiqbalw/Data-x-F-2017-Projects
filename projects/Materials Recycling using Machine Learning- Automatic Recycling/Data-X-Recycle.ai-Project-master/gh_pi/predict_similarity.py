'''
	Functions for image similarity
'''

from PIL import Image

def similarity(img1, img2):
    hash1 = avhash(img1)
    hash2 = avhash(img2)
    dist = hamming(hash1, hash2)
    similarity_to_previous = (64-dist)*100/64
    return similarity_to_previous

# for image pmatch
def avhash(im):
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    im = im.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, im.getdata()) / 64.
    return reduce(lambda x, (y, z): x | (z << y),
                  enumerate(map(lambda i: 0 if i < avg else 1, im.getdata())),
                  0)


def hamming(h1, h2):
    h, d = 0, h1 ^ h2
    while d:
        h += 1
        d &= d - 1
    return h
