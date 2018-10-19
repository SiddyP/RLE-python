import numpy as np

#Run length encoding to mask function
shape=(768,768)
def rle_to_mask(rle, shape=shape):
	rle = rle.split()
	start = np.array(rle[0:][::2],dtype=int)
	length = np.array(rle[1:][::2],dtype=int)
	ends = start + length

	mask = np.zeros(768*768, dtype=np.uint8)
	for start, ends in zip(start,ends):
		mask[start:ends] = 1
	return mask.reshape(shape).T