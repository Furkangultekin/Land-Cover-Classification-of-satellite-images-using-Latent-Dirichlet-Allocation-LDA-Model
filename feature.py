from skimage import io
from scipy import ndimage
import scipy.ndimage
import sys
import os
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from osgeo import gdal
import numpy as np
from math import sqrt
from sklearn.decomposition import LatentDirichletAllocation

def load_all_data(img_path=None):
        if img_path is None:
                img_path = ''
        tem_pat = img_path + "binary/"
	temm = [tem_pat+f for f in os.listdir(tem_pat) if re.search(r'[a-zA-Z0-9]*[4]\.bsq$',f)]
	image_values =[]
	for j in range(len(temm)):
                #im = gdal.Open(filename)
                im = np.fromfile(temm[j],dtype = 'uint16').reshape(10,1024,1024)
                #arr = im.ReadAsArray()
                image_values.append(im)
        return temm,image_values

def load_data():
	im = np.fromfile('20170626.bsq',dtype='uint16').reshape(10,1024,1024)
	return im

def load_clus_data(nband,nclus):
	bands=['blue','red','green','four','three']
	num = [1,1,1,4,3]
	im=[]
	for i in range(len(nband)):
		#inde = bands.index(nband[i])
		im.append(np.fromfile('{}_b_{}_clus.bsq'.format(nband[i],nclus[i]),dtype='int32').reshape(1024,1024))
	#centers = np.fromfile('{}_b_{}_cent.bsq'.format(nband,nclus)).reshape(nclus,num[inde]*9)
        return im #centers


def load_tci():
	im = np.fromfile('tci_0626.bsq',dtype = 'uint8').reshape(1024,1024,3)
	return im

def visulize_all(cropp,columns,rows):
        for i in range(len(cropp)):
                fig=plt.figure(figsize=(8, 8))
                for j in range(1,columns*rows +1):
                        fig.add_subplot(rows, columns, j)
                        k = ((cropp[i][j-1].astype('float')/cropp[i][j-1].max())*65536).astype('uint16')
                        plt.imshow(k,cmap='gray')
        plt.show()

def visualise_1(im,columns,rows,color=None):
	fig=plt.figure(figsize=(8,8))
        for i in range(1,columns*rows +1):
                fig.add_subplot(rows,columns,i)
                plt.imshow(im[i-1],cmap=color)
        plt.show()

def visulize_all_hist(cropp,columns,rows):
        for i in range(len(cropp)):
                fig=plt.figure(figsize=(8, 8))
                for j in range(1,columns*rows +1):
                        fig.add_subplot(rows, columns, j)
                        plt.hist(cropp[i][j-1])
        plt.show()

def histog(im,columns,rows):
	fig=plt.figure(figsize=(8,8))
	for i in range(1,columns*rows +1):
		fig.add_subplot(rows,columns,i)
		plt.hist(im[i-1][0],rwidth=5000)
	plt.show()

def number_of_pixels(im):
	un, coun = np.unique(im,return_counts=True)
	print 'pixel values','--->','number_of_pixel'
	for i in range(len(un)):
		print un[i],'--->',coun[i]

def select_band(im,bands):
	new_img= np.array([[[0 for l in xrange(len(im[0][0]))]for f in xrange(len(im[0]))]for l in xrange(len(bands))],dtype='uint16')
	for i in range(len(bands)):
		new_img[i]=im[bands[i]]
	return new_img

def select_im_band(im,img,bands):
        new_img= np.array([[[0 for l in xrange(len(im[0][0][0]))]for f in xrange(len(im[0][0]))]for l in xrange(len(bands[0]))],dtype='uint16')
        selec = []
	for j in range(len(img)):
		new_img= np.array([[[0 for l in xrange(len(im[0][0][0]))]for f in xrange(len(im[0][0]))]for l in xrange(len(bands[j]))],dtype='uint16')
		for i in range(len(bands[j])):
                	new_img[i]=im[j][bands[j][i]]
		selec.append(new_img)
        return selec

def remove_sat(im,min_thres,max_thres):
	new_img= np.array([[[0 for l in xrange(len(im[0][0]))]for f in xrange(len(im[0]))]for l in xrange(len(im))],dtype='float')
	for i in range(len(new_img)):
		for j in range(len(im[0])):
			for k in range(len(im[0][0])):
				if im[i][j][k]<min_thres[i]:
					new_img[i][j][k] = min_thres[i]
				elif im[i][j][k]>max_thres[i]:
					new_img[i][j][k] = max_thres[i]
				else :
					new_img[i][j][k] = im[i][j][k]
	return new_img

def ndvi(red,nir):
	nd = (nir-red)/(nir+red)
	return nd

def ndvi_selected()

def mirror_for_boundaries(im):
	new_img= np.array([[[0 for l in xrange(len(im[0][0])+2)]for f in xrange(len(im[0])+2)]for l in xrange(len(im))],dtype='float')
	sz = len(im[0])
	for i in range(len(im)):
		new_img[i][1:sz+1,1:sz+1] = im[i]
		new_img[i][0,1:sz+1] = new_img[i][1,1:sz+1]
		new_img[i][sz+1,1:sz+1] = new_img[i][sz,1:sz+1]
		new_img[i][0:sz+2,0] = new_img[i][0:sz+2,1]
		new_img[i][0:sz+2,sz+1] = new_img[i][0:sz+2,sz]
	return new_img

def normalize(g):
	h=[]
	for i in range(len(g)):
		tem = g[i].astype('float')
		tt = (tem-tem.min())/(tem.max()-tem.min())
		h.append(tt)
	return h

def vectorize(im,band_num=None,band=None):
	if band_num==1:
		size = (len(im[0])-2)*(len(im[0][0])-2)
		new_img= np.array([[0 for i in xrange(9)]for f in xrange(size)],dtype='float')
		tem = 0
		for i in range(1,len(im[0])-1):
			for j in range(1,len(im[0][0])-1):
				new_img[tem]=im[band][i-1:i+2,j-1:j+2].reshape(9)
				tem+=1
	else:
		size = (len(im[0])-2)*(len(im[0][0])-2)
		new_img= np.array([[0 for i in xrange(9*band_num)]for f in xrange(size)],dtype='float')
		tem_arr = np.array([[0 for k in xrange(9)]for l in xrange(band_num)],dtype='float')
		tem = 0
		for i in range(1,len(im[0])-1):
			for j in range(1,len(im[0][0])-1):
				for k in range(band_num):
					tem_arr[k] = im[k][i-1:i+2,j-1:j+2].reshape(9)
				tem_array = tem_arr.reshape(band_num*9)
				new_img[tem]=tem_array
				tem+=1
	return new_img

def clusters(vec,num_clus):
	clus = []
	centers = []
	sz = sqrt(len(vec[0]))
	for i in range(len(vec)):
		k = KMeans(n_clusters=num_clus, random_state=0).fit(vec[i])
		clus.append(k.labels_.reshape(sz,sz))
		centers.append(k.cluster_centers_)
	#plt.imshow(clus);plt.show()
	return clus,centers

def merge_color(clus):
	a = []
	d = np.array([[0 for i in xrange(len(clus[0][0]))]for j in xrange(len(clus[0]))],dtype = 'int32')
	t = []
	for k in range(len(clus)):
		for i in range(len(clus[k])):
			for j in range(len(clus[k][i])):
        			if clus[k][i][j] not in a :
                			a.append(clus[k][i][j])
                			z = a.index(clus[k][i][j])
                			d[i][j] = z
       				else:
                			z = a.ind+ex(clus[k][i][j])
                			d[i][j]=z
		a=[]
		t.append(d)
		d = np.array([[0 for i in xrange(len(clus[0][0]))]for j in xrange(len(clus[0]))],dtype = 'int32')
	return t

def get_tiles(clus,size):
	num = len(clus[0])/size
        tiles = np.array([[[0 for i in xrange(size)]for j in xrange(size)]for k in xrange(num*num)],dtype='int32')
	in_tile = 0
	til = []
	for k in range(len(clus)):
		for i in range(num):
			for j in range(num):
				a = i*size
				b = j*size
				tiles[in_tile]=clus[k][a:a+size,b:b+size]
				in_tile+=1
		til.append(tiles)
                in_tile=0
		num = len(clus[k])/size
                tiles = np.array([[[0 for i in xrange(size)]for j in xrange(size)]for k in xrange(num*num)],dtype='int32')
	return til



def BoVW(tiles,nclusters,size):
	hists = []
	ran = []
	for i in range(nclusters):
		ran.append(i)
	for i in range(len(tiles)):
		tem = plt.hist(tiles[i].reshape(size*size),bins=range(nclusters+1))[0]
		hists.append(tem)
	return hists,ran #plt.bar(rang,hists[i])

def BoVW(tiles,nclus):
	tem  = 0
	all_his = []
	for k in range(len(tiles)):
		his=np.array([[0 for i in xrange(nclus[k])]for j in xrange(len(tiles[k]))],dtype='uint32')
		for i in range(len(tiles[k])):
			un, coun = np.unique(tiles[k][i],return_counts=True)
			for j in range(nclus[k]):
				if j in un:
					his[i][j]=coun[tem]
					tem+=1
				else:
					his[i][j]=0
			tem = 0
		all_his.append(his)
	return all_his

def lda(his,topics,alpha):
	sz = sqrt(len(his[0]))
	com,exp_com,doc_top,dtop = [],[],[],[]
	for j in range(len(his)):
		ld_1 = LatentDirichletAllocation(n_topics = topics[j],doc_topic_prior = alpha[j]).fit(his[j])
		com.append(ld_1.components_)
		exp_com.append(ld_1.exp_dirichlet_component_)
		doc_top.append(ld_1.transform(his[j]))
		d_t = np.array([0 for k in xrange(len(his[j]))])
		for i in range(len(his[j])):
			k, = np.where(doc_top[j][i]==doc_top[j][i].max())
			d_t[i] = k[0]
			k=None
		dtop.append(d_t.reshape(sz,sz))
		ld_1 = None
	return com,exp_com,doc_top,dtop

def word_to_top(exp,nword,ntop):
	w_t = []
	for i in range(nword):
		k, = np.where(exp[0][0:ntop,i]==exp[0][0:ntop,i].max())
		w_t.append(k[0])
	return w_t

def BoT(clus,w_t):
	bot=np.array([[0 for i in xrange(len(clus[0][0]))]for j in xrange(len(clus[0]))],dtype='uint32')
	for i in range(len(clus[0])):
		for j in range(len(clus[0][0])):
			ind = clus[0][i][j]
			bot[i][j] = w_t[ind]
	return bot

def merge_clus(bot,ls,ntop):
	t = bot.reshape(len(bot)**2)
	for i in range(len(ls)):
		for j in range(len(ls[i])):
			t[t==ls[i][j]]=ntop
		ntop+=1
	tet = t.reshape(int(sqrt(len(t))),int(sqrt(len(t))))
	return tet
