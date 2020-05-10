# -*- coding: utf-8 -*-
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
from osgeo import gdal

#---importing sentinel-2 jpeg2000 data to python---
    #im_name,im_arr,im_geo,im_proj = import_data([5,6],"C:/Users/asus/Desktop/data_19042018/IMG_DATA")
    #im_name,im_arr,im_geo,im_proj = import_data([5,6],"C:/Users/asus/Desktop/data_16092018/IMG_DATA")
    #im_name,im_arr,im_geo,im_proj = import_data([4,8],"C:/Users/asus/Desktop/data_16092018/IMG_DATA")

def import_data(band,img_path=None):
    #band ==> band number of sentinel-2 data which which user want to import
    #img_path ==> path where images are located
    if img_path is None:
        img_path = ''
    #tem_pat = img_path + "/IMG_DATA/"
    bands = '['
    for i in band:
        bands += str(i)
    bands += ']'
    tem_pat = img_path + "/"
    temm = [tem_pat+f for f in os.listdir(tem_pat) if re.search(r'[a-zA-Z0-9]*{}\.jp2$'.format(bands),f)]
    image_values=[]
    im_geo,im_proj = [],[]
    for j in range(len(temm)):
        im = gdal.Open(temm[j],gdal.GA_ReadOnly)
        im_array = np.array(im.GetRasterBand(1).ReadAsArray())
        im_geo.append(im.GetGeoTransform())
        im_proj.append(im.GetProjection())
        image_values.append(im_array)
    #temm ==> a list, keep images path and images name
    #image_values ==> 3D list, keep the image bands seperately
    #im_geo ==> images geometry
    #im_proj ==> images projection
    return temm,image_values,im_geo,im_proj

#--cropping specific area on the data---
    #cropped_im,geo = crop_box(im_arr,2048,[3441,3441],im_geo)
def crop_box(image_arr,size,left_corner,im_geo):
    #image_arr ==> numpy arrays includes imported bands of the data
    #size ==> pixel size of specific box that user want to achieve
    #left_corner ==> pixel coordinates of the specific box left corner
    #im_geo ==> image geometry
    x = left_corner[1]
    y = left_corner[0]
    cropped= np.zeros(shape=(len(image_arr),size,size),dtype='uint16')
    for j in range(len(image_arr)):
        cropped[j]=image_arr[j][x:x+size,y:y+size]
        y_geo = x*im_geo[j][1]
        x_geo = y*im_geo[j][5]
        im_geo[j] = (im_geo[j][0]+y_geo,im_geo[j][1],im_geo[j][2],im_geo[j][3]+x_geo,im_geo[j][4],im_geo[j][5])
    #cropped ==> cropped images as list
    #im_geo ==> images geometry of the cropped images
    return cropped,im_geo #return cropped images list 'cropped'


#---showing occurrence of each pixel in one band---
def number_of_pixels(im):
    # im ==> a band, numpy list
    un, coun = np.unique(im,return_counts=True)
    print 'pixel values','--->','number_of_pixel'
    for i in range(len(un)):
        print un[i],'--->',coun[i]


#---removing the saturations---
        #rem_im =remove_sat(cropped_im,[239,239],[6000,6000])
def remove_sat(im,min_thres,max_thres):
    #im ==> 3D list that have bands
    #min_thres ==> a list have minimum threshold for each band
    #max_thres ==> a list have maximum threshold for each band
    new_img= np.zeros(shape=im.shape,dtype='float')
    for i in range(len(new_img)):
        for j in range(len(im[0])):
            for k in range(len(im[0][0])):
                if im[i][j][k]<min_thres[i]:
                    new_img[i][j][k] = min_thres[i]
                elif im[i][j][k]>max_thres[i]:
                    new_img[i][j][k] = max_thres[i]
                else :
                    new_img[i][j][k] = im[i][j][k]
    #new_img ==> images that saturations are removed
    return new_img


#---creating rendvi---
    #rendvi= rendvi(rem_im[0],rem_im[1])
def rendvi(b1,b2):
    #b1 ==> numpy list that has red edge band of the sentinel-2 (wavelength = 0.705 μm)
    #b2 ==> numpy list that has red edge band of the sentinel-2 (wavelength = 0.740 μm)
    rendvi = (b2-b1)/(b1+b2)
    #rendvi ==> rendvi band 
    return rendvi


#---creating ndvi---
    #ndvi = ndvi(rem_im[0],rem_im[1])
def ndvi(red,nir):
    #red ==> red band
    #nir ==> near infrared band
    nd = (nir-red)/(nir+red)
    #nd ==> ndvi band
    return nd

#---creating mirror for boundaries---
    #mir_im = mirror_for_boundaries([rendvi])
def mirror_for_boundaries(im):
        #im ==> numpy list, bands
	new_img= np.array([[[0 for l in xrange(len(im[0][0])+2)]for f in xrange(len(im[0])+2)]for l in xrange(len(im))],dtype='float')
	sz = len(im[0])
	for i in range(len(im)):
		new_img[i][1:sz+1,1:sz+1] = im[i]
		new_img[i][0,1:sz+1] = new_img[i][1,1:sz+1]
		new_img[i][sz+1,1:sz+1] = new_img[i][sz,1:sz+1]
		new_img[i][0:sz+2,0] = new_img[i][0:sz+2,1]
		new_img[i][0:sz+2,sz+1] = new_img[i][0:sz+2,sz]
        #new_img ==> bands with mirror edge
	return new_img

#---creating vector for each pixel using 3x3 pixel box---
    #vec = vectorize(mir_im,1,0)
def vectorize(im,band_num=None,band=None):
        #im ==> numpy array that has images band to vectorize
        #band_num ==> band number that user want to include to vector for each pixel
        #band ==> if user want to vectorize only one band index of the band in the 'im' array
	if band_num==1:
		size = (len(im[0])-2)*(len(im[0][0])-2)
		new_img= np.zeros(shape = (size,9),dtype='float')
		tem = 0
		for i in range(1,len(im[0])-1):
			for j in range(1,len(im[0][0])-1):
				new_img[tem]=im[band][i-1:i+2,j-1:j+2].reshape(9)
				tem+=1
	else:
		size = (len(im[0])-2)*(len(im[0][0])-2)
		new_img= np.zeros([[0 for i in xrange(9*band_num)]for f in xrange(size)],dtype='float')
		tem_arr = np.zeros([[0 for k in xrange(9)]for l in xrange(band_num)],dtype='float')
		tem = 0
		for i in range(1,len(im[0])-1):
			for j in range(1,len(im[0][0])-1):
				for k in range(band_num):
					tem_arr[k] = im[k][i-1:i+2,j-1:j+2].reshape(9)
				tem_array = tem_arr.reshape(band_num*9)
				new_img[tem]=tem_array
				tem+=1
        #new_img ==> a 2D list that has vectors for each pixels
	return new_img


#---apply K-means classification to vectors---
    #clus,cen = clusters([vec],30)
    #clus[0].astype('uint16').tofile('test_1.bsq')
    #t = np.fromfile('test_1.bsq',dtype = 'uint16').reshape(2048,2048)  
def clusters(vec,num_clus):
        #vec ==> vector list
        #num_clus ==> number of cluster that user can define
	clus = []
	centers = []
	sz = int(sqrt(len(vec[0])))
	for i in range(len(vec)):
		k = KMeans(n_clusters=num_clus, random_state=0).fit(vec[i])
		clus.append(k.labels_.reshape(sz,sz))
		centers.append(k.cluster_centers_)
	#plt.imshow(clus);plt.show()
	#clus ==> a numpy list that has class id for each pixel
	#centers ==> clusters center
	return clus,centers


#---tiling the image to the pacthes---
    #til = get_tiles(t,32)
def get_tiles(clus,size):
    #clus ==> clustered data
    #size ==> size of the patches that user can define
    num = len(clus)/size
    tiles = np.zeros(shape = (num*num,size,size),dtype='int32')
    in_tile = 0
    for i in range(num):
        for j in range(num):
            a = i*size
            b = j*size
            tiles[in_tile]=clus[a:a+size,b:b+size]
            in_tile+=1
    #tiles = np.array([[[0 for i in xrange(size)]for j in xrange(size)]for k in xrange(num*num)],dtype='int32')
    #tiles ==> a list has tiles which has specific pixel sizes
    return tiles
    
#---creating Bag of Visual Words Model---
    #bovw = BoVW(til,30)
def BoVW(tiles,nclus):
    #tiles ==> a list has tiles
    #nclus ==> number of clusters that user defined when was appliying K-means algorithm
    tem  = 0
    his=np.zeros(shape=(len(tiles),nclus),dtype='uint16')
    for i in range(len(tiles)):
        un, coun = np.unique(tiles[i],return_counts=True)
        for j in range(nclus):
            if j in un:
                his[i][j]=coun[tem]
                tem+=1
        else:
            his[i][j]=0
        tem=0
    #his ==> a list that has pixel occurance number for each patches
    return his



#---applying Latent Dirichlet Allocation to BoVW model---
    #com,exp_com,doc_top,dtop = lda([bovw],[10])
    #com[0].tofile('bot_com.bsq') ; comm_2 = np.fromfile('bot_com.bsq').reshape(10,30);
    #exp_com[0].tofile('bot_exp_com.bsq') ; exp = np.fromfile('bot_exp_com.bsq').reshape(10,30);
    #doc_top[0].tofile('bot_doc_top.bsq') ; docc = np.fromfile('bot_doc_top.bsq').reshape(4096,10);
    #dtop[0].astype('float').tofile('bot_dtop.bsq') ; dd = np.fromfile('bot_dtop.bsq').reshape(64,64);
def lda(his,topics):
        #his ==> a list that has pixel occurance number for each patches
        #topics ==> topics number that user can define
	sz = int(sqrt(len(his[0])))
	com,exp_com,doc_top,dtop = [],[],[],[]
	for j in range(len(his)):
		ld_1 = LatentDirichletAllocation(n_components = topics[j]).fit(his[j])#,doc_topic_prior = alpha[j]
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
	#com ==> components of topics and words
	#exp_com ==> exponential probability distribution between words and topics
	#doc_top ==> probability distribution between documents and topics
	#dtop ==> distribution of the topics for each patches
	return com,exp_com,doc_top,dtop

#---selecting higher probability in prob. distribution between topics and words---
    #w_t = word_to_top(exp_com,30,10)
def word_to_top(exp,nword,ntop):
        #exp ==> exponential probability distribution between words and topics
        #nword ==> number of the word that defined by user when was applying the K-mean clasification
        #ntop ==> number of the topics that defined by user when was applying Latent Dirichlet Allocation 
	w_t = []
	for i in range(nword):
		k, = np.where(exp[0][0:ntop,i]==exp[0][0:ntop,i].max())
		w_t.append(k[0])
	#w_t ==> a list that each word assigned to topics which has most probablity
	return w_t


#---assigning topic to each pixel on the clustered image---
    #bot = BoT([t],w_t)
    #bot.tofile('bot.bsq')
    #bot_2 = np.fromfile('bot.bsq',dtype = 'uint16').reshape(2048,2048)
def BoT(clus,w_t):
	bot=np.zeros(shape = (len(clus[0][0]),len(clus[0])) ,dtype='uint16')
	for i in range(len(clus[0])):
		for j in range(len(clus[0][0])):
			ind = clus[0][i][j]
			bot[i][j] = w_t[ind]
	#bot ==> a band after classified usin BOT model
	return bot


#---defining specific class and discrimination of it---
    #merge_clus(bot,[[0,1,3,4,5,6,7,8,9],[2]],10)
    #merge_clus(bot,[[0,1,2,3,6,8,9],[4,5,7]],10)
def merge_clus(bot,ls,ntop):
        #bot ==> a band after classified using BOT model
        #ls ==> 2d list that has assigned clusters id
        #ntop ==> topics number that defined by user when was applying Latent Dirichlet Allocation
	t = bot.reshape(len(bot)**2)
	for i in range(len(ls)):
		for j in range(len(ls[i])):
			t[t==ls[i][j]]=ntop
		ntop+=1
	tet = t.reshape(int(sqrt(len(t))),int(sqrt(len(t))))
	#tet ==> a discriminated imaged
	return tet
    
#---save file as ENVI binary format---
    #save_as_envi('out_bot.bsq',bot,geo[0],im_proj[0])
def save_as_envi(path,np_arr,im_geo,im_proj):
    #path ==> save directory
    #np_arr ==> numpy array that user want to save
    #im_geo ==> image geometry
    #im_proj ==> image projection
    driver = gdal.GetDriverByName('ENVI')
    col,row = np_arr.shape
    arr = np_arr.astype('uint16')
    destination = driver.Create(path, col, row, 1, gdal.GDT_UInt16)
    destination.SetGeoTransform(im_geo)
    destination.SetProjection(im_proj)
    rb = destination.GetRasterBand(1)
    rb.WriteArray(arr)
    destination = None
    rb = None

#---change detection division method
def change_div(bot,bot_2):
    chan = np.zeros(shape=(bot.shape),dtype='uint16')
    for i in range(len(bot)):
        for j in range(len(bot[i])):
            if bot[i][j] == 0 and bot_2[i][j]==0:
                a = 1
            else:
                a = bot[i][j]/bot_2[i][j]
            chan[i][j] = a
    return chan




    
    
