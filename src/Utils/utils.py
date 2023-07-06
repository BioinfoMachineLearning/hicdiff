import torch
import gc
import matplotlib.pyplot as plt
import pdb
from scipy.stats import pearsonr
import numpy as np
# import cupy as cp

def splitPieces(fn, piece_size, step):
    data   = np.load(fn)
    pieces = []
    bound  = data.shape[0]
    for i in range(0, bound-piece_size+1, step):
        pieces.append(data[i:i+piece_size, i:i+piece_size])
    pieces = np.asarray(pieces)
    pieces = np.expand_dims(pieces,1)
    return pieces

def loadSingleConstraints(stri, res):
    contact_map  = np.loadtxt(stri)
    rows         = (contact_map[:,0]/res).astype(int)
    cols         = (contact_map[:,1]/res).astype(int)
    vals         = contact_map[:,2]
    bigbin       = np.max((rows, cols))
    #smallbin     = np.min((rows, cols))
    mat          = np.zeros((bigbin+1, bigbin+1), dtype='float32')
    coordinates  = list(range(0, bigbin+1))
    for r, c, v in zip(rows, cols, vals):
        if v == "NaN" or v == "nan" or v == np.nan:
            v = 0
        mat[r, c] = v
        mat[c, r] = v
    diag      = np.diag(mat)
    removeidx = np.argwhere(diag==0)[:,0].tolist()
    removeidx.extend(np.argwhere(np.isnan(diag))[:,0])
    for rem in removeidx:
        coordinates.remove(rem)
    mat = np.delete(mat, removeidx, axis=0)
    mat = np.delete(mat, removeidx, axis=1)
    per = np.percentile(mat, 99.9)
    mat = np.clip(mat, 0, per)
    mat = mat/per
    return mat, np.array(coordinates)

def loadBothConstraints(stria, strib, res):
    contact_mapa  = np.loadtxt(stria)
    contact_mapb  = np.loadtxt(strib)
    rowsa         = (contact_mapa[:,0]/res).astype(int)
    colsa         = (contact_mapa[:,1]/res).astype(int)
    valsa         = contact_mapa[:,2]
    rowsb         = (contact_mapb[:,0]/res).astype(int)
    colsb         = (contact_mapb[:,1]/res).astype(int)
    valsb         = contact_mapb[:,2]
    bigbin        = np.max((np.max((rowsa, colsa)), np.max((rowsb, colsb))))
    smallbin      = np.min((np.min((rowsa, colsa)), np.min((rowsb, colsb))))
    mata          = np.zeros((bigbin-smallbin+1, bigbin-smallbin+1), dtype='float32')
    matb          = np.zeros((bigbin-smallbin+1, bigbin-smallbin+1), dtype='float32')
    coordinates   = list(range(smallbin, bigbin))
    i=0
    for ra,ca,ia in zip(rowsa, colsa, valsa):
        i = i+1
        print(str(i)+"/"+str(len(valsa)+len(valsb)))
        mata[ra-smallbin, ca-smallbin] = ia
        mata[ca-smallbin, ra-smallbin] = ia

    for rb,cb,ib in zip(rowsb, colsb, valsb):
        i = i+1
        print(str(i)+"/"+str(len(valsa)+len(valsb)))
        matb[rb-smallbin, cb-smallbin] = ib
        matb[cb-smallbin, rb-smallbin] = ib
    diaga         = np.diag(mata)
    diagb         = np.diag(matb) # np.unique(array) find the unique elements of an array
    removeidx     = np.unique(np.concatenate((np.argwhere(diaga==0)[:,0], np.argwhere(diagb==0)[:,0], np.argwhere(np.isnan(diagb))[:,0])))
    mata = np.delete(mata, removeidx, axis=0)  #  delete the removeidx along the row
    mata = np.delete(mata, removeidx, axis=1)  #  delete 5he renoveidx along the column
    per_a       = np.percentile(mata, 99.9)  #  statistical in Numpy
    mata        = np.clip(mata, 0, per_a)  # mathematical functions. limit the value in an array
    mata        = mata/per_a
    matb = np.delete(matb, removeidx, axis=0) #  similar things above
    matb = np.delete(matb, removeidx, axis=1)
    per_b       = np.percentile(matb, 99.9)
    matb        = np.clip(matb, 0, per_b)
    matb        = matb/per_b
    return mata, matb


def loadConstraintAsMat(stri, res=50000):
	contact_map = np.loadtxt(stri)
	rows = contact_map[:,0]
	cols = contact_map[:,1]
	vals = contact_map[:,2]
	rows = (rows/res).astype(int)
	cols = (cols/res).astype(int)
	mat  = constraints2mats(rows, cols, vals)
	return mat

def constraints2mats(row, col, ifs):
	bigbin   = np.max((row,col))
	smallbin = np.min((row,col))
	mat = np.zeros((bigbin-smallbin+1, bigbin-smallbin+1))
	for r,c,i in zip(row, col, ifs):
		mat[r-smallbin,c-smallbin] = i
		mat[c-smallbin,r-smallbin] = i
	return mat

def splitto40(data, target):
    split_data   = []
    split_target = []
    for i in range(0, data.shape[2]-40,40):
        for j in range(0, target.shape[2]-40, 40):
            split_data.append(data[:,:,i:i+40, j:j+40])
            split_target.append(target[:,:,i:i+40, j:j+40])
    split_data   = torch.cat(split_data, 0)
    split_target = torch.cat(split_target, 0)
    return split_data, split_target
