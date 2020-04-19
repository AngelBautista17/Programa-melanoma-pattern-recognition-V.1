#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:02:28 2017

proyecto de grado

@author: franciscorealescastro
"""
import cv2
import numpy as np
import itertools
from scipy import ndimage
from skimage import morphology 
from time import time

def roundOdd(n):
    answer = round(n)
    if  answer%2:
        return answer
    if abs(answer+1-n) < abs(answer-1-n):
        return answer + 1
    else:
        return answer - 1

def getsamples(img):
    x, y, z = img.shape
    samples = np.empty([x * y, z])
    index = 0
    for i in range(x):
        for j in range(y):
            samples[index] = img[i, j]
            index += 1
    return samples


def EMSegmentation(img, no_of_clusters=2):
    output = img.copy()
    samples = getsamples(img)
    em = cv2.ml.EM_create()
    em.setClustersNumber(no_of_clusters)
    em.trainEM(samples)
    means = em.getMeans()
    covs = em.getCovs()
    media1=means[0,:]
    media2=means[1,:]
    Mvar1=covs[0]
    Mvar2=covs[1]
    k1=1/np.sqrt(np.linalg.det(Mvar1))
    k2=1/np.sqrt(np.linalg.det(Mvar2))
    inv1=np.linalg.inv(covs[0])
    inv2=np.linalg.inv(covs[1])
    imgd=np.double(img)
    x, y, z = img.shape
    for i in range(x):
        for j in range(y):
            c=imgd[i,j,:]
            p1=k1*np.exp(-0.5*(c-media1).dot(inv1.dot(c-media1)))
            p2=k2*np.exp(-0.5*(c-media2).dot(inv2.dot(c-media2)))
            maximo=np.max([p1,p2])
            if maximo==p1:
               imgd[i,j,:]=media1
            elif maximo==p2:
                imgd[i,j,:]=media2
    output=np.uint8(imgd) 
    return output
def datosInterp(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#se convierte a HSV
    V=hsv[:,:,2]#se retiene el canal V
    [fil,col,c]=img.shape
    mascara=np.zeros((fil,col))-1
    mascara[0:np.uint64(np.round(0.31*fil)),0:np.uint64(np.round(0.28*col))]=1
    mascara[0:np.uint64(np.round(0.31*fil)),np.uint64(np.round(0.72*col)):col]=1
    mascara[np.uint64(np.round(0.69*fil)):fil,0:np.uint64(np.round(0.28*col))]=1
    mascara[np.uint64(np.round(0.69*fil)):fil,np.uint64(np.round(0.72*col)):col]=1
    fraccionesV=mascara*V
    x=np.zeros(np.sum(mascara==1))
    y=np.zeros(np.sum(mascara==1))
    z=np.zeros(np.sum(mascara==1))
    c=0
    for i in range(0,fil):
        for j in range(0,col):
            if fraccionesV[i,j]>0:
               x[c]=j
               y[c]=i
               z[c]=fraccionesV[i,j]
               c=c+1
    return x,y,z
def obtenerInterp(x,y,z,img):

    #z=np.random.random(numdata)
    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)

    # Evaluate it on a grid...
    [ny,nx,c] = img.shape
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), 
                         np.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)
    return xx,yy,zz

def polyfit2d(x, y, z, order=2):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j    
    return z 
def elimSombras(img):
    x,y,z=datosInterp(img)
    xx,yy,zz=obtenerInterp(x,y,z,img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#se convierte a HSV
    Vorig=hsv[:,:,2]#se retiene el canal V
    Vproc=Vorig/zz
    Uorig=np.mean(Vorig)
    Uproc=np.mean(Vproc)
    Vnew=Vproc*(Uorig/Uproc)
    hsv[:,:,2]=Vnew
    imgProc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return imgProc
def segmentarLunar(img):
    imgProc=elimSombras(img)
    output= EMSegmentation(imgProc,2)
    mascara=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
    mascara=np.uint8((mascara==mascara.min())*255)
    output = cv2.connectedComponentsWithStats(mascara, 4, cv2.CV_32S)
    labels = output[1]
    stats = output[2]
    filtrarAreaTodo=(stats[:,0]*stats[:,1])>0
    Areas=stats[:,4]*filtrarAreaTodo
    Amax=np.max(Areas)
    objAmax=np.argmax(Areas)
    mascaraHuecos=(labels==objAmax)*255
    mascara=ndimage.binary_fill_holes(mascaraHuecos).astype(int)
    mascara=np.uint8(255*mascara)
    mascaraDil=mascara
    return mascaraDil,Amax

def obtenerMask(mascara,Amax):
    sqA=np.sqrt(Amax)
    r=np.round(0.0266*sqA)
    kernel=morphology.disk(r)
    mascaraDil= cv2.dilate(mascara, kernel, iterations=1)
    return mascaraDil
#caracteristicas de asimetria (todas reciben la mascara)
#     Ap area de la lesion
def AreaLesion(mascara):
    mascara=np.double(mascara)/255
    Ap=np.sum(mascara)
    return Ap
#     Pp perimetro de la lesion     
def PerLesion(mascara):
    mascara1=mascara.copy()
    im2, contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    Pp = cv2.arcLength(cnt,True)
    return Pp
#     Ac Area del convexHull y Pc perimetro del convexHull
def CaracConvexHull(mascara):
    mascara1=mascara.copy()
    im2, contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    hull = cv2.convexHull(cnt)
    puntosConvex=hull[:,0,:]
    m,n=mascara1.shape
    ar=np.zeros((m,n))
    mascaraConvex=cv2.fillConvexPoly(ar, puntosConvex, 1)
    Ac=np.sum(mascaraConvex)
    mascaraConvex1=np.uint8(mascaraConvex.copy())
    imC, contoursC, hierarchyC = cv2.findContours(mascaraConvex1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cntC = contoursC[0]
    Pc = cv2.arcLength(cntC,True)
    return Ac,Pc
#   Ab  Area del bounding box y Pb perimetro del bounding box y W/h la tasa de aspecto 
def CaracBoundBox(mascara):
    mascara1=mascara.copy()
    im2, contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    m,n=mascara1.shape
    ar=np.zeros((m,n))
    mascaraRect=cv2.fillConvexPoly(ar, box, 1)
    Ab=np.sum(mascaraRect)
    mascaraRect1=np.uint8(mascaraRect.copy())
    imR, contoursR, hierarchyR = cv2.findContours(mascaraRect1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cntR = contoursR[0]
    Pb = cv2.arcLength(cntR,True)
    x,y,w,h = cv2.boundingRect(cntR)
    tasaAspecto= float(w)/float(h) if w>h else float(h)/float(w)
    return Ab,Pb,tasaAspecto
#   Ae area de la elipse Pe perimetro de la elipse MA eje mayor y ma eje menor
def CaracElipse(mascara):
    mascara1=mascara.copy()
    im2, contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
    Ae=np.pi*MA*ma/4#area de la elipse
    Pe=np.pi*np.sqrt((MA**2+ma**2)/2)#perimetro elipse
    return Ae,Pe,MA,ma

def fraccionar(mascara,n): 
    vx,vy,x,y=orientacion(mascara)
    maskFrac=fracAngulo(mascara,vx,vy,x,y,n)
    return maskFrac

def fracAsim(mascara):
    a=fraccionar(mascara,4)
    b=[]
    for i in a:
        b.append(mascara*i)
    return b

def CaracAsimetria(mascara):
    Ap=AreaLesion(mascara)
    Pp=PerLesion(mascara)
    Ac,Pc=CaracConvexHull(mascara)
    Ab,Pb,tasaAspecto=CaracBoundBox(mascara)
    Ae,Pe,MA,ma=CaracElipse(mascara)
    A1=Ap/Ab #area lesion/area bounding box
    A2=Ac/Ab #area convex hull/area bounding box
    A3=Ap/Ac #area de la lesion/area del convex hull
    A4=np.sqrt(4*Ap/np.pi)/Pb#diametro equivalente/perimetro bounding box
    A5=4*np.pi*(Ap/(Pp**2))#circularidad 4*pi(Ap/Pp**2) da 1 si es una circunferencia perecta
    A6=Pp/Pb#perimetro de la lesion/perimetro bounding box
    A7=ma/MA#radio inferior elipse/radio inferior elipse
    A8=Pc/Pb#Perimetro convex hull/perimetro bounding box
    A9=tasaAspecto#tasa de aspecto bb/ab division de los lados del bounding box
    A10=Ap/Ae# Area de la lesion/Area de la elipse
    A11=Pp/Pe#perimetro de la lesion/perimetro de la elipse
    frac=fracAsim(mascara)
    B1=np.double(frac[0]+frac[1])/255
    B2=np.double(frac[2]+frac[3])/255
    A12=np.abs(np.sum(B1)-np.sum(B2))/np.sum(np.double(mascara)/255)#tasa de areas ap=(A1-A2)/Ap diferencia de areas de los pedazos cortados por el eje ap entre el area de la lesion 
    A13=np.sum(B2)/np.sum(B1) if np.sum(B1)>np.sum(B2) else np.sum(B1)/np.sum(B2)#tasa de forma ap=A1/A2 
    B1=frac[1]+frac[2]
    
    B1=np.double(frac[1]+frac[2])/255
    B2=np.double(frac[0]+frac[3])/255
    A14=np.abs(np.sum(B1)-np.sum(B2))/np.sum(np.double(mascara)/255)#tasa de areas bp=(B1-B2)/Ap
    A15=np.sum(B2)/np.sum(B1) if np.sum(B1)>np.sum(B2) else np.sum(B1)/np.sum(B2)#tasa de forma bp=B1/B2
    return A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15  
def textuVar(img):
    img1=np.double(img)/255
    L=(img1[:,:,0]+img1[:,:,1]+img1[:,:,2])/3
    [m,n]=L.shape
    Taumax=np.zeros((m,n))
    for c in range(0,9):
        k=c*4
        desv=(7+k)/7
        tam=np.uint8(7*desv)
        S=cv2.GaussianBlur(L,(tam,tam),desv,0)  
        Sn=1-S
        tau=L*(Sn)/S
        for i in range(0,m):
            for j in range(0,n):
               if tau[i,j]>Taumax[i,j]:
                  Taumax[i,j]=tau[i,j] 
    #Taumax=Taumax+L
    i1=(Taumax-np.min(Taumax))/(np.max(Taumax)-np.min(Taumax))
    return i1
def oscuInfo(img):
    img1=np.double(img)/255
    i2=(1-img1[:,:,0])
    return i2
def colInfo(img,mascara):
    mascara1=mascara.copy()
    mascara1=np.double(mascara1)/255
    a1=img[:,:,0]
    a2=img[:,:,1]
    a3=img[:,:,2]
    d1=a1.flatten()
    d2=a2.flatten()
    d3=a3.flatten()
    m=d1.shape
    m=m[0]
    mean=[np.mean(d1),np.mean(d2),np.mean(d3)]
    datos=np.array((d1,d2,d3))
    datos=datos.T-mean
    cov=np.cov(datos.T)
    valores, vectores = np.linalg.eigh(cov)#estan ordenados?
    vectores=-vectores
    U=vectores.copy()
    U[:,2]=vectores[:,0]
    U[:,0]=vectores[:,2]
    u1=U[:,0]
    ic=np.double(img.copy())/255
    c=np.double(a1.copy())
    [m,n,ca]=ic.shape
    for i in range(m):
        for j in range(n):
            ic[i,j]=ic[i,j]-mean 
            c[i,j]=np.abs(np.dot(u1,ic[i,j]))
            i3=(c-np.min(c))/(np.max(c)-np.min(c))
    a=np.sqrt(np.sum(mascara1))        
    k=np.uint64(roundOdd(0.0735*a))
    I3=cv2.medianBlur(np.uint8(255*i3),k)    
    I3=np.double(I3)/255    
    return I3
def varIm(I3seg,mascara):
    mascara1=mascara.copy()
    meanI3=np.sum(I3seg)/np.sum(mascara1)#media I3
    I3segCentrada=(I3seg-meanI3)*mascara1
    I3segCentrada=I3segCentrada**2
    varI3=np.sum(I3segCentrada)/np.sum(mascara1)#varianza I3
    return varI3

def fracBordes(I):
    a=fraccionar(I,8)
    b=[]
    for i in a:
        b.append(I*i)
    return b   

def caraColor(img,mascara):
    mascara1=mascara.copy()
    mascara1=np.double(mascara1)/255
    I3=colInfo(img,mascara)
    ic=np.double(img)/255
    I3seg=I3*mascara1
    maxI3=np.max(I3seg)#maximo I3
    minI3=np.min(I3seg+np.not_equal(mascara1,1))#minimo I3
    meanI3=np.sum(I3seg)/np.sum(mascara1)#media I3
    I3segCentrada=(I3seg-meanI3)*mascara1
    I3segCentrada=I3segCentrada**2
    varI3=np.sum(I3segCentrada)/np.sum(mascara1)#varianza I3
    icSeg=ic.copy()
    icSeg[:,:,0]=ic[:,:,0]*mascara1
    icSeg[:,:,1]=ic[:,:,1]*mascara1
    icSeg[:,:,2]=ic[:,:,2]*mascara1
    a=np.sqrt(np.sum(mascara1))
    k=0.0245
    ic1=cv2.GaussianBlur(icSeg,(0,0),k*a,0)  
    c1=maxI3
    c2=minI3
    c3=meanI3
    c4=varI3
    c5=np.max(ic1[:,:,0])#maximo de R
    c6=np.max(ic1[:,:,1])#maximo de G
    c7=np.max(ic1[:,:,2])#maximo de B
    c8=np.min(ic1[:,:,0]+np.not_equal(mascara1,1))#minimo de R
    c9=np.min(ic1[:,:,1]+np.not_equal(mascara1,1))#minimo de G
    c10=np.min(ic1[:,:,2]+np.not_equal(mascara1,1))#minimo de B
    c11=np.sum(ic1[:,:,0])/np.sum(mascara1)#media R
    c12=np.sum(ic1[:,:,1])/np.sum(mascara1)#media G
    c13=np.sum(ic1[:,:,2])/np.sum(mascara1)#media B
    c14=varIm(ic1[:,:,0],mascara1)#varianza R
    c15=varIm(ic1[:,:,1],mascara1)#varianza G
    c16=varIm(ic1[:,:,2],mascara1)#varianza B
    c17=c14/c15#R/G
    c18=c14/c16#R/B
    c19=c15/c16#G/B
    cont=np.zeros((6,1))
    cont=cont.flatten()
    blanco=np.array([1,1,1])
    red=np.array([0.8,0.2,0.2])
    cafeC=np.array([0.6,0.4,0])
    cafeO=np.array([0.2,0,0])
    grisAzul=np.array([0.2,0.6,0.6])
    [m,n,c]=ic.shape
    Idist=np.zeros((m,n,6))
    Idist[:,:,0]=np.linalg.norm(ic1-blanco.T,axis=2)*mascara1
    Idist[:,:,1]=np.linalg.norm(ic1-red.T,axis=2)*mascara1
    Idist[:,:,2]=np.linalg.norm(ic1-cafeC.T,axis=2)*mascara1
    Idist[:,:,3]=np.linalg.norm(ic1-cafeO.T,axis=2)*mascara1
    Idist[:,:,4]=np.linalg.norm(ic1-grisAzul.T,axis=2)*mascara1
    Idist[:,:,5]=np.linalg.norm(ic1,axis=2)*mascara1
    for i in range(0,m):
        for j in range(0,n):
            if mascara1[i,j]==1:
                c=Idist[i,j,:]
                d=(c==np.min(c))
                cont+=d
                
    cont=cont/np.sum(mascara1)
    c20=cont[0]#contador blanco        
    c21=cont[1]#contador rojo  
    c22=cont[2]#contador cafe claro
    c23=cont[3]#contador cafe oscuro  
    c24=cont[4]#contador gris azul  
    c25=cont[5]#contador negro  
    #caracteristicas agragadas
    c26=np.sum(Idist[:,:,0])/np.sum(mascara1)
    c27=np.sum(Idist[:,:,1])/np.sum(mascara1)
    c28=np.sum(Idist[:,:,2])/np.sum(mascara1)
    c29=np.sum(Idist[:,:,3])/np.sum(mascara1)
    c30=np.sum(Idist[:,:,4])/np.sum(mascara1)
    c31=np.sum(Idist[:,:,5])/np.sum(mascara1)
    c32=varIm(Idist[:,:,0],mascara1)
    c33=varIm(Idist[:,:,1],mascara1)
    c34=varIm(Idist[:,:,2],mascara1)
    c35=varIm(Idist[:,:,3],mascara1)
    c36=varIm(Idist[:,:,4],mascara1)
    c37=varIm(Idist[:,:,5],mascara1)
    
    frac=fracBordes(mascara)
    k1=np.sum(I3*np.double(frac[0])/255)/np.sum(np.double(frac[0])/255)
    k2=np.sum(I3*np.double(frac[1])/255)/np.sum(np.double(frac[1])/255)
    k3=np.sum(I3*np.double(frac[2])/255)/np.sum(np.double(frac[2])/255)
    k4=np.sum(I3*np.double(frac[3])/255)/np.sum(np.double(frac[3])/255)
    k5=np.sum(I3*np.double(frac[4])/255)/np.sum(np.double(frac[4])/255)
    k6=np.sum(I3*np.double(frac[5])/255)/np.sum(np.double(frac[5])/255)
    k7=np.sum(I3*np.double(frac[6])/255)/np.sum(np.double(frac[6])/255)
    k8=np.sum(I3*np.double(frac[7])/255)/np.sum(np.double(frac[7])/255)
    c38=(k1+k2+k3+k4+k5+k6+k7+k8)/8
    c39=np.var([k1,k2,k3,k4,k5,k6,k7,k8])
    return c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37,c38,c39
def orientacion(mascara):
    mascara1=mascara.copy()
    im2, contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    return vx,vy,x,y


def fracAngulo(mascara,vx,vy,x,y,n):
    rows,cols = mascara.shape[:2]
    a1=np.zeros((rows,cols))
    a2=np.zeros((rows,cols)) 
    k=90*(4/n)  
    t=k*(np.pi/180)
    for i in range(0,rows):
        for j in range(0,cols):
            a1[i,j]=np.dot([i,j],[-vx[0],vy[0]])-np.dot([y[0],x[0]],[-vx[0],vy[0]])>0      
    [vx1,vy1]=[vx*np.cos(t)-vy*np.sin(t),vx*np.sin(t)+vy*np.cos(t)]
    if np.dot([vx1[0],vy1[0]],[vx[0],vy[0]])<0:
        [vx1[0],vy1[0]]=[-vx1[0],-vy1[0]]
    for i in range(0,rows):
        for j in range(0,cols):
            a2[i,j]=np.dot([i,j],[-vx1[0],vy1[0]])-np.dot([y[0],x[0]],[-vx1[0],vy1[0]])>0 
    vx=vx1
    vy=vy1
    af1=(a1-a2)>0
    fracciones=[af1]
    for i in range(1,n):
        M = cv2.getRotationMatrix2D((round(x[0]),round(y[0])),k*i,1)
        dst = cv2.warpAffine(np.uint8(af1),M,(cols,rows))
        fracciones.append(dst)
    return fracciones
  
def caracBordes(img,mascara,I):
    B=np.double(I)/255
    mascara1=mascara.copy()
    m,n=mascara1.shape
    I1=textuVar(img)
    I2=oscuInfo(img)
    I3=colInfo(img,mascara1)
    
    I1dx = cv2.Sobel(I1,cv2.CV_64F,1,0)
    I1dy = cv2.Sobel(I1,cv2.CV_64F,0,1)
    dI1=np.sqrt(I1dx**2+I1dy**2)
    dI1B=dI1*B
    B1=np.sum(dI1B)/np.sum(B)#media del gradiente en I1
    
    I2dx = cv2.Sobel(I2,cv2.CV_64F,1,0)
    I2dy = cv2.Sobel(I2,cv2.CV_64F,0,1)
    dI2=np.sqrt(I2dx**2+I2dy**2)
    dI2B=dI2*B
    B2=np.sum(dI2B)/np.sum(B)#media del gradiente en I2
    I3dx = cv2.Sobel(I3,cv2.CV_64F,1,0)
    I3dy = cv2.Sobel(I3,cv2.CV_64F,0,1)
    dI3=np.sqrt(I3dx**2+I3dy**2)
    dI3B=dI3*B
    B3=np.sum(dI3B)/np.sum(B)#media del gradiente en I3
    B4=varIm(dI1B,B)#varianza de gradiente en I1
    B5=varIm(dI2B,B)#varianza de gradiente en I2
    B6=varIm(dI3B,B)#varianza de gradiente en I3
    fraccionesB=fracBordes(I)
    
    gB1=[]
    for i in fraccionesB:
        gB1.append(np.double(i)*dI1B/255)
    
    gB2=[]
    for i in fraccionesB:
        gB2.append(np.double(i)*dI2B/255)
    
    gB3=[]
    for i in fraccionesB:
        gB3.append(np.double(i)*dI3B/255)    
    
    k1=np.sum(gB1[0])/np.sum(np.double(fraccionesB[0])/255)
    k2=np.sum(gB1[1])/np.sum(np.double(fraccionesB[1])/255)
    k3=np.sum(gB1[2])/np.sum(np.double(fraccionesB[2])/255)
    k4=np.sum(gB1[3])/np.sum(np.double(fraccionesB[3])/255)
    k5=np.sum(gB1[4])/np.sum(np.double(fraccionesB[4])/255)
    k6=np.sum(gB1[5])/np.sum(np.double(fraccionesB[5])/255)
    k7=np.sum(gB1[6])/np.sum(np.double(fraccionesB[6])/255)
    k8=np.sum(gB1[7])/np.sum(np.double(fraccionesB[7])/255)
    
    B7=(k1+k2+k3+k4+k5+k6+k7+k8)/8
    
    B8=np.var([k1,k2,k3,k4,k5,k6,k7,k8])
    
    k1=np.sum(gB2[0])/np.sum(np.double(fraccionesB[0])/255)
    k2=np.sum(gB2[1])/np.sum(np.double(fraccionesB[1])/255)
    k3=np.sum(gB2[2])/np.sum(np.double(fraccionesB[2])/255)
    k4=np.sum(gB2[3])/np.sum(np.double(fraccionesB[3])/255)
    k5=np.sum(gB2[4])/np.sum(np.double(fraccionesB[4])/255)
    k6=np.sum(gB2[5])/np.sum(np.double(fraccionesB[5])/255)
    k7=np.sum(gB2[6])/np.sum(np.double(fraccionesB[6])/255)
    k8=np.sum(gB2[7])/np.sum(np.double(fraccionesB[7])/255)
    
    B9=(k1+k2+k3+k4+k5+k6+k7+k8)/8
    
    B10=np.var(np.array([k1,k2,k3,k4,k5,k6,k7,k8]))   
    
    k1=np.sum(gB3[0])/np.sum(np.double(fraccionesB[0])/255)
    k2=np.sum(gB3[1])/np.sum(np.double(fraccionesB[1])/255)
    k3=np.sum(gB3[2])/np.sum(np.double(fraccionesB[2])/255)
    k4=np.sum(gB3[3])/np.sum(np.double(fraccionesB[3])/255)
    k5=np.sum(gB3[4])/np.sum(np.double(fraccionesB[4])/255)
    k6=np.sum(gB3[5])/np.sum(np.double(fraccionesB[5])/255)
    k7=np.sum(gB3[6])/np.sum(np.double(fraccionesB[6])/255)
    k8=np.sum(gB3[7])/np.sum(np.double(fraccionesB[7])/255)
    
    B11=(k1+k2+k3+k4+k5+k6+k7+k8)/8
    
    B12=np.var(np.array([k1,k2,k3,k4,k5,k6,k7,k8]))
    
    return B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12

def caracDifEstruct(img,mascara):
    mascara1=np.double(mascara.copy())/255
    I1=textuVar(img)
    I1seg=I1*mascara1
    D1=np.max(I1seg)
    D2=np.min(I1seg+np.not_equal(mascara1,1))
    D3=np.sum(I1seg)/np.sum(mascara1)
    D4=varIm(I1seg,mascara1)
    return D1,D2,D3,D4
#--------------------------------CARACTERISTICAS-----------------------------------------------
#--------------------------------asimetria-----------------------------------------------
#    A1=Ap/Ab area lesion/area bounding box
#    A2=Ac/Ab area convex hull/area bounding box
#    A3=Ap/Ac area de la lesion/area del convex hull
#    A4=np.sqrt(4*Ap/np.pi)/Pb#diametro equivalente/perimetro bounding box
#    A5=4*np.pi*(Ap/(Pp**2)) circularidad da 1 si es una circunferencia perecta
#    A6=Pp/Pb perimetro de la lesion/perimetro bounding box
#    A7=ma/MA radio inferior elipse/radio inferior elipse
#    A8=Pc/Pb Perimetro convex hull/perimetro bounding box
#    A9=tasa de aspecto bb/ab division de los lados del bounding box
#    A10=Ap/Ae Area de la lesion/Area de la elipse
#    A11=Pp/Pe perimetro de la lesion/perimetro de la elipse
#    A12=tasa de areas ap=(A1-A2)/Ap diferencia de areas de los pedazos cortados por el eje ap entre el area de la lesion 
#    A13=tasa de forma ap=A1/A2 
#    A14=tasa de areas bp=(B1-B2)/Ap
#    A15=tasa de forma bp=B1/B2
    
imagen="melanoma.jpg"

img = cv2.imread(imagen)

mask,amax=segmentarLunar(img)

mascara=obtenerMask(mask,amax)

I=mascara-mask

B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12=caracBordes(img,mascara,I)

D1,D2,D3,D4=caracDifEstruct(img,mascara)

A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15=CaracAsimetria(mascara)

c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37,c38,c39=caraColor(img,mascara)

carac=[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37,c38,c39,D1,D2,D3,D4]

