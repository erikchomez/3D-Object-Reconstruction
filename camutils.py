import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.spatial import Delaunay

def makerotation(rx,ry,rz):
    """
    Generate a rotation matrix    

    Parameters
    ----------
    rx,ry,rz : floats
        Amount to rotate around x, y and z axes in degrees

    Returns
    -------
    R : 2D numpy.array (dtype=float)
        Rotation matrix of shape (3,3)
    """
    rx = np.pi*rx/180.0
    ry = np.pi*ry/180.0
    rz = np.pi*rz/180.0

    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,-np.sin(ry)],[0,1,0],[np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    R = (Rz @ Ry @ Rx)
    
    return R 

class Camera:
    """
    A simple data structure describing camera parameters 
    
    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation 

    
    """    
    def __init__(self,f,c,R,t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'
    
    def project(self,pts3):
        """
        Project the given 3D points in world coordinates into the specified camera    

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        Returns
        -------
        pts2 : 2D numpy.array (dtype=float)
            Image coordinates of N points stored in an array of shape (2,N)

        """
        assert(pts3.shape[0]==3)

        # get point location relative to camera
        pcam = self.R.transpose() @ (pts3 - self.t)
         
        # project
        p = self.f * (pcam / pcam[2,:])
        
        # offset principal point
        pts2 = p[0:2,:] + self.c
        
        assert(pts2.shape[1]==pts3.shape[1])
        assert(pts2.shape[0]==2)
    
        return pts2
 
    def update_extrinsics(self,params):
        """
        Given a vector of extrinsic parameters, update the camera
        to use the provided parameters.
  
        Parameters
        ----------
        params : 1D numpy.array (dtype=float)
            Camera parameters we are optimizing over stored in a vector
            params[0:2] are the rotation angles, params[2:5] are the translation

        """
        self.R = makerotation(params[0],params[1],params[2])
        self.t = np.array([[params[3]],[params[4]],[params[5]]])


def triangulate(pts2L,camL,pts2R,camR):
    """
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coordinates relative
    to the global coordinate system


    Parameters
    ----------
    pts2L : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camL camera

    pts2R : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camR camera

    camL : Camera
        The first "left" camera view

    camR : Camera
        The second "right" camera view

    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
        (3,N) array containing 3D coordinates of the points in global coordinates

    """

    npts = pts2L.shape[1]

    qL = (pts2L - camL.c) / camL.f
    qL = np.vstack((qL,np.ones((1,npts))))

    qR = (pts2R - camR.c) / camR.f
    qR = np.vstack((qR,np.ones((1,npts))))
    
    R = camL.R.T @ camR.R
    t = camL.R.T @ (camR.t-camL.t)

    xL = np.zeros((3,npts))
    xR = np.zeros((3,npts))

    for i in range(npts):
        A = np.vstack((qL[:,i],-R @ qR[:,i])).T
        z,_,_,_ = np.linalg.lstsq(A,t,rcond=None)
        xL[:,i] = z[0]*qL[:,i]
        xR[:,i] = z[1]*qR[:,i]
 
    pts3L = camL.R @ xL + camL.t
    pts3R = camR.R @ xR + camR.t
    pts3 = 0.5*(pts3L+pts3R)

    return pts3


def residuals(pts3,pts2,cam,params):
    """
    Compute the difference between the projection of 3D points by the camera
    with the given parameters and the observed 2D locations

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    params : 1D numpy.array (dtype=float)
        Camera parameters we are optimizing over stored in a vector

    Returns
    -------
    residual : 1D numpy.array (dtype=float)
        Vector of residual 2D projection errors of size 2*N
        
    """

    cam.update_extrinsics(params)
    residual = pts2 - cam.project(pts3)
    
    return residual.flatten()

def calibratePose(pts3,pts2,cam_init,params_init):
    """
    Calibrate the provided camera by updating R,t so that pts3 projects
    as close as possible to pts2

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    cam : Camera
        Initial estimate of camera

    Returns
    -------
    cam_opt : Camera
        Refined estimate of camera with updated R,t parameters
        
    """

    # define our error function
    efun = lambda params: residuals(pts3,pts2,cam_init,params)        
    popt,_ = scipy.optimize.leastsq(efun,params_init)
    cam_init.update_extrinsics(popt)

    return cam_init



def decode(imprefix,c_imprefix,start,threshold,c_thresh):
    """
    Decode 10bit gray code pattern with the given difference
    threshold.  We assume the images come in consective pairs
    with filenames of the form <prefix><start>.png - <prefix><start+20>.png
    (e.g. a start offset of 20 would yield image20.png, image01.png... image39.png)

    Parameters
    ----------
    imprefix : str
      prefix of where to find the images (assumed to be .png)
    
    c_imprefix : str
      prefix of where to find the color images

    start : int
      image offset

    threshold : float
      decodability threshold
      
    c_thresh : float
      threshold for color images

    Returns
    -------
    code : 2D numpy.array (dtype=float)
        
    mask : 2D numpy.array (dtype=float)
    
    c_mask : 2D numpy.array (dtype=float)
    
    
    """
    nbits = 10
    
    imgs = list()
    imgs_inv = list()
    print('loading',end='')
    for i in range(start,start+2*nbits,2):
        fname0 = '%s%2.2d.png' % (imprefix,i)
        fname1 = '%s%2.2d.png' % (imprefix,i+1)
       
        print('(',i,i+1,')',end='')
        img = plt.imread(fname0)
        img_inv = plt.imread(fname1)
        if (img.dtype == np.uint8):
            img = img.astype(float) / 256
            img_inv = img_inv.astype(float) / 256
        if (len(img.shape)>2):
            img = np.mean(img,axis=2)
            img_inv = np.mean(img_inv,axis=2)
        imgs.append(img)
        imgs_inv.append(img_inv)
        
    (h,w) = imgs[0].shape
    print('\n')
    
    gcd = np.zeros((h,w,nbits))
    mask = np.ones((h,w))
    for i in range(nbits):
        gcd[:,:,i] = imgs[i]>imgs_inv[i]
        mask = mask * (np.abs(imgs[i]-imgs_inv[i])>threshold)
        
    bcd = np.zeros((h,w,nbits))
    bcd[:,:,0] = gcd[:,:,0]
    for i in range(1,nbits):
        bcd[:,:,i] = np.logical_xor(bcd[:,:,i-1],gcd[:,:,i])
        
    code = np.zeros((h,w))
    for i in range(nbits):
        code = code + np.power(2,(nbits-i-1))*bcd[:,:,i]
        
    img1_c = plt.imread('%s%2.2d.png' % (c_imprefix,0))
    img2_c = plt.imread('%s%2.2d.png' % (c_imprefix,1))
    
#     print(img1_c.shape)
#     print(img2_c.shape)
    
    # we can compute the mask by taking the difference of the object color image
    # and the background image and thresholding the difference to determine
    # the set of pixels that belong to the foreground object
    
    c_mask = np.ones((h,w))
    c_mask = c_mask * ((np.sum(np.square(img1_c - img2_c), axis=-1)) > c_thresh)
    
#     imc1 = plt.imread(c_imprefix +"%02d" % (0)+'.png')
#     imc2= plt.imread(c_imprefix +"%02d" % (1)+'.png')
#     c_mask = np.ones((h,w))
#     c_mask = c_mask*((np.sum(np.square(imc1-imc2), axis=-1))>c_thresh)
    
    return code,mask,c_mask



def reconstruct(imprefixLF,imprefixLC,imprefixRF,imprefixRC,threshold,c_thresh,camL,camR):
    """
    Simple reconstruction based on triangulating matched pairs of points
    between to view which have been encoded with a 20bit gray code. Also
    records rgb values for corresponding pixels to maintain synchronization.

    Parameters
    ----------
    imprefixLF,imprefixRF : str
      prefix for where the images frames are stored
    
    imprefixLC,imprefixRC : str
      prefix for where the color images are stored
      
    threshold : float
      decodability threshold
    
    c_thresh : float
      threshold for color images
    
    camL,camR : Camera
      camera parameters

    Returns
    -------
    pts2L,pts2R : 2D numpy.array (dtype=float)

    pts3 : 2D numpy.array (dtype=float)
    
    rgb_values : 3xN numpy.array (dtype=float)

    """
    CLh,maskLh,lc_mask = decode(imprefixLF,imprefixLC,0,threshold,c_thresh)
    CLv,maskLv,_ = decode(imprefixLF,imprefixLC,20,threshold,c_thresh)
    CRh,maskRh,rc_mask = decode(imprefixRF,imprefixRC,0,threshold,c_thresh)
    CRv,maskRv,_ = decode(imprefixRF,imprefixRC,20,threshold,c_thresh)

    CL = CLh + 1024*CLv
    maskL = maskLh*maskLv
    CR = CRh + 1024*CRv
    maskR = maskRh*maskRv

    h = CR.shape[0]
    w = CR.shape[1]

    subR = np.nonzero(maskR.flatten())
    subL = np.nonzero(maskL.flatten())

    CRgood = CR.flatten()[subR]
    CLgood = CL.flatten()[subL]

    _,submatchR,submatchL = np.intersect1d(CRgood,CLgood,return_indices=True)

    matchR = subR[0][submatchR]
    matchL = subL[0][submatchL]

    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))
    
    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)
    
    rgb_values = compute_rgbvals(imprefixLC,imprefixRC,pts2L,pts2R)

    pts3 = triangulate(pts2L,camL,pts2R,camR)

    return pts2L,pts2R,pts3,rgb_values


def compute_rgbvals(imprefixLF,imprefixRF,pts2L,pts2R):
    """
    For each point that we triangulate, we want to record the color of the 
    corresponding pixel in the color image.
    
    Parameters
    ----------
    imprefixLF : str
      prefix for where the left images are stored

    imprefixRF : str
      prefix for where the right images are stored
      
    pts2L,pts2R : 2D numpy.array (dtype=float)
      the 2D pixel coordinates of the matched pixels in the legt and right
      image stored in arras of shape 2xN

    Returns
    -------
    rgb_values : 3xN numpy.array (dtype=float)
    
    """
    left_image = plt.imread('%s%2.2d.png' % (imprefixLF,1))
    right_image = plt.imread('%s%2.2d.png' % (imprefixRF,1))
#     left_image= plt.imread(imprefixLF +"%02d" % (1)+'.png')
#     right_image = plt.imread(imprefixRF +"%02d" % (1)+'.png')
#     left_rgb = [left_image[pts2L[1][i]][pts2L[0][i]] for i in range(pts2L.shape[1])]
#     right_rgb = [right_image[pts2R[1][i]][pts2R[0][i]] for i in range(pts2L.shape[1])]
    l_rgb = []
    r_rgb = []
    for i in range(pts2L.shape[1]):
        l_rgb.append(left_image[pts2L[1][i]][pts2L[0][i]])
        r_rgb.append(right_image[pts2R[1][i]][pts2R[0][i]])
        
    rgb_values = ((np.array(l_rgb).T + np.array(r_rgb).T) / 2)
    
    return rgb_values


def bounding_box_pruning(boxlimits, pts3, pts2L, pts2R, rgb_values):
    """
    Points that are outside the range specified are dropped from pts3, pts2L,
    pts2R, and rgb_values.
    This function is an adaptation to what I had in my assignment4
    
    Parameters
    ----------
    boxlimits : numpy.array in shape (6,) 
      x,y,z limits
      does not work with float values
    
    pts3, pts2L, pts2R, rgb_values : numpy.arrays
      points to be pruned from
      
    Returns
    -------
    pts3, pts2L, pts2R, rgb_values
    
    """
    to_delete = np.where(
                (pts3[0,:] < boxlimits[0]) | (pts3[0,:] > boxlimits[1]) |\
                (pts3[1,:] < boxlimits[2]) | (pts3[1,:] > boxlimits[3]) |\
                (pts3[2,:] < boxlimits[4]) | (pts3[2,:] > boxlimits[5]))

    pts3 = np.delete(pts3, to_delete, 1)
    pts2L = np.delete(pts2L, to_delete, 1)
    pts2R = np.delete(pts2R, to_delete, 1)
    rgb_values = np.delete(rgb_values, to_delete, 1)
    
    return pts3, pts2L, pts2R, rgb_values


def neighboring_points(k, tri):
    """
    We can find the neighboring 
    points using scipy.spatial.Delaunay.vertex_neighbor_vertices
    
    Which returns a tuple of two ndarrays of int: (indptr, indices)
    # taken directly from the documentation
    The indices of neighboring vertices of 
    vertex k are indices[indptr[k]:indptr[k+1]]
    
    Parameters
    ----------
    k : int
      point index
    tri : Delaunay points
      set of points that have been triangulated
      
    Returns
    -------
    numpy.ndarray 
      neighboring vertices
    """
    indptr = tri.vertex_neighbor_vertices[0]
    indices = tri.vertex_neighbor_vertices[1]
    
    return indices[indptr[k]:indptr[k+1]]

    
def mesh_smoothing(pts3, tri, n):
    """
    Mesh smoothing
    
    Parameters
    ----------
    pts3 : numpy.ndarray
    tri : Delaunay
    n : int
    
    Returns
    ----------
    pts3 : numpy.ndarray
      with mesh smoothing applied
    """
    for i in range(n):
        pts3[:,i] = np.mean(pts3[:,neighboring_points(i,tri)], axis=1)
    
    return pts3
        

def mesh_gen(pts3, pts2L, pts2R, rgb_values,boxlimits, trithresh):
    pts3, pts2L, pts2R, rgb_values = bounding_box_pruning(boxlimits, pts3, pts2L, pts2R, rgb_values)
    
    tri = Delaunay(pts2L.T)
    s = tri.simplices

    pts3 = mesh_smoothing(pts3,tri,pts3.shape[1])
    pts3 = mesh_smoothing(pts3,tri,pts3.shape[1])
    #pts3 = mesh_smoothing(pts3,tri,pts3.shape[1])
    #trithresh = 0.09

#     allowed_edges = np.where( 
#                      (np.sqrt(np.sum(np.power(pts3[:,s[:,0]] - pts3[:,s[:,1]],2),axis=0)) < trithresh) &\
#                      (np.sqrt(np.sum(np.power(pts3[:,s[:,0]] - pts3[:,s[:,2]],2),axis=0)) < trithresh) &\
#                      (np.sqrt(np.sum(np.power(pts3[:,s[:,1]] - pts3[:,s[:,2]],2),axis=0)) < trithresh))

    edges_1 = np.sqrt(np.sum(np.power(pts3[:,s[:,0]]-pts3[:,s[:,1]],2),axis=0))
    edges_2 = np.sqrt(np.sum(np.power(pts3[:,s[:,0]]-pts3[:,s[:,2]],2),axis=0))
    edges_3 = np.sqrt(np.sum(np.power(pts3[:,s[:,1]]-pts3[:,s[:,2]],2),axis=0))

    allowed_edges = (edges_1 < trithresh) & (edges_2 < trithresh) & (edges_3 < trithresh)
    s = s[allowed_edges,:]  # s = tri.simplices
    
    map_ = np.zeros(pts3.shape[1])
    to_keep = np.unique(s)
    
    pts3 = pts3[:, to_keep]
    pts2L = pts2L[:, to_keep]
    pts2R = pts2R[:, to_keep]
    rgb_values = rgb_values[:, to_keep]

    map_[to_keep] = np.arange(0, (to_keep).shape[0])
    s = map_[s]
    
    return pts3, s, rgb_values
    
    
    
    
        