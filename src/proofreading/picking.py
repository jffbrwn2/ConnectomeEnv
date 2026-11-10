"""Pick the frontmost triangle in the exact saved orthographic scene."""
import numpy as np


class PickMiss(ValueError):
    """A valid click that did not hit foreground geometry."""


def ray(camera, bounds, x, y):
    if not np.isfinite([x,y]).all() or not (0<=x<900 and 0<=y<700):
        raise ValueError('Click outside 3D image (900 by 700 pixels)')
    az,el=np.deg2rad([camera['azimuth'],camera['elevation']])
    eye=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right=np.array([-np.sin(az),np.cos(az),0.]);up=np.cross(eye,right)
    target=np.asarray(camera['target_nm'],float);lo,hi=np.asarray(bounds,float)
    corners=np.array([[a,b,c] for a in (lo[0],hi[0]) for b in (lo[1],hi[1]) for c in (lo[2],hi[2])])
    depth=max(float(np.max(np.abs((corners-target)@eye))),1000.)+1000.
    half=camera['view_height_nm']/2
    origin=target+right*((2*(x+.5)/900-1)*half*9/7)+up*((1-2*(y+.5)/700)*half)+eye*depth
    return origin,-eye,2*depth


def surface(meshes, camera, bounds, x, y):
    origin,direction,limit=ray(camera,bounds,x,y)
    hit=None;closest=limit
    for sid,vertices,faces in meshes:
        for start in range(0,len(faces),32768):
            t=np.asarray(vertices[faces[start:start+32768]],float)
            e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0]
            h=np.cross(direction,e2);det=np.einsum('ij,ij->i',e1,h)
            valid=np.abs(det)>1e-12
            inv=np.divide(1.,det,out=np.zeros_like(det),where=valid)
            s=origin-t[:,0];u=np.einsum('ij,ij->i',s,h)*inv
            q=np.cross(s,e1);v=q@direction*inv
            distance=np.einsum('ij,ij->i',e2,q)*inv
            valid&=(u>=0)&(v>=0)&(u+v<=1)&(distance>=0)&(distance<=closest)
            if valid.any():
                index=np.argmin(np.where(valid,distance,np.inf));closest=float(distance[index])
                hit=(int(sid),origin+closest*direction,direction)
    if hit is None:raise PickMiss('No segment surface at this pixel; click inside a visible surface')
    return hit
