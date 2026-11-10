"""One-shot mesh inspection using the same accelerated renderer as set_scene."""
import time
import numpy as np
from .scene import geometry
from .raster import render


def render_mesh(session, ids, elevation, azimuth, stride, output):
    started=time.perf_counter()
    meshes=geometry(session,ids,stride)
    extracted=time.perf_counter()
    lo=np.min([v.min(axis=0) for _,v,_ in meshes],axis=0)
    hi=np.max([v.max(axis=0) for _,v,_ in meshes],axis=0)
    target=(lo+hi)/2
    az,el=np.deg2rad([azimuth,elevation])
    eye=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right=np.array([-np.sin(az),np.cos(az),0.]);up=np.cross(eye,right)
    corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])-target
    height=max(np.ptp(corners@up),np.ptp(corners@right)*7/9,1.)*1.08
    camera={'target_nm':target,'azimuth':azimuth,'elevation':elevation,'view_height_nm':height}
    v=session.view
    plane=[v.xyz(x,y) for x,y in [(-.5,-.5),(v.width-.5,-.5),(v.width-.5,v.height-.5),(-.5,v.height-.5),(-.5,-.5)]]
    timings=render(meshes,camera,(session.dataset.atoms.origin,session.dataset.atoms.upper),plane,output)
    timings.update(geometry_seconds=extracted-started,total_seconds=time.perf_counter()-started,
                   triangles=sum(len(f) for _,_,f in meshes),stride=stride,
                   geometry_sources=session._scene_geometry_sources)
    return timings
