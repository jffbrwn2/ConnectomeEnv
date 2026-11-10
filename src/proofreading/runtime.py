"""Explicit renderer selection for CPU or GPU hosts."""
import ctypes
import ctypes.util
import json
import os
from pathlib import Path


def configure():
    mode = os.environ.get('CONNECTOME_RENDERER', 'system')
    if mode == 'system':
        return
    if mode != 'software':
        raise ValueError('CONNECTOME_RENDERER must be system or software')
    vendor=Path('/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    if not vendor.is_file():raise RuntimeError('System Mesa EGL vendor descriptor is required')
    os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
    os.environ['EGL_PLATFORM']='surfaceless'
    os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']=str(vendor)
    library=ctypes.util.find_library('EGL')
    if not library:raise RuntimeError('System EGL library is required')
    egl=ctypes.CDLL(library)
    egl.eglGetProcAddress.argtypes=[ctypes.c_char_p]
    egl.eglGetProcAddress.restype=ctypes.c_void_p
    query_ptr=egl.eglGetProcAddress(b'eglQueryDevicesEXT')
    string_ptr=egl.eglGetProcAddress(b'eglQueryDeviceStringEXT')
    if not query_ptr or not string_ptr:raise RuntimeError('EGL device enumeration extensions unavailable')
    query=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int))(query_ptr)
    strings=ctypes.CFUNCTYPE(ctypes.c_char_p,ctypes.c_void_p,ctypes.c_int)(string_ptr)
    count=ctypes.c_int()
    if not query(0,None,ctypes.byref(count)):raise RuntimeError('Cannot enumerate EGL devices')
    devices=(ctypes.c_void_p*count.value)()
    if not query(count,devices,ctypes.byref(count)):raise RuntimeError('Cannot read EGL devices')
    software=[i for i,device in enumerate(devices) if b'EGL_MESA_device_software' in (strings(device,0x3055) or b'').split()]
    if len(software)!=1:raise RuntimeError(f'Expected one explicit Mesa software device, found {len(software)}')
    os.environ['GLCONTEXT_DEVICE_INDEX']=str(software[0])
    print(json.dumps({'renderer_policy':'explicit Mesa software device','device_index':software[0]}),flush=True)


def main():
    configure()
    from .raster import Renderer
    renderer = Renderer()
    print(json.dumps({'renderer': renderer.ctx.info['GL_RENDERER'],
                      'version': renderer.ctx.info['GL_VERSION']}))


if __name__ == '__main__':
    main()
