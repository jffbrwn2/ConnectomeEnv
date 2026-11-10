"""Persistent, thread-local headless OpenGL rendering of full-resolution meshes.

Linux uses EGL; macOS uses CGL. The driver name is returned in timing metadata,
so software Mesa rendering is never misreported as GPU acceleration. Context
creation errors are explicit; there is no hidden Matplotlib fallback.
"""
import sys
import threading
from collections import OrderedDict
import time
import numpy as np
from PIL import Image
from .view import color

_LOCAL = threading.local()
_VERTEX = '''#version 330
in vec3 position;
uniform mat4 camera;
out vec3 world;
void main() { world = position; gl_Position = camera * vec4(position, 1.0); }
'''
_FRAGMENT = '''#version 330
in vec3 world;
uniform vec3 tint;
uniform vec3 eye;
uniform bool lines;
out vec4 frag;
void main() {
    float shade = lines ? 1.0 : 0.5 + 0.5 * abs(dot(normalize(cross(dFdx(world), dFdy(world))), eye));
    frag = vec4(tint * shade, 1.0);
}
'''


class Renderer:
    def __init__(self):
        import moderngl
        self.gl = moderngl
        try:
            self.ctx = moderngl.create_context(standalone=True, **({'backend':'egl'} if sys.platform.startswith('linux') else {}))
        except Exception as exc:
            raise RuntimeError('Headless OpenGL initialization failed; Linux requires EGL/Mesa or an NVIDIA EGL driver') from exc
        self.program = self.ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self.meshes = None
        self.mesh_origin = None
        self.gpu_cache = OrderedDict()
        self.buffers = []
        self.fbo = self.ctx.simple_framebuffer((900,700), components=3, samples=4)
        self.resolved = self.ctx.simple_framebuffer((900,700), components=3)
        self.line_buffer = self.ctx.buffer(reserve=32*3*4)
        self.line_vao = self.ctx.vertex_array(self.program, [(self.line_buffer,'3f','position')])

    def upload(self, meshes, origin):
        origin_key=tuple(map(float,origin))
        if self.meshes is meshes and self.mesh_origin==origin_key:
            return False
        buffers=[];active=set();uploaded=False
        for sid,vertices,faces in meshes:
            key=(id(vertices),id(faces),origin_key);active.add(key)
            if key in self.gpu_cache:
                self.gpu_cache.move_to_end(key)
                vao,vbo,ibo,_,_,size=self.gpu_cache[key]
            else:
                vertex_data=np.asarray((vertices-origin)/1000,dtype='f4')
                face_data=np.asarray(faces,dtype='u4')
                vbo=self.ctx.buffer(vertex_data.tobytes());ibo=self.ctx.buffer(face_data.tobytes())
                vao=self.ctx.vertex_array(self.program,[(vbo,'3f','position')],ibo,index_element_size=4)
                size=vertex_data.nbytes+face_data.nbytes
                # Retain input arrays so Python cannot recycle their identity
                # while the corresponding GPU buffers remain cached.
                self.gpu_cache[key]=(vao,vbo,ibo,vertices,faces,size)
                uploaded=True
            buffers.append((sid,vao,vbo,ibo))
        size=sum(entry[-1] for entry in self.gpu_cache.values())
        for key in list(self.gpu_cache):
            if size<=256*1024**2:break
            if key in active:continue
            vao,vbo,ibo,_,_,n=self.gpu_cache.pop(key)
            vao.release();vbo.release();ibo.release();size-=n
        self.buffers=buffers;self.meshes=meshes;self.mesh_origin=origin_key
        return uploaded

    def render(self, meshes, camera, bounds, plane, output, *, show_bounds=True):
        started=time.perf_counter()
        lo,hi=np.asarray(bounds,dtype=float)
        uploaded=self.upload(meshes,lo)
        uploaded_at=time.perf_counter()
        az,el=np.deg2rad([camera['azimuth'],camera['elevation']])
        eye=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
        right=np.array([-np.sin(az),np.cos(az),0.]);up=np.cross(eye,right)
        target=(np.asarray(camera['target_nm'])-lo)/1000
        half=camera['view_height_nm']/2000
        corners=np.array([[x,y,z] for x in (0,(hi[0]-lo[0])/1000) for y in (0,(hi[1]-lo[1])/1000) for z in (0,(hi[2]-lo[2])/1000)])
        # Depth bounds cover the entire volume even when the camera target is far outside it.
        depth=max(float(np.max(np.abs((corners-target)@eye))),1.)+1.
        matrix=np.eye(4,dtype='f4')
        matrix[:3,:3]=np.stack([right/(half*9/7),up/half,-eye/depth])
        matrix[:3,3]=-matrix[:3,:3]@target
        self.program['camera'].write(matrix.T.copy().tobytes())
        self.program['eye'].value=tuple(eye)
        self.fbo.use();self.fbo.clear(9/255,21/255,29/255,1,depth=1)
        self.ctx.disable(self.gl.BLEND|self.gl.CULL_FACE|self.gl.DEPTH_TEST)
        self.program['lines'].value=True
        lines=np.array([corners[k] for i in range(8) for j in range(i+1,8) if (i^j).bit_count()==1 for k in (i,j)],dtype='f4')
        self.line_buffer.write(lines.tobytes());self.program['tint'].value=(56/255,87/255,97/255)
        if show_bounds:self.line_vao.render(self.gl.LINES,vertices=len(lines))
        self.ctx.enable(self.gl.DEPTH_TEST);self.ctx.depth_func='<='
        self.program['lines'].value=False
        for sid,vao,_,_ in self.buffers:
            self.program['tint'].value=tuple(color(sid)/255)
            vao.render(self.gl.TRIANGLES)
        # The EM plane outline is an annotation, always visible like the prior renderer.
        self.ctx.disable(self.gl.DEPTH_TEST);self.program['lines'].value=True
        self.program['tint'].value=(249/255,201/255,121/255)
        points=np.asarray((np.asarray(plane)-lo)/1000,dtype='f4')
        self.line_buffer.write(points.tobytes());self.line_vao.render(self.gl.LINE_STRIP,vertices=len(points))
        self.ctx.copy_framebuffer(self.resolved,self.fbo)
        raw=self.resolved.read(components=3,alignment=1)
        rendered_at=time.perf_counter()
        image=Image.frombytes('RGB',(900,700),raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if output is not None:image.save(output,compress_level=1)
        return {**({'image':image} if output is None else {}),'backend':'opengl','samples':4,'device':self.ctx.info['GL_RENDERER'],'uploaded':uploaded,
                'upload_seconds':uploaded_at-started,'draw_readback_seconds':rendered_at-uploaded_at,
                'png_seconds':time.perf_counter()-rendered_at}


def render(meshes,camera,bounds,plane,output):
    if not hasattr(_LOCAL,'renderer'):
        _LOCAL.renderer=Renderer()
    return _LOCAL.renderer.render(meshes,camera,bounds,plane,output)
