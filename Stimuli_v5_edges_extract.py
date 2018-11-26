import bpy, bmesh
from math import *
from mathutils import Vector
import random
import bpy
from bpy import context
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view
from itertools import accumulate
import numpy as np
import math


def getAllCoords(obj):
    coords = [(obj.matrix_world * v.co) for v in obj.data.vertices]
    #print(coords)
    
# Create a BVH tree and return bvh and vertices in world coordinates 
def BVHTreeAndVerticesInWorldFromObj( obj ):
    mWorld = obj.matrix_world
    vertsInWorld = [mWorld * v.co for v in obj.data.vertices]
    bvh = BVHTree.FromPolygons( vertsInWorld, [p.vertices for p in obj.data.polygons] )
    return bvh, vertsInWorld

def createVertices(NUMVERTS=5):
    rsum = 0
    rands = [random.randrange(2,5,1) for _ in range (NUMVERTS)]
    rsum = sum(rands)
    dphis = [(2*pi*rands[i])/rsum for i in range(NUMVERTS)]
    Dphis = np.cumsum(dphis)
    
    #calculate x,y coordinate pairs
    #coords = [(cos(i*dphis[i]),sin(i*dphis[i]),0) for i in range(NUMVERTS)]
    coords = [(cos(Dphis[i]),sin(Dphis[i]),0) for i in range(NUMVERTS)]

    bm = bmesh.new()
    for v in coords:
        bm.verts.new(v)

    # think of this new vertices as bottom of the extruded shape
    bottom = bm.faces.new(bm.verts)

    # next we create top via extrude operator, note it doesn't move the new face
    # we make our 1 face into a list so it can be accepted to geom
    top = bmesh.ops.extrude_face_region(bm, geom=[bottom])

    # here we move all vertices returned by the previous extrusion
    # filter the "geom" list for vertices using list constructor
    bmesh.ops.translate(bm, vec=Vector((0,0,1)), verts=[v for v in top["geom"] if isinstance(v,bmesh.types.BMVert)])
    bm.normal_update()

    me = bpy.data.meshes.new("cube")
    bm.to_mesh(me)
    

    # add bmesh to scene
    ob = bpy.data.objects.new("cube",me)
    bpy.context.scene.objects.link(ob)
    bpy.context.scene.update()
    
    ob.rotation_euler = (radians(90), radians(30), radians(10))
    bpy.context.scene.update()   
    
    coords_all = []
    for verts in bm.verts:
        print("verts:",verts,"coords:", ob.matrix_world * verts.co)
        coords_all.append(ob.matrix_world * verts.co)
        
    for edge in bm.edges:
        print("edge vertices:", edge.verts[0].index, edge.verts[1].index)
    
    N=NUMVERTS*2
    conn=[]
    for j in range(1,N):
        for i in range(1,N-j+1):
            print ("search for:", i-1,i+j-1)
            F=0
            for edge in bm.edges:
                if (edge.verts[0].index == i-1 and edge.verts[1].index == i+j-1) or (edge.verts[0].index == i+j-1 and edge.verts[1].index == i-1):
                    print ("Y:",edge.verts[0].index, edge.verts[1].index)
                    F=1
            conn.append(F)        
  
    print("conn vec", conn, "len: ", len(conn))    
            
    print("std dev of all angles: ", np.std(dphis), "\n")
    print("\n")
    return ob, dphis, np.std(dphis),coords_all, conn

def createNObjects(N=10):
    data = []
    for i in range(N):
        # delete all objects
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()
        ob, dphis, sda, coords, conn  = createVertices(NUMVERTS)
        #data.append([dphis, sda, coords])
        # relocate and rotate
        #ob.location += Vector((1,2,1))

        #print("coords::")
        getAllCoords(ob)
        # Get context elements: scene, camera and mesh
        scene = bpy.context.scene
        cam = bpy.data.objects['Camera']
        #obj = bpy.data.objects['cube']
        visible_vertices = getVisibleVertices(ob, cam, scene)
        #data.append([visible_vertices])
        
        obj = np.asarray(coords)
        print(obj)
        nv = obj.shape[0]
        print(obj.shape)
        obj=(obj.T.reshape(3,nv))
        conn = np.tile(conn, (3,1))
        data.append(np.concatenate((obj,conn), axis=1))    

    return data    

def getVisibleVertices(obj, cam, scene):    
    # In world coordinates, get a bvh tree and vertices
    bvh, vertices = BVHTreeAndVerticesInWorldFromObj( obj )
    visible_vertices = []
    
    for i, v in enumerate( vertices ):
        #print("vertex #")
        #print(i)
        # Get the 2D projection of the vertex
        co2D = world_to_camera_view( scene, cam, v )
        #print(co2D)
        #print("\n")

        # By default, deselect it
        obj.data.vertices[i].select = False

        # If inside the camera view
        if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0: 
            #print(i,"this vertex is inside camera view")
            # Try a ray cast, in order to test the vertex visibility from the camera
            location, normal, index, distance = bvh.ray_cast( cam.location, (v - cam.location).normalized() )
            #print("cam.location, (v - cam.location).normalized() , location, normal, index, distance")
            #print(cam.location, (v - cam.location).normalized() , location, normal, index, distance)
            # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
            if location and (v - location).length < limit:
                obj.data.vertices[i].select = True
                #print("selected vertex is:", i)
                #print(getCoords(obj.data.vertices[i]))
            visible_vertices.append([v.x,v.y,v.z])
        #print("\n\n")        
    print("#verts:", NUMVERTS, " #visible vertices:", len(visible_vertices))
    del bvh
    print("visible vertices:",visible_vertices)
    return visible_vertices


def camera_shots(a):
    cam = bpy.data.objects['Camera']
    origin = bpy.data.objects['cube']

    step_count = 3

    for step in range(0, step_count):
        origin.rotation_euler[a] = radians(step * (360.0 / step_count))
        #print(math.degrees(origin.rotation_euler[2]))
        bpy.data.scenes["Scene"].render.filepath = PATH + 'rot_xyz_%d_%d_%d.png' % (math.degrees(origin.rotation_euler[0]), math.degrees(origin.rotation_euler[1]), math.degrees(origin.rotation_euler[2]))
        bpy.ops.render.render( write_still=True )

PATH="/Users/pmishra/Library/Mobile Documents/com~apple~CloudDocs/blender_scripts/3D vision/Pilot Test Stimuli/New Obj/"
NUMVERTS = 4
# Threshold to test if ray cast corresponds to the original vertex
limit = 1     
data = createNObjects(1000)

#print(data)

#camera_shots(0)
#camera_shots(1)
#camera_shots(2)
    


import pickle
with open(PATH + 'data.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(list(data), f)