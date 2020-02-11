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
from mathutils import Matrix 
from bmesh.types import BMVert
from bmesh.types import BMEdge
from bmesh.types import BMFace
import pickle

def get_mat():
    # Get material
    mat = bpy.data.materials.get("Material.001")
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name="Material.001")
        mat.use_nodes=True
        
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    bsdf_node_1 = nodes.new(type='ShaderNodeBsdfToon')
    bsdf_node_1.name = "bsdf1"
    bsdf_node_1.inputs[0].default_value=(0.6,0.6,0.6, 1.0)
    bsdf_node_1.component="GLOSSY"
    bsdf_node_1.inputs[1].default_value=0.5
    bsdf_node_1.inputs[2].default_value=0.5

    normal_node = nodes.new(type='ShaderNodeGeometry')
    tree.links.new(normal_node.outputs[3], bsdf_node_1.inputs[3])
    return mat

def store_real_verts_ids(verts):
    ids = []
    for v in verts:
        ids.append(v.index)
    #print("real ids: ", ids)    
    return ids

def add_new_obj(randomize, cursor_pos, subdivide, merge_dist_mirror, merge_dist_symm, size=((1,1,1)), loc=((0,0,0))):
    
    bpy.ops.mesh.primitive_cube_add(location=loc,enter_editmode=True)
    bpy.context.object.rotation_euler[0]=1.5708

    
    #if(subdivide):
    #    bpy.ops.mesh.subdivide(number_cuts=1, smoothness=0, seed=1)

    for i in range(randomize):
        bpy.ops.transform.vertex_random()
        #bpy.ops.mesh.vertices_smooth()

    context = bpy.context
    scene = context.scene
    ob = context.edit_object
    me = ob.data
    
    bm = bmesh.from_edit_mesh(me)
    bpy.context.scene.cursor_location = cursor_pos
    pp = ob.matrix_world.inverted() * scene.cursor_location

    ret = bmesh.ops.mirror(bm, 
            geom=bm.faces[:] + bm.verts[:] + bm.edges[:],
            axis=0,  # x, y, z <==> 0, 1, 2
            matrix=Matrix.Translation(-pp), 
            merge_dist=merge_dist_mirror # disable so center verts dont merge.
            )
    new_geom = ret["geom"]
    
    verts=[ele for ele in new_geom if isinstance(ele, bmesh.types.BMVert)]
    edges=[ele for ele in new_geom if isinstance(ele, bmesh.types.BMEdge)]
    faces=[ele for ele in new_geom if isinstance(ele, bmesh.types.BMFace)]
    
    #print("BEFORE verts:", verts)
    ids = store_real_verts_ids(verts)
    #print("edges:", edges)
    #print("faces:", faces)
    
    
    #bpy.ops.mesh.vertices_smooth()
    #ret = bmesh.ops.symmetrize(bm, input=bm.faces[:] + bm.verts[:] + bm.edges[:], direction=0, dist=merge_dist_symm)
    #ret = bmesh.ops.symmetrize(bm, input=bm.faces[:] + bm.verts[:] + bm.edges[:], direction=1, dist=merge_dist_symm)
    #bpy.ops.mesh.vertices_smooth()
    
    new_geom = ret["geom"]
    
    verts=[ele for ele in new_geom if isinstance(ele, bmesh.types.BMVert)]
    edges=[ele for ele in new_geom if isinstance(ele, bmesh.types.BMEdge)]
    faces=[ele for ele in new_geom if isinstance(ele, bmesh.types.BMFace)]
    
    #print("AFTER verts:", verts)
    #print("edges:", edges)
    #print("faces:", faces)
    
    bmesh.update_edit_mesh(me)

    mat = get_mat()            
    # Assign it to object
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = mat
    else:
        # no slots
        ob.data.materials.append(mat)
        
    bpy.context.object.scale=size    
       
    bpy.ops.object.mode_set(mode='OBJECT')
    return ob,ids



# Create a BVH tree and return bvh and vertices in world coordinates 
def BVHTreeAndVerticesInWorldFromObj( obj ):
    mWorld = obj.matrix_world
    vertsInWorld = [mWorld * v.co for v in obj.data.vertices]
    bvh = BVHTree.FromPolygons( vertsInWorld, [p.vertices for p in obj.data.polygons] )
    return bvh, vertsInWorld

def isVisibleEdge(e, vv_ids):    
    if (e.verts[0].index in vv_ids and e.verts[1].index in vv_ids):
        #print("edge visible:", e)
        return True
    else:    
        #print("edge not visible", e)
        return False

def getVisibleVertices(obj, ids, cam, scene):    
    # In world coordinates, get a bvh tree and vertices
    bvh, vertices = BVHTreeAndVerticesInWorldFromObj(obj)
    
    #print("vertices:",vertices)
    #print("obj.data.vertices: ",obj.data.vertices)
    visible_vertices_cood = []
    visible_vertices_id = []
    limit = 0.1
    
    for i, v in enumerate(vertices):
        #print("vertex #")
        #print(i)
        
        if (i not in ids):
            continue
        
        # Get the 2D projection of the vertex
        co2D = world_to_camera_view(scene,cam,v)
        #print("co 2d: ",co2D)
        #print("\n")
        
        bpy.ops.mesh.primitive_cube_add(location=(v))
        bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))

        # By default, deselect it
        obj.data.vertices[i].select = False

        # If inside the camera view
        if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0: 
            #print(i,"this vertex is inside camera view")
            # Try a ray cast, in order to test the vertex visibility from the camera
            location= scene.ray_cast( cam.location, (v - cam.location).normalized() )
            if location[0] and (v - location[1]).length < limit:
                obj.data.vertices[i].select = True
                #print("selected vertex is:", i)
                visible_vertices_id.append(i)
                visible_vertices_cood.append([v.x,v.y,v.z])
                #bpy.ops.mesh.primitive_uv_sphere_add(size=0.02,location=(v.x,v.y,v.z),enter_editmode=False)
    del bvh
    #print("# visible vertices: ", len(visible_vertices_id), "ids: ", visible_vertices_id)
    return visible_vertices_cood, visible_vertices_id


def get_visible_conn_matrix(ob, vverts):
    me = ob.data    
    # Get a BMesh representation
    bm = bmesh.new()   # create an empty BMesh
    bm.from_mesh(me)   # fill it in from a Mesh

    conn=[]
    N=len(vverts)
    vert_id = []
    
    for v in bm.verts:
        if v.select:
            #print("vertex index:", v.index)
            vert_id.append(v.index)
         
    for j in range(1,N):
        for i in range(N-j):
            F=0
            i1=vert_id[i]
            i2=vert_id[i+j]
            #print ("search for:", i1,i2)
            for edge in bm.edges:
                if (edge.verts[0].index == i1 and edge.verts[1].index == i2) or (edge.verts[0].index == i2 and edge.verts[1].index == i1):
                    #print ("Y:",edge.verts[0].index, edge.verts[1].index)
                    me.edges[edge.index].use_freestyle_mark = True
                    F=1
            conn.append(F)   
    bpy.ops.object.mode_set(mode='OBJECT')        
    return conn  



def rotate_obj(ob, axis):
    cam = bpy.data.objects['Camera']
    origin = ob #bpy.data.objects['Cube']
    step_count = 5
    for step in range(0, step_count):
        #print("step: ", step)
        origin.rotation_mode = 'XYZ'
        origin.rotation_euler[axis] = radians(step * (90.0 / step_count))



def write(ids, data):
    scene = bpy.context.scene
    cam = bpy.data.objects['Camera']
    obj = bpy.data.objects['Cube']
    vv_coords, vv_ids = getVisibleVertices(obj, ids, cam, scene)
    
    #print("visible vtx:", visible_vertices)
    conn = get_visible_conn_matrix(obj, vv_coords)
    #print("connection mat:", conn)
    vverts = np.asarray(vv_coords)
    nv = vverts.shape[0]
    vverts = (vverts.T.reshape(3,nv))
    #print("vverts shape:", vverts.shape)
    conn = np.tile(conn, (3,1))
    #print("conn shape:", conn.shape)
    angles,degrees = compute_edge_angles_degrees(ob,vv_ids)
    #print("angles SDA: ", np.std(angles))
    data.append(vverts) 
    data.append(conn)
    data.append(degrees)
    data.append(angles)
    data.append(np.std(angles))    
    
    
def remove_objs():
    # gather list of items of interest.
    candidate_list = [item.name for item in bpy.data.objects if item.type == "MESH"]
    #print("candidate list", candidate_list)
    # select them only.
    for object_name in candidate_list:
      bpy.data.objects[object_name].select = True
    # remove all selected.
    bpy.ops.object.delete()
    # remove the meshes, they have no users anymore.
    for item in bpy.data.meshes:
      bpy.data.meshes.remove(item)
        
    
def compute_edge_angles_degrees(ob,ids):
    me = ob.data    
    # Get a BMesh representation
    bm = bmesh.new()   # create an empty BMesh
    bm.from_mesh(me)   # fill it in from a Mesh
    all_angles = []
    all_degrees = np.zeros_like(ids)
    
    for i, id in enumerate(ids):
        edges = []
        prev_edge=0
        #print("Searching all edges for vert: ", id)
        
        for edge in bm.edges:
            if (edge.verts[0].index == id or edge.verts[1].index == id):
                #print("edge found:", edge)
                edges.append(edge)
                if (isVisibleEdge(edge,ids)):
                    all_degrees[i]+=1    
            else:
                continue  
            
        for edge in edges:        
            if (prev_edge == 0):
                prev_edge = edge
                continue
            #print("edge pair: ", prev_edge, edge) #edge.verts[0].index
            v00 = edge.verts[0]     
            v01 = edge.verts[1]    
            v10 = prev_edge.verts[0]     
            v11 = prev_edge.verts[1]
            v0 = Vector(v00.co) - Vector(v01.co)
            v1 = Vector(v10.co) - Vector(v11.co)
            #print("Angle: ", math.degrees(v0.angle(v1))) 
            all_angles.append(v0.angle(v1))   
            prev_edge = edge
            
    #print("All degrees: ",all_degrees)        
    return all_angles, all_degrees         
                   
print("*******start script********")
PATH="/Users/pallavimishra/3DVisionOpNet/training/vverts/cube/"

data=[]

#N=1999
for i in range(1000):
    remove_objs()
    randomize = 1 #random.randint(25,30)
    cursor_pos = (0,0,0) #((random.random()*10, random.random()*10, random.random()*10))
    subdivide = False
    merge_dist_mirror = 0.1
    merge_dist_symm = 0.1
    size = ((random.random()*1, random.random()*1, random.random()*1))
    #print("random_times, cursor_pos, subdivide, merge_dist_mirror, merge_dist_symm,size :",randomize,cursor_pos,subdivide,merge_dist_mirror,merge_dist_symm,size)
    ob,ids = add_new_obj(randomize,cursor_pos,subdivide,merge_dist_mirror,merge_dist_symm,size)
    #rotate_obj(ob, 1)
    #rotate_obj(ob, 2)
    angles,degrees = compute_edge_angles_degrees(ob,ids)
    write(ids, data)
    

print("model data:", data)
print("len data:", len(data))

with open(PATH + '1000_cubes.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(list(data), f)
   
          