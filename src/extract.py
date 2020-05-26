import os
import json
import math
import numpy as np
import multiprocessing
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull
from shapely.geometRotation_matrix_y import Polygon



def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper                                                         #truncate large decimals to desired digits after decimal point

def distance_between_points(f,p):
    return math.sqrt(((f[0]-p[0])*(f[0]-p[0]))+((f[1]-p[1])*(f[1]-p[1]))+((f[2]-p[2])*(f[2]-p[2])))      #distance between two points


def lerp(v0, v1, i):
    return v0 + i * (v1 - v0)                                                                                   #getting count of equidistant points on faces

def getEquidistantPoints(p1, p2, n):
    return [(lerp(p1[0],p2[0],1./n*i), lerp(p1[1],p2[1],1./n*i),lerp(p1[2],p2[2],1./n*i)) for i in range(n+1)]      #getting equidistant points on face


def extract_image(face):
    new_face=[]

    for f in range(len(face)-1):
        new_points=getEquidistantPoints(face[f],face[f+1],int(distance_between_points(face[f],face[f+1])//0.1+1))
        new_face=new_face+new_points

    new_points=getEquidistantPoints(face[0],face[-1],int(distance_between_points(face[0],face[-1])//0.1+1))               #Divide face into multiple smaller pieces
    new_face=new_face+new_points

    pose_path=os.listdir('pose')                                                                                         #Get path of the poses provided ofr the images
    image_path=os.listdir('rgb')                                                                                         #Get path of the spherical images from which the mesh was obtained
    distance=10
    for pose in pose_path:
        if pose.endswith('.json'):                                      
            average_distance=0
            pose_tracking=json.load(open('pose/'+pose))
            position=pose_tracking['camera_location']
            position_x,position_y,position_z=float(position[0]),-float(position[1]),float(position[2])                   #extract x,y,z values from pose 
            camera_point=[position_x,position_y,position_z]
            for f in face:
                average_distance=average_distance+distance_between_points(f,camera_point)                                #Get nearest point to the boundaries provided
            average_distance=average_distance/len(face)
            if average_distance<distance:
                distance=average_distance
                cam_pose=pose_tracking




    for img in image_path:
        if cam_pose['camera_uuid'] in img:
            image=Image.open('rgb/'+img)
    position=cam_pose['camera_location']
    rotation=cam_pose['final_camera_rotation']
    position_x,position_y,position_z=float(position[0]),-float(position[1]),float(position[2])
    rotation_x,rotation_y,rotation_z=float(rotation[0])+math.pi,float(rotation[1]),float(rotation[2])                          #Extract the rotation angles of the closest image

    """ Calculate Rotation matrix for x,y and z """
    Rotation_matrix_x=np.array([[1,0,0],[0,math.cos(rotation_x),-math.sin(rotation_x)],[0,math.sin(rotation_x),math.cos(rotation_x)]])
    Rotation_matrix_y=np.array([[math.cos(rotation_y),0,math.sin(rotation_y)],[0,1,0],[-math.sin(rotation_y),0,math.cos(rotation_y)]])
    Rotation_matrix_z=np.array([[math.cos(rotation_z),-math.sin(rotation_z),0],[math.sin(rotation_z),math.cos(rotation_z),0],[0,0,1]])

    Rotation_matrix=np.matmul(np.matmul(Rotation_matrix_x,Rotation_matrix_y),Rotation_matrix_z)                                    #Get final rotation matrix
    polygon=[]
    for f in range(len(new_face)):
        world_coordinates=np.array([[new_face[f][0]-position_x],[new_face[f][1]-position_y],[new_face[f][2]-position_z]])           # obtain world coordinates wrt camera
        Rotation_matrix=np.matmul(Rotation_matrix,world_coordinates).tolist()                                                       #Multiply rottion matrix and world coordinates
        camera_x,camera_y,camera_z=Rotation_matrix[0][0],Rotation_matrix[1][0],Rotation_matrix[2][0]                                #Extract camera coordinates
        sq=math.sqrt((camera_x*camera_x)+(camera_y*camera_y)+(camera_z*camera_z))
        camera_x,camera_y,camera_z=camera_x/sq,camera_y/sq,camera_z/sq
        phi=math.asin(camera_y)                                                                                                     #Calculate phi and theea for spherical image
        rotation_x=math.asin(camera_x/math.cos(phi))
        rotation_y=math.asin(camera_x/math.cos(math.pi-phi))
        """ Verify the spherical maera theta nad phi and convert them to pixel positions """
        if truncate(math.cos(phi)*math.cos(rotation_x),4)==truncate(camera_z,4) and truncate(math.cos(phi)*math.sin(rotation_x),4)==truncate(camera_x,4) :
            camera_x=int(540*rotation_x+2048)
            camera_y=int(540*phi+1024)
        elif truncate(math.cos(phi)*math.cos(math.pi-rotation_x),4)==truncate(camera_z,4) and truncate(math.cos(phi)*math.sin(math.pi-rotation_x),4)==truncate(camera_x,4) :
            camera_x=int(540*(math.pi-rotation_x)+2048)
            camera_y=int(540*phi+1024)
        elif truncate(math.cos(math.pi-phi)*math.cos(rotation_y),4)==truncate(camera_z,4) and truncate(math.cos(math.pi-phi)*math.sin(rotation_y),4)==truncate(camera_x,4) :
            camera_x=int(540*(rotation_y)+2048)
            camera_y=int(540*(math.pi-phi)+1024)
        elif truncate(math.cos(math.pi-phi)*math.cos(math.pi-rotation_y),4)==truncate(camera_z,4) and truncate(math.cos(math.pi-phi)*math.sin(math.pi-rotation_y),4)==truncate(camera_x,4):
            camera_x=int(540*(math.pi-rotation_y)+2048)
            camera_y=int(540*(math.pi-phi)+1024)
        polygon.append([camera_x,camera_y])

    convex_hull=ConvexHull(polygon)                                                                                                        #Get convex hull of obtained pixels
    sp_xy=[]
    for h_v in convex_hull.vertices:                                                                                                       #Extracting coordinates of hull to an array
        sp_xy.append(polygon[h_v])
    polygon_xy=Polygon(sp_xy)
    coordinates=list(polygon_xy.exterior.coords)
    polygon=[]
    for c in range(len(coordinates)-1):
        polygon.append((coordinates[c][0],coordinates[c][1]))
    Image_array = np.asarray(image)
        
    Image_mask = Image.new('L', (Image_array.shape[1], Image_array.shape[0]), 0)                                                                #Create mask for empty image
    ImageDraw.Draw(Image_mask).polygon(polygon, outline=1, fill=1)                                                                      #Draw the polygon using pixels as coordinates
    mask = np.array(Image_mask)

    shape=(Image_array.shape[0],Image_array.shape[1],4)
    new_Image_Array = np.empty(shape,dtype='uint8')                                                                              #Create empty image using old image dimensions
    new_Image_Array[:,:,:3] = Image_array[:,:,:3]                                                                                            #Copy rgb values from old image
    new_Image_Array[:,:,3] = mask*255                                                                                                    #Copy mask to new image
    new_image = Image.fromarray(new_Image_Array, "RGBA")                                                                                     #Create image from array
    datas = new_image.getdata()                                                                                                         #Extract dats from new image

    new_data = []
    for item in datas:
        if item[3]==0:                                                                                                              #Check if alpha value is 0
            new_data.append((0, 0, 0, 0))                                                                                            #Convert pixel to black
        else:
            new_data.append(item)

    new_image.putdata(new_data)                                                                                                           #Convert cahnged dats to image
    new_image.saverage_distancee('extracted_images/'+str(face[0])+'.png',optimize=True)#Saverage_distancee image