import os
import cv2
import json
import numpy as np
from PIL import Image
import multiprocessing
from src.model import get_dilated_unet

##############################################################################################################################




def image_classification():
    blue = [255,0,0]
    green=[0,255,0]
    red=[0,0,255]
    white=[255,255,255]
    image_input=os.listdir('extracted_images')                                                                           #Create lsit of extracted images

    model = get_dilated_unet(input_shape=(1024,1024, 3), mode='cascade', filters=32,n_class=4)                                                        #Initalize CNN
    model.load_weights('./models/walls_model_weights.hdf5')                                                                                         # Add weights to the network

    new_arr=[]
    for img in image_input:                                                                                                                          #Take each image
        if not(os.path.exists('classified_images/'+img)):                                                                                           
            im = Image.open('extracted_images/'+img)
            x_train = np.array(im,'f')
            transparent_indices=np.argwhere(x_train[:,:,3]==0)                                                                                      #Create list of all transparent indices
            transparent_indices=transparent_indices.tolist()
            x=np.zeros((x_train.shape[0],x_train.shape[1],3))                                                                                       #create an empaintingy image with same dimensions as original
            width=0
            img_size=1024
            while width<x_train.shape[1]:                                                                                                           #Crop image to remove memory errors
                height=0
                while height<x_train.shape[0]:
                    x_traintest=np.reshape(x_train[height:height+img_size,width:width+img_size,0:3],(1,1024,1024,3))                                  #reshape data for cnn
                    x_train1=model.predict(x_traintest)                                                                                             #predict on the image
                    y=np.zeros((1024,1024,3))                                                                                                         #Create empaintingy array with dimensions of cropped image
                    for i in range(0,1024):                                                                                                          #Extract the predicted values as an image
                        for j in range(0,1024):
                            if(x_train1[0][i][j][0] > 0.50):
                                y[i][j] = blue
                            elif(x_train1[0][i][j][1] > 0.50): 
                                y[i][j] = green
                            elif(x_train1[0][i][j][2] > 0.50): 
                                y[i][j] = red
                            else:
                                y[i][j] =white

                    x[height:height+img_size,width:width+img_size,:3]=y                                                                            #Add predicted data to full size empaintingy image
                    height=height+1024
                width=width+1024
            for i in transparent_indices:                                                                                                          #Convert transparent indices from list to black color
                x[i[0],i[1]]=[0,0,0]
            cv2.imwrite('classified_images/'+img,x)
            
        else:
            x=cv2.imread('classified_images/'+img)


        brickwork= np.count_nonzero(np.all(x==red,axis=2))
        plastering=np.count_nonzero(np.all(x==green,axis=2))
        whitecoat=np.count_nonzero(np.all(x==blue,axis=2))
        painting=np.count_nonzero(np.all(x==white,axis=2))

        new_object={img:{'children':[{'name':'Brickwork','progress':brickwork*100/(brickwork+plastering+whitecoat+painting)},{'name':'plasteringastering','progress':plastering*100/(brickwork+plastering+whitecoat+painting)},{'name':'Whitecoat','progress':whitecoat*100/(brickwork+plastering+whitecoat+painting)},{'name':'Painting','progress':painting*100/(brickwork+plastering+whitecoat+painting)}]}}
        new_arr.append(new_object)
        print(brickwork,plastering,whitecoat,painting)

        print(img)
        print('Brickwork:%s',brickwork/(brickwork+plastering+whitecoat+painting))
        print('plasteringastering:%s',plastering/(brickwork+plastering+whitecoat+painting))
        print('Whitecoat:%s',whitecoat/(brickwork+plastering+whitecoat+painting))
        print('Painting:%s',painting/(brickwork+plastering+whitecoat+painting))

                                                                           

    return new_arr

def image_classification2():
    blue = [255,0,0]
    green=[0,255,0]
    red=[0,0,255]
    white=[255,255,255]
    image_input=os.listdir('sep_images')                                                                           #Create lsit of extracted images

    model = get_dilated_unet(input_shape=(1024,1024, 3), mode='cascade', filters=32,n_class=4)                                                        #Initalize CNN
    model.load_weights('./models/walls_model_weights.hdf5')                                                                                         # Add weights to the network

    new_arr=[]
    for img in image_input:                                                                                                                          #Take each image
        if not(os.path.exists('sepc_images/'+img)):                                                                                           
            im = Image.open('sep_images/'+img)
            wid,ht=im.size
            im=im.resize(((wid//1024+1)*1024,(ht//1024+1)*1024))
            x_train = np.array(im,'f')
            x=np.zeros((x_train.shape[0],x_train.shape[1],3))                                                                                       #create an empaintingy image with same dimensions as original
            width=0
            img_size=1024
            while width<x_train.shape[1]:                                                                                                           #Crop image to remove memory errors
                height=0
                while height<x_train.shape[0]:
                    x_traintest=np.reshape(x_train[height:height+img_size,width:width+img_size,0:3],(1,1024,1024,3))                                  #reshape data for cnn
                    x_train1=model.predict(x_traintest)                                                                                             #predict on the image
                    y=np.zeros((1024,1024,3))                                                                                                         #Create empaintingy array with dimensions of cropped image
                    for i in range(0,1024):                                                                                                          #Extract the predicted values as an image
                        for j in range(0,1024):
                            if(x_train1[0][i][j][0] > 0.50):
                                y[i][j] = blue
                            elif(x_train1[0][i][j][1] > 0.50): 
                                y[i][j] = green
                            elif(x_train1[0][i][j][2] > 0.50): 
                                y[i][j] = red
                            else:
                                y[i][j] =white

                    x[height:height+img_size,width:width+img_size,:3]=y                                                                            #Add predicted data to full size empaintingy image
                    height=height+1024
                width=width+1024
            x=cv2.resize(x, (wid,ht), interpolation = cv2.INTER_AREA)
            cv2.imwrite('sepc_images/'+img,x)
            
        else:
            x=cv2.imread('sepc_images/'+img)


        brickwork= np.count_nonzero(np.all(x==red,axis=2))
        plastering=np.count_nonzero(np.all(x==green,axis=2))
        whitecoat=np.count_nonzero(np.all(x==blue,axis=2))
        painting=np.count_nonzero(np.all(x==white,axis=2))

        new_object={img:{'children':[{'name':'Brickwork','progress':brickwork*100/(brickwork+plastering+whitecoat+painting)},{'name':'plasteringastering','progress':plastering*100/(brickwork+plastering+whitecoat+painting)},{'name':'Whitecoat','progress':whitecoat*100/(brickwork+plastering+whitecoat+painting)},{'name':'Painting','progress':painting*100/(brickwork+plastering+whitecoat+painting)}]}}
        new_arr.append(new_object)
        print(brickwork,plastering,whitecoat,painting)

        print(img)
        print('Brickwork:%s',brickwork/(brickwork+plastering+whitecoat+painting))
        print('plasteringastering:%s',plastering/(brickwork+plastering+whitecoat+painting))
        print('Whitecoat:%s',whitecoat/(brickwork+plastering+whitecoat+painting))
        print('Painting:%s',painting/(brickwork+plastering+whitecoat+painting))

                                                                           

    return new_arr






##############################################################################################################################

##############################################################################################################################

