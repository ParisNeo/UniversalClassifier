"""TextPinner
Author : Saifeddine ALOUI
Description : This uses UniversalClassifier to classify multiple video images from the webcam by classifying multiple 
regions of interest in order to have a general yolo-rcnn like classification system. It uses fasterrcnn to determine rigions of interest then uses universal classifier to 
classify the content of each box.

This example shows the interest of this approach. While RCNN can only detect if there is a person or a bock and several other classes. Using Universal classifier can help
having more detailed classification like the gender of the person and the color of the book for example.

Feel free to add more classes to universal classifier
Using gpu is very advised as RCNN is very heavy. In this example we use the RESNET50 implementation. You may test other networks from the torchvision library if you want
to have more or less speed/accuracy.

"""
from UniversalClassifier import UniversalClassifier
from PIL import Image
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

# Objects classifier. just execute the code and show objects that are in your list and it will classify them
uc = UniversalClassifier(["a male person","a female person","a red book","a green book","keys", "glass", "keyboard", "a watch","spoon"], 0.8)

# Now you come up with your own classifications.

# open camera
cap = cv2.VideoCapture(0)
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
rcnnmodel = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
rcnnmodel.eval()
preprocess = weights.transforms()

while cap.isOpened():
    # Read image
    success, cv_image = cap.read()
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    #ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    #ROIs = [[x1,y1,np.random.randint(x1+150,cv_image.shape[1]),np.random.randint(y1+150,cv_image.shape[0])] for i in range(5) for x1 in [np.random.randint(0,cv_image.shape[1]-150)] for y1 in [np.random.randint(0,cv_image.shape[0])-150] ]#cv2.selectROIs("Select Rois",cv_image)
    prediction = rcnnmodel([preprocess(image)])[0]
    ROIs = prediction["boxes"].detach().numpy()
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    # get regions of interest
    i=0
    for rect in ROIs:
        x1=int(rect[0])
        y1=int(rect[1])
        x2=int(rect[2])
        y2=int(rect[3])

        #crop roi from original image
        box = (x1, y1, x2, y2)
        region = image.crop(box) 
        
        #classify
        output_text, index, similarity=uc.process(region) # try other images red_apple, green_apple, yellow_apple
        #print(np.max(similarity))
        cv2.rectangle(cv_image, (x1,y1),(x2,y2),(255,0,0),2)
        # If the universal classifier figured it out, then use its estimation, else use the fasterrcnn output
        if index>=0:
            cv2.putText(cv_image, output_text + f" (sim={100*np.max(similarity):0.2f}%)", (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(cv_image, labels[i], (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        i += 1


    
    # Show the image
    cv2.imshow('Video', cv_image)
    # Wait for key stroke for 5 ms
    wk = cv2.waitKey(5)
    if wk & 0xFF == 27: # If escape is pressed then return
        break


