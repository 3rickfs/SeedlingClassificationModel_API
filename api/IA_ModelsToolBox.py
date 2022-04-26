import os
import cv2
import numpy as np

#limits for every group of seedling predictions
g3min = 60
g3max = 175
g2min = 175
g2max = 275
g1min = 275
g1max = 400
hmin = 194
hmax = 232
score_threshold = 0.98
left_threshold = 2614
right_threshold = 4676
iimg_height, iimg_width = 416, 416
IA_Model_path = '~/IA_Models' #Change this!

def get_prediction(modelpath, image_input):
    print("begin PREDICTION PROCESS")
    weight_path = os.path.join(IA_Model_path, 'yolov3_training_last.weights')
    cfg_path = os.path.join(modelpath, 'yolov3_testing.cfg')
    net = cv2.dnn.readNet(weight_path, cfg_path)
    print("yolov3 loaded")

    classes = []
    with open(os.path.join(modelpath,"classes.txt"), "r") as f:
        classes = f.read().splitlines()
    
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    
    img = cv2.imread(image_input)
        
    height, width, c = img.shape
    if width > iimg_width or width < iimg_width or height > iimg_height:
    	img = img.resize(iimg_height, iimg_width, c)
        #img = cropimage(img,403,1082,443,720,0.5625) #according to 416x146pixel image size input for yolov3 model
        #img = imageremix(imageio.imread(os.path.join(modelpath,"size_ref.jpg")),img,int(679/2-416/2),int(416/2-277/2))
        height, width, _ = img.shape
    
    print("height: " + str(height) + ", width: " + str(width))
  
    print("Start processing image with Yolov3 model")
    blob = cv2.dnn.blobFromImage(img, 1/255, (iimg_height, iimg_width), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    
    boxes = [(0,0,0,0),(0,0,0,0),(0,0,0,0)]
    confidences = [0,0,0]
    class_ids = [0,0,0]
    
    PM = np.ones([10,3,6]) #*-1 #Position matrix
    idx = [0,0,0] #np.zeros([1,3]) #with zeros considering 0 position at the beginning
    pred = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                pred.append([classify_pred([x, y, w, h],confidence),x,y,w,h,confidence])


    
    #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    #print("indexes: ", indexes)
    #seedlfnames = [" "," "," "]
    
    #This is just for saving seedling images each one in different file
    if len(pred) > 0:
        for i, p in enumarate(pred):
            x, y, w, h = p[i][1:4]
            label = p[0]
            confidence = str(round(p[-1],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)
            
            crop_img = img[int(y):int(y+h), int(x):int(x+w)]
            now = datetime.now()
            dt_str = now.strftime("%d-%m-%Y_%H-%M-%S.%f")
            sfn = 'seedling_' + dt_str + '.png'
            dpath = os.path.join(seedldestpath,sfn)
            cv2.imwrite(dpath,crop_img)
            seedlfnames[i] = sfn
            #seedlfnames.append(sfn)
            
            #cv2.imshow("cropped", crop_img)
            #cv2.waitKey(0)

    #seedlfnames = np.array(seedlfnames)
    return seedlfnames, class_ids

def classify_pred(box, score): #classifications images according to calibrated thresholds
    print("Classifying predictions done by yolov3")    
    x, y, w, h = box
    area = w * h
    
    if score < score_threshold: #It is suspected that it is not average seedling
        #then it is necessary to know if it is bad or good seedling
        if area < left_threshold:
            #bad seedling
            return "bad" #1#0
        if area > right_threshold:
            #good seeling
            return "good" #3#2
        else:
            return "avrg" #2#1  #TO BE CONSIDERED!
    else:
        #at least that it has a big area
        if area <= right_threshold: #area > left_threshold and area < right_threshold: #OJOOOO!! take it into account
            #average seedling
            return "avrg" #2#1
        if area > right_threshold:
            #good seedling
            return "good" #3#2
        #else:
        #    return 1#0

    print("no pasa")
    return "unknown"

def start_processing_image(img):
    print("Processing image")
    sn,sc = get_prediction(modelpath, img)
    

    return sn, sc


