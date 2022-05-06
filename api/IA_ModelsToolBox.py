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
confidence_threshold = 0.2
IA_Model_path = '/home/emfs/IA_Models' #Change this!

def get_seedling_class(modelpath, image_input):
    print("Begin PREDICTION PROCESS")
    weight_path = os.path.join(IA_Model_path, 'yolov3_training_last.weights')
    cfg_path = os.path.join(modelpath, 'yolov3_testing.cfg')
    net = cv2.dnn.readNet(weight_path, cfg_path)
    print("Yolov3 loaded")

    classes = []
    with open(os.path.join(modelpath,"classes.txt"), "r") as f:
        classes = f.read().splitlines()
    
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    
    #print(f"The image path is: {image_input}")
    img = cv2.imread(image_input)
    #print(f"The img is: {img}")
        
    height, width, c = img.shape
    print("Current image size:")
    print("height: " + str(height) + ", width: " + str(width))
    if width > iimg_width or width < iimg_width or height > iimg_height:
        dim = (iimg_height, iimg_width)
        img = cv2.resize(img, dim)
        height, width, _ = img.shape
    print("New image size:")
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
            
            if confidence > confidence_threshold:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                if center_x > g1min and center_x <= g1max and center_y >= hmin and center_y <= hmax:
                    #1st group region
                    PM[idx[0],0,:] = [classify_pred([x, y, w, h],confidence),x,y,w,h,confidence]
                    idx[0] += 1
                if center_x > g2min and center_x <= g2max and center_y >= hmin and center_y <= hmax:
                    #2nd group region
                    PM[idx[1],1,:] = [classify_pred([x, y, w, h],confidence),x,y,w,h,confidence]
                    idx[1] += 1
                if center_x > g3min and center_x <= g3max and center_y >= hmin and center_y <= hmax:
                    #3th group region
                    PM[idx[2],2,:] = [classify_pred([x, y, w, h],confidence),x,y,w,h,confidence]
                    idx[2] += 1

                
                #pred.append([classify_pred([x, y, w, h],confidence),x,y,w,h,confidence])

        #Clean predictions, filter out that ones innecesary preds at the same region. That is, one or none prediction per region
        for pidx in range(3):
            val = -1
            #print(idx[pidx])
            if idx[pidx] > 0: #if there is 0 value at position it must be recognized as no seedling pred
                for i in range(idx[pidx]):
                    #print("i: ", i)
                    c = PM[i,pidx,0] #this stage goes for discrimination in case more than one seedling was detected and classified at the same region
                    if c > val: #the best class is our guy!
                        indx = i
                        val = c
                
                #print(f"PM {pidx}: {PM[indx,pidx,:]}")
                boxes[pidx] = PM[indx,pidx,1:5]
                confidences[pidx] = float(PM[indx,pidx,5])
                class_ids[pidx] = PM[indx,pidx,0]

    for i in range(3):
        pred.append([class_ids[i],boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3],confidences[i]])
                
    #for i, p in enumarate(pred):
    #    x, y, w, h = p[i][1:4]
    #    label = p[0]
    #    confidence = str(round(p[-1],2))
    #    color = colors[i]
    #    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    #    cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)

    return pred

def classify_pred(box, score): #classifications images according to calibrated thresholds
    #print("Classifying predictions done by yolov3")    
    x, y, w, h = box
    area = w * h
    
    if score < score_threshold: #It is suspected that it is not average seedling
        #then it is necessary to know if it is bad or good seedling
        if area < left_threshold:
            #bad seedling
            return 1 #"bad" #1#0
        if area > right_threshold:
            #good seeling
            return 3 #"good" #3#2
        else:
            return 2 #"avrg" #2#1  #TO BE CONSIDERED!
    else:
        #at least that it has a big area
        if area <= right_threshold: #area > left_threshold and area < right_threshold: #OJOOOO!! take it into account
            #average seedling
            return 2 #"avrg" #2#1
        if area > right_threshold:
            #good seedling
            return 3 #"good" #3#2
        #else:
        #    return 1#0

    print("no pasa")
    return "unknown"
