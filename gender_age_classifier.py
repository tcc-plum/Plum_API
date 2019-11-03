# https://data-flair.training/blogs/python-project-gender-age-detection/

import cv2
import random

faceProto="models/gad/opencv_face_detector.pbtxt"
faceModel="models/gad/opencv_face_detector_uint8.pb"
ageProto="models/gad/age_deploy.prototxt"
ageModel="models/gad/age_net.caffemodel"
genderProto="models/gad/gender_deploy.prototxt"
genderModel="models/gad/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Homem','Mulher']


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


def DetectGenderAge(image, padding=20):
    #image = cv2.imread(imagem, cv2.IMREAD_COLOR)
    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)
    
    resultImg,faceBoxes=highlightFace(faceNet,image)
    
    for faceBox in faceBoxes:
        face=image[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,image.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, image.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        gender_est = float(genderPreds.item(genderPreds[0].argmax())*100)
        gender_est = round(gender_est, 2)
        #print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        ageEstimates = age[1:-1].split("-")
        ageEstimates = [int(age) for age in ageEstimates]
        age = random.randrange(ageEstimates[0], ageEstimates[1])
        age_est = float(agePreds.item(agePreds[0].argmax())*100)
        age_est = round(age_est, 2)
        #print(f'Age: {age} anos')
        
        return gender, gender_est, age, age_est