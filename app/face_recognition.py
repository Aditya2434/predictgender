import cv2
import numpy as np
import sklearn
import pickle

##LOAD MODEL

harr = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
model_svm = pickle.load(open('./model/model_svm.pickle', mode ='rb'))
pca_models = pickle.load(open('./model/pca_dict.pickle',mode ='rb'))

model_pca = pca_models['pca']
mean_face_arr = pca_models['mean_face']



def faceRecognitionPipeline(filename,path=True):
    if path:
        img = cv2.imread(filename)
       
    else:
        img = filename

    #02. convert into gray scale

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #03. crop he face using harr cascade classifier

    faces = harr.detectMultiScale(gray , 1.5,3)
    predictions = []

    for x,y,w,h in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        roi= gray[y:y+h,x:x+w]

        #04. normalization
        roi = roi/255.0
        #05. resize images
        if roi.shape[1]>100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)

        #06. flattening
        roi_reshape = roi_resize.reshape(1,10000)
        #07. substract with mean
        roi_mean = roi_reshape - mean_face_arr

        #08. get eigen images
        eigen_image = model_pca.transform(roi_mean)

        #09. Eigen Images for visualization
        eig_img = model_pca.inverse_transform(eigen_image)

        #10. pass to ml model to prediction 
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()

        #11. generate report
        text = "%s : %d"%(results[0], prob_score_max*100)
        print(text)

        #COLOR FOR MALE AND FEMALE

        if results[0]=='male':
            color = (255,255,0)
        else:
            color = (255,0,255)

        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),5)

        output = {
            'roi':roi,
            'eig_img': eig_img,
            'prediction_name':results[0],
            'score':prob_score_max
        }

        predictions.append(output)
        
    return img ,predictions        