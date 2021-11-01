# -*- coding:utf-8 -*-
#!/usr/bin/python3
import socket  
import cv2  
import numpy as np  
import os  
import time  

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def getimage():

    # 送信先のIPアドレスとポート番号を設定
    HOST = "192.168.0.0"
    PORT = 5569
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)  
    sock.connect((HOST,PORT))   
    sock.send(b'test')  

    # バイト型
    buf=b''   
    recvlen=100  
    while recvlen>0:  
        receivedstr=sock.recv(1024*8)  
        recvlen=len(receivedstr)  
        buf += receivedstr
    sock.close()

    narray=np.fromstring(buf,dtype='uint8')
    return cv2.imdecode(narray,1)

# while True:  
#     img = getimage()
#     cv2.imshow('Capture',img)  

#     # qを入力すれば処理終了
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    

model=load_model('./model/not_bad_model_3.h5')

# face
face_cascade_file = './haarcascade_frontalface_default.xml'  
front_face_detector = cv2.CascadeClassifier(face_cascade_file)  
# eye 
eye_cascade_file = './haarcascade_eye.xml'  
eye_detector = cv2.CascadeClassifier(eye_cascade_file)  


names = ['A','B','C','D']   

#cap = cv2.VideoCapture(0)  

#cap = cv2.VideoCapture("sugeno.mp4")

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  

# 最小Windowサイズを定義  
#minW = 0.1*cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
#minH = 0.1*cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  

while True:  

    
    tick = cv2.getTickCount()  

    img = getimage()
    
    #ret, img =cap.read()
    #print(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # face detect  
    faces = front_face_detector.detectMultiScale(   
        gray,  
        scaleFactor = 1.2,  
        minNeighbors = 3,  
        #minSize = (int(minW), int(minH)),  
       )  
    # predict person
    for(x,y,w,h) in faces:  

         
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)  

        #t1 = time.time()  
        #id ,confidence = recognizer.predict(gray[y:y+h,x:x+w]) 
        #print(confidence)
        face_cut = img[y:y+h, x:x+w]
        face_cut = cv2.resize(face_cut,(256,256))
        #face_cut = np.array(face_cut)
        #face_cut = face_cut[np.newaxis, :,:]
        
        b,g,r = cv2.split(face_cut)
        face_cut = cv2.merge([r,g,b])
        face_cut = face_cut[np.newaxis, :,:,:]
        #print(face_cut.shape)
        
        #Learning from scratch
        #id = model.predict_classes(face_cut, batch_size=1, verbose=0)
        #Transfer Learning
        predict_prob=model.predict(face_cut, batch_size=1, verbose=0)
        id =np.argmax(predict_prob,axis=1)
        #model.summary()
        confidence =  model.predict(face_cut)[0]
        print(confidence)
        #print(id)

        

        id_name = names[id[0]]
        confidence = confidence[id[0]]*100
        #
        
        #print(id)
        
        #t2 = time.time()  
         
        #dt1 = t2 - t1  

        
        ''' 
        if confidence < 60 and confidence > 0:  
              
            id = names[id]  
            confidence = "{0}%".format(round(100 - confidence))  
        else:  
            id = "unknown"  
            confidence = "{0}%".format(round(100 - confidence))  

        # addition 
        '''     
        cv2.putText(img, "confidence: " + str(confidence), (x+5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 2)           
        
        cv2.putText(img, str(id_name), (x+5,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)  

    # FPS算出と表示用テキスト作成  
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)  
    # FPS  
    cv2.putText(img, "FPS:{} ".format(int(fps)),   
        (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2, cv2.LINE_AA)  

      
    cv2.imshow('camera',img)   

    # ESC  
    k = cv2.waitKey(10) & 0xff   
    if k == 27:  
        break  

 
print("\n Exit Program")  
cap.release()  
cv2.destroyAllWindows()  
