import cv2
MODEL_MAIN_VALUE = ( 78.4263377603, 87.7689143744, 114.895847746 )
age_list =['0,2','4,6','8,12','15,20','25,32','38,43','48,53',
           '60,100']
def initialize_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe(
        'age_deploy.prototxt','age_net.caffemodel')
    return (age_net)
def red_from_image(age_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.imread("IMG_20171206_110610747.jpg")
    face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if(len(faces)>0):
        print("found {} faces".format(str(len(faces))))
        for ( x , y , w , h)in faces:
            cv2.rectangle(image ,(x,y),(x+w,y+h),(255,255,0),2)
            
            face_img = image[y:y+h,h:h+w].copy()
            blob = cv2.dnn.blobFromImage(face_img,1,
                    (227,227),MODEL_MAIN_VALUE,swapRB=False)
            
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("age range" + age)
            overlay_text = (age)
            cv2.putText(image,overlay_text, (x,y),
            font, 0.5 , (100,100,225), 2,cv2.LINE_AA  )
            cv2.imshow("",image)
            cv2.waitKey(0)
            #main
if __name__=="__main__":
    age_net = initialize_caffe_models()
    red_from_image(age_net)
            
    
    
