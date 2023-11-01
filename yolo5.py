import torch
import cv2
import numpy as np




def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)



model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
count=0
cap=cv2.VideoCapture('sample.mp4')

area= [(239,260),(217,433),(1000,422),(846,269),(239,260)]

while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    frame=cv2.resize(frame,(1020,500))
    results = model(frame)
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
#        print(d)
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        results= cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        print(results)
        if results>=0:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.3,(255,0,0),1)
            cv2.circle(frame,(cx,cy),5,(255,0,0),-1)
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,0,255),2)
    cv2.imshow("SRIDHAR",frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
#stream.release()
cv2.destroyAllWindows()
