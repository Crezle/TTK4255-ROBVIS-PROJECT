import cv2
import urllib.request
import serial
import json
import numpy as np
import time
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
 
url='http://192.168.127.18/cam-hi.jpg'
cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
 
count=0
 
 
gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.txt")
if gauth.credentials is None:
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()


gauth.SaveCredentialsFile("mycreds.txt")
drive = GoogleDrive(gauth)
folder ="11k4ixREv9L3b-vGr5mtqtC6-8zjIDuPV"
file_id = '1_-qiJ0x9Vzm5BJuIB8POhLZ_zmxKapYy'
file_obj = drive.CreateFile({'id': file_id})
file_obj.GetContentFile('data.json')
image = cv2.imread("sample_image.jpg")
height, width = image.shape[:2]
print(f'Resolution of sample image: {width}x{height}')

 
while True:
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgnp,-1)
    
    cv2.imshow("live transmission", frame)
 
    key=cv2.waitKey(5)
    
    if key==ord('k'):
        count+=1
        t=str(count).zfill(4)+'.jpg'
        cv2.imwrite(t,frame)
        print("image saved as: "+t)
        f= drive.CreateFile({'parents':[{'id':folder}],'title':t})
        f.SetContentFile(t)
        #f.Upload()
        print("image uploaded as: "+t)
        image = cv2.imread(t)
        height, width = image.shape[:2]
        print(f'Resolution: {width}x{height}')
        
    if key==ord('q'):
        break
    if key==ord('s'):
        with open('data.json', 'r') as file:
            json_data = file.read()

        print(json_data)
        ser = serial.Serial('/dev/ttyUSB0', 115200) 
        
        ser.write(json_data.encode())

        print("Data sent to ESP32CAM")
        ser.close()
    else:
        continue
 
cv2.destroyAllWindows()