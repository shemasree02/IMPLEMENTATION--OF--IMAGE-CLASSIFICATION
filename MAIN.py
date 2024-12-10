import cv2
from google.colab.patches import cv2_imshow

# Provide the full path to the XML file if it's not in the current working directory.
# You can download it to the current directory or your Google Drive and mount it.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  

img = cv2.imread('/content/PSPK_DARLING.JPEG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(img)

faces_result = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces_result:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    cv2_imshow(img)

eye = eye_detector.detectMultiScale(roi_gray)
for (ex, ey, ew, eh) in eye:
    img = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
cv2_imshow(img)
