import cv
 
HAAR_CASCADE_PATH = "/usr/local/Cellar/opencv/2.4.2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
CAMERA_INDEX = 0
 
def detect_faces(image):
	tup = (image.width, image.height)
	gr = cv.CreateImage(tup,8,1)
	cv.CvtColor(image,gr,cv.CV_BGR2GRAY)
	cv.EqualizeHist(gr,gr)
	faces = []
	detected = cv.HaarDetectObjects(gr, cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (100,100))
	if detected:
		for (x,y,w,h),n in detected:
			faces.append((x,y,w,h))
	return faces
 
if __name__ == "__main__":
    cv.NamedWindow("Video", cv.CV_WINDOW_AUTOSIZE)
 
    capture = cv.CaptureFromCAM(CAMERA_INDEX)
    storage = cv.CreateMemStorage()
    cascade = cv.Load(HAAR_CASCADE_PATH)
    faces = []
 
    i = 0
    while True:
        image = cv.QueryFrame(capture)
        # Only run the Detection algorithm every 5 frames to improve performance
        if i%3==0:
            faces = detect_faces(image)
        mask = cv.LoadImage("black-mustache-hi.png")
        for (x,y,w,h) in faces:
        	img_mask = cv.CreateImage((w,h/6),mask.depth,mask.nChannels)
        	cv.SetImageROI(image,(x,y + (h/6)*4,w,h/6))
        	cv.Resize(mask,img_mask,cv.CV_INTER_LINEAR)
        	cv.Sub(image,img_mask,image)
        	cv.ResetImageROI(image)
            # cv.Rectangle(image, (x,y), (x+w,y+h), 255)
        cv.Flip(image, flipMode=1)
        cv.ShowImage("w1", image)
        i += 1
