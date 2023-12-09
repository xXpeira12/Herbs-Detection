from tkinter import filedialog
from tkinter import *
from PIL import Image,ImageTk
import cv2
import numpy as np
import time

def detect_image():
	LABELS_FILE='coco.names'
	CONFIG_FILE='custom-yolov4-tiny-detector.cfg'
	WEIGHTS_FILE='custom-yolov4-tiny-detector_best.weights'
	CONFIDENCE_THRESHOLD=0.3
	LABELS = open(LABELS_FILE).read().strip().split("\n")
	INPUT_FILE = filedialog.askopenfilename()
	if len(INPUT_FILE) > 0:
		np.random.seed(4)
		COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
			dtype="uint8")
		net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
		image = cv2.imread(INPUT_FILE)
		(H, W) = image.shape[:2]
		# determine only the *output* layer names that we need from YOLO
		ln = net.getLayerNames()
		ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()
		#print("[INFO] YOLO took {:.6f} seconds".format(end - start))
		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []
		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > CONFIDENCE_THRESHOLD:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
			CONFIDENCE_THRESHOLD)
		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, color, 2)
		# show the output image
		cv2.imshow("image", image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
	
def detect_video():
	INPUT_FILE = filedialog.askopenfilename()
	if len(INPUT_FILE) > 0:
		cap = cv2.VideoCapture(INPUT_FILE)
		whT = 320
		confThreshold = 0.5
		nmsThreshold = 0.1

		classesFile = "coco.names"
		classNames = []
		with open(classesFile, 'rt') as f:
			classNames = f.read().rstrip('\n').split('\n')

		modelConfiguration ="custom-yolov4-tiny-detector.cfg"
		modelWeights = "custom-yolov4-tiny-detector_best.weights"

		net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

		def findObjects(outputs,img):
			hT, wT, cT = img.shape
			bbox = []
			classIds = []
			confs = []

			for output in outputs:
				for det in output:
					scores = det[5:]
					classId = np.argmax(scores)
					confidence = scores[classId]
					if confidence > confThreshold:
						w,h = int(det[2]*wT) , int(det[3]*hT) 
						x,y = int((det[0]*wT) - w/2) , int((det[1]*hT)-h/2)
						bbox.append([x,y,w,h])
						classIds.append(classId)
						confs.append(float(confidence))

				#print(len(bbox))
			indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
				
			for i in indices:
				i = i[0]
				box = bbox[i]
				x,y,w,h = box[0],box[1],box[2],box[3]
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
				cv2.putText(img,f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
				
		while True :
			success, img=cap.read()
				
			blob = cv2.dnn.blobFromImage(img, 1/255,(whT,whT),[0,0,0],1,crop=False)
			net.setInput(blob)

			layerNames = net.getLayerNames()
				#print(layerNames)
			outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
				#print(outputNames)

			outputs = net.forward(outputNames)

			findObjects(outputs,img)

			cv2.imshow('Image',img)
			if cv2.waitKey(10) & 0xFF == ord('q'):
					break
		cap.release()
		cv2.destroyAllWindows()
	
def detect_realtime():
	cap = cv2.VideoCapture(0)
	whT = 320
	confThreshold = 0.5
	nmsThreshold = 0.1

	classesFile = "coco.names"
	classNames = []
	with open(classesFile, 'rt') as f:
		classNames = f.read().rstrip('\n').split('\n')

	modelConfiguration ="custom-yolov4-tiny-detector.cfg"
	modelWeights = "custom-yolov4-tiny-detector_best.weights"

	net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

	def findObjects(outputs,img):
		hT, wT, cT = img.shape
		bbox = []
		classIds = []
		confs = []

		for output in outputs:
			for det in output:
				scores = det[5:]
				classId = np.argmax(scores)
				confidence = scores[classId]
				if confidence > confThreshold:
					w,h = int(det[2]*wT) , int(det[3]*hT) 
					x,y = int((det[0]*wT) - w/2) , int((det[1]*hT)-h/2)
					bbox.append([x,y,w,h])
					classIds.append(classId)
					confs.append(float(confidence))

		#print(len(bbox))
		indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
		
		for i in indices:
			i = i[0]
			box = bbox[i]
			x,y,w,h = box[0],box[1],box[2],box[3]
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
			cv2.putText(img,f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
		
	while True :
		success, img=cap.read()
		
		blob = cv2.dnn.blobFromImage(img, 1/255,(whT,whT),[0,0,0],1,crop=False)
		net.setInput(blob)

		layerNames = net.getLayerNames()
		#print(layerNames)
		outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
		#print(outputNames)

		outputs = net.forward(outputNames)
		#print(outputs[0].shape)
		#print(outputs[1].shape)	
		#print(outputs[2].shape)

		findObjects(outputs,img)

		cv2.imshow('Image',img)
		if cv2.waitKey(10) & 0xFF == ord('q'):
				break
	cap.release()
	cv2.destroyAllWindows()
	
# initialize the window toolkit along with the two image panels
root = Tk()
root.geometry('500x570')
root.title('Herbs Cam')
root.config(background='light green')
frame = Frame(root, relief=RIDGE, borderwidth=0)
frame.pack(fill=BOTH)
frame.config(background='DarkOliveGreen4')
label = Label(frame, text="Herbs Cam",bg='DarkOliveGreen4',font=('Times 35 bold'))
label.pack(side=TOP)
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Detect Image", background="DarkOliveGreen3", fg="darkgreen",font=('Times 24'),command=detect_image)
btn.pack(fill="both", expand="yes", padx="50", pady="32")
btn1 = Button(root, text="Detect Video", background="DarkOliveGreen3", fg="darkgreen" ,font=('Times 24'),command=detect_video)
btn1.pack(fill="both", expand="yes", padx="50", pady="32")
btn2 = Button(root, text="Detect Real-time", background="DarkOliveGreen3", fg="darkgreen" ,font=('Times 24'),command=detect_realtime)
btn2.pack(fill="both", expand="yes", padx="50", pady="32")
# kick off the GUI
root.mainloop()