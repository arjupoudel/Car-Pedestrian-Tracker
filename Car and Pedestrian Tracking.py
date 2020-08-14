import cv2

# Our car Image
img_file = 'carImage.jpg'
video = cv2.VideoCapture('pedVideo.mp4')
#video = cv2.VideoCapture('carVideo.mp4')

# Our pre-tained car classifier and pedestrian classifier
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file='pedestrian_detector.xml'

# create car classifier from above file; it's classifying things as a car, pedestrian, apple or face anything
car_tracker = cv2.CascadeClassifier(car_tracker_file)

# create pedestrian classifier from xml file
ped_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# Run forever until car stops or crashes
while True:

    # Read the current frame
    (read_successful, frame) = video.read() # read one frame from the video

    # Safe Coding
    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # detect cars; multiscale means detect car of any scale, any size
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = ped_tracker.detectMultiScale(grayscaled_frame)

    # Draw  rectangles around the cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x+1,y+1), (x+w, y+h), (0, 0, 255), 1) # [542   9  24  24] = x, y, w, h
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 1)

    # Draw  rectangles around the pedestrians
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 1) # [542   9  24  24] = x, y, w, h


    # Display the image with the cars spotted
    cv2.imshow('Arzz Car Detector',frame)

    # Don't autoclose (Wait here in the code and listen for a key press)
    key = cv2.waitKey(1)
    
    # Stop if Q or q key is pressed
    if key==81 or key==ord('q'):
        break

# Release the VideoCapture object, stop reading the file, release all sources, memory management thing
video.release()
   

    

''' #FOR CAR DETECTION IN AN IMAGE
# create opencv image
img = cv2.imread(img_file)

# convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Our pre-tained car classifier
car_tracker_file = 'car_detector.xml'

# create car classifier from above file; it's classifying things as a car
car_tracker = cv2.CascadeClassifier(car_tracker_file)

# detect cars; multiscale means detect car of any scale, any size
cars = car_tracker.detectMultiScale(black_n_white)

# Draw  rectangles around the cars
for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 1) # [542   9  24  24] = x, y, w, h


# Display the image with the cars spotted
cv2.imshow('Arzz Car Detector',img)

# Don't autoclose (Wait here in the code and listen for a key press)
cv2.waitKey()

# was stuck here, it showed opencv error- the function is not implemented but i uninstalled opencv
# re-installed pip install opencv-python instead of pip install opencv-python-headless and it displayed the photo


print("Code completed.")'''
