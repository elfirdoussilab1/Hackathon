from ultralytics import YOLO
import cv2


def count_buildings(path_to_image, path_to_weights):
    model = YOLO(path_to_weights)
    # Object segmentation results
    results = model(path_to_image, show=True)

    # Counting the number of detected objects
    num_objects = len(results[0].boxes)

    # Display the number of detected objects
    print("Number of objects detected:", num_objects)
    return num_objects

# Downloading YOLO (model) weights
model = YOLO('../YOLO-weights/best-3.pt')

# Object segmentation results
results = model("../Yolo/images/5.png", show = True)

# Counting the number of detected objects
first_image_objects = results[0].boxes[0] # Access the tensor for the first image
num_objects = len(results[0].boxes)

# Display the number of detected objects
print("Number of objects detected:", num_objects)

# Stopping the process to see the image classified
cv2.waitKey(0)
