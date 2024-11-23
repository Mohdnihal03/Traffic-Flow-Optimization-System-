import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model with CPU-only support
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    model.to('cpu')  # Force model to run on CPU
    return model

# Function to count vehicles in the detected image and draw bounding boxes
def count_vehicles(image, model):
    # Use the YOLO model to predict objects in the image
    results = model(image)
    vehicle_count = 0
    img_with_boxes = np.array(image)  # Convert image to numpy array for drawing
    
    # The results object contains a list of predictions
    for result in results:  # results is a list of predictions
        boxes = result.boxes  # The bounding boxes from the prediction
        for box in boxes:
            class_id = int(box.cls.item())  # Extract the class ID for each box
            if class_id in [2, 3, 5, 7]:  # 2: Car, 3: Motorcycle, 5: Bus, 7: Truck 
                vehicle_count += 1
                
                # Draw bounding box around detected vehicle
                x1, y1, x2, y2 = map(int, box.xywh[0])  # Coordinates of the box
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green rectangle
                
                # Optionally, label the bounding box with the class name
                cv2.putText(img_with_boxes, f"Vehicle ", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vehicle_count, img_with_boxes

# Function to determine traffic light timing based on vehicle count
def determine_traffic_light(vehicle_counts):
    # Find the road with the highest vehicle count
    max_road = max(vehicle_counts, key=vehicle_counts.get)
    max_count = vehicle_counts[max_road]
    
    # Assign timings based on vehicle count
    light_timings = {}
    
    for road, count in vehicle_counts.items():
        if road == max_road and count > 0:
            light_timings[road] = 15  # 15 seconds for the road with the most vehicles
        else:
            light_timings[road] = 5  # 5 seconds for other roads
    
    return light_timings

# Streamlit Interface
st.title("Traffic Flow Optimization System")

# Load YOLOv8 model
model = load_model('yolov8n.pt')  # Replace with the correct path to your YOLO model

# Initialize a dictionary to store vehicle counts for each road
vehicle_counts = {}

# Function to handle road cameras and reset after each cycle
def capture_images_from_roads():
    # Capture images from 4 roads (Use the camera input for each road)
    camera_inputs = {}
    for i in range(1, 5):
        st.subheader(f"Road {i} Camera")
        # Provide a dynamic unique key for each camera input widget using a timestamp
        unique_key = f"road_{i}_camera"
        camera_inputs[i] = st.camera_input(f"Open Camera for Road {i}", key=unique_key)
    return camera_inputs

# Capture new images after every cycle
def run_traffic_flow_cycle():
    # Initialize vehicle counts dictionary for this cycle
    vehicle_counts.clear()

    # Capture images from the roads
    camera_inputs = capture_images_from_roads()

    if all(camera_inputs.values()):  # Ensure all cameras are available
        # Capture vehicle counts and process images for each road
        for road, camera_input in camera_inputs.items():
            if camera_input:
                # Convert captured image into a PIL Image object
                img = Image.open(camera_input)

                # Count vehicles using the YOLO model and get the image with bounding boxes
                vehicle_count, img_with_boxes = count_vehicles(img, model)
                vehicle_counts[road] = vehicle_count
                
                # Convert the image with bounding boxes back to PIL to display in Streamlit
                img_with_boxes_pil = Image.fromarray(img_with_boxes)
                
                # Display the image with bounding boxes and vehicle count
                st.image(img_with_boxes_pil, caption=f"Road {road} - Vehicles detected: {vehicle_count}")
        
        # Once images are processed, make a decision about the traffic light
        light_timings = determine_traffic_light(vehicle_counts)
        
        # Display traffic light timings for each road
        for road, timing in light_timings.items():
            st.write(f"Road {road} will have a green light for {timing} seconds.")
        
        # Simulate traffic light cycle
        for road, timing in light_timings.items():
            st.write(f"Green light for Road {road} for {timing} seconds. Red light for other Roads")
            time.sleep(timing)
        # After each full cycle, show reload button
        # if st.button("Reload page"):
        streamlit_js_eval(js_expressions="parent.window.location.reload()")  # Reload the page using JS

# Run the traffic flow cycle
run_traffic_flow_cycle()
