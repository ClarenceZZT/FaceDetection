import torch
import numpy as np
import cv2
import config
from model import FaceKeypointResNet50
import time

options = ["original", "half_precision", "prune", "augment"]
option = options[3]

if option == options[0]:
    model = FaceKeypointResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
    checkpoint = torch.load('../outputs/model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
elif option == options[1]:
    model = FaceKeypointResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
    checkpoint = torch.load('../outputs/model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model=model.to(torch.float16)
elif option == options[2]:
    model = FaceKeypointResNet50(pretrained=False, requires_grad=False,pruning_amount = 0.2).to(config.DEVICE)
    checkpoint = torch.load('../outputs/model_prune_0.2_new.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model = FaceKeypointResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
    checkpoint = torch.load('../outputs/model_data_aug_new.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
model.eval()
print('model loaded successfully')

# capture the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if (cap.isOpened() == False):
    print('Error while trying to open webcam. Plese check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# set up the save file path
save_path = f"{config.OUTPUT_PATH}/vid_keypoint_detection.mp4"
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"{save_path}", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                      (frame_width, frame_height))

# total_frames = 0
# total_time = 0

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        with torch.no_grad():
            start_time = time.time()
            image = frame
            image = cv2.resize(image, (224, 224))
            orig_frame = image.copy()
            orig_h, orig_w, c = orig_frame.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            if option == options[1]:
                image = torch.tensor(image, dtype=torch.float16)
            else:
                image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to(config.DEVICE)
            outputs = model(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.reshape(-1, 2)
        keypoints = outputs
        for p in range(keypoints.shape[0]):
            cv2.circle(orig_frame, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                        1, (0, 0, 255), -1, cv2.LINE_AA)
        orig_frame = cv2.resize(orig_frame, (frame_width, frame_height))
        
        elapsed_time = time.time() - start_time
        # total_frames += 1
        # total_time += elapsed_time

        # Calculate and display FPS
        # fps = total_frames / total_time
        fps = 1 / (elapsed_time)
        cv2.putText(orig_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        
        cv2.imshow('Facial Keypoint Frame', orig_frame)
        out.write(orig_frame)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
 
    else: 
        break
out.release()
# release VideoCapture()
cap.release()
 
# close all frames and video windows
cv2.destroyAllWindows()