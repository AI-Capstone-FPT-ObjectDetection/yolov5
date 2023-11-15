from ultralytics.utils.plotting import Annotator, colors, save_one_box
# names should be imported from data.yaml file
names = ['Breadboard', 'Button', 'Buzzer', 'Infared_Sensor', 'Keypad', 'LCD', 'LED', 'LED_Matrix', 'Number_LED', 'Photoresistor', 'Potentiometer', 'Remote', 'Resistor', 'Rheostat', 'Servo_Engine', 'Thermistor', 'Tool_Box', 'Tool_Box_Tray', 'UNO_R3', 'USB_Cable', 'Ultrasonic_Sensor', 'Wire']
import cv2, os, torch
from utils.general import LOGGER, clip_boxes, increment_path, xywh2xyxy, xyxy2xywh,xywhn2xyxy
from pathlib import Path




def make_image_wbbox(image, preds, hide_conf=False, hide_labels=False):
  print('\nCombining Image and Labels...')
  im0 = image
  imgsz = im0.shape[:2] # h w 
  print('Shape of the output image is [w, h]: ', imgsz[1], imgsz[0])

  line_thickness = 2* int(imgsz[1]/640)

  # Process predictions, plot onto image
  for i, bbox_str in enumerate(preds): 
      if bbox_str == '':
        continue
      bbox = bbox_str.split(' ')
      c, xywh_ratio_str, conf = int(float(bbox[0])), bbox[1:5], float(bbox[5])
      xywh_ratio = [float(i) for i in xywh_ratio_str]
      xyxy = xywhn2xyxy(torch.tensor(xywh_ratio), w=imgsz[1], h=imgsz[0])
      annotator = Annotator(im0, line_width=line_thickness, example=str(names))
      label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
      annotator.box_label(xyxy, label, color=colors(c, True))

  img_with_bboxes = annotator.result()
  return img_with_bboxes



# # Test the above function: run this python file
# label_file = '/content/drive/MyDrive/iot/yolov5_codingIoU/runs/detect/exp27/labels/1698811617178.txt'
# img_file = '/content/drive/MyDrive/AI_Engineer/Dataset-IoT/Nov1-testmore-fund best case/1698811617178.jpg'
# SAVE_DIR = '/content/drive/MyDrive/iot/yolov5_codingIoU/img_with_pred'

# with open(label_file, 'r') as f:
#    bboxes = f.read().split('\n')
#    print('Labels loaded from ', label_file)

# image = cv2.imread(img_file)

# img_to_save = make_image_wbbox(image, bboxes)

# save_path = SAVE_DIR+'/'+ Path(img_file).stem + '.jpg'
# cv2.imwrite(save_path, img_to_save)
# print(f'Image saved to {save_path}')



