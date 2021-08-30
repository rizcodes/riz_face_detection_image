import os
import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from settings import project_dirs, logger


class ProcessImage:
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.faces = None
        self.extracted_faces = []
        
    def detect_faces(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.faces = face_cascade.detectMultiScale(self.image,
                                                   scaleFactor=1.3,
                                                   minNeighbors=3,
                                                   minSize=(30, 30))
        logger.info(f'Found [{len(self.faces)}] faces')

    def extract_faces(self):
        if self.faces is None:
            self.detect_faces()
        idx = 1
        for (x, y, w, h) in self.faces:
            roi_color = self.image[y:y + h, x:x + w]
            face_file = f'face_{str(w)}_{str(h)}_crop_{idx}.jpg'
            face_path = os.path.join(project_dirs['proc_dir'], face_file)
            self.extracted_faces.append(face_path)
            cv2.imwrite(face_path, roi_color)
            idx += 1

    def _classify_face(self, _face, _net, _device):
        classes = {0: 'OTHER', 1: 'RIZWAN'}
        if _face is None:
            result = "NO FACE DETECTED"
        else:
            image_transform = transforms.Compose([
                    transforms.Resize((32,32)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor()
                    ])
            image = image_transform(_face)
            image = torch.unsqueeze(image,0)
            image = image.to(_device)
            prediction = _net.forward(image)
            _, output = torch.max(prediction, 1)
            zero_dim_example_tensor = output.to(torch.int)
            label_int = zero_dim_example_tensor.item()
            result = classes[label_int]
        return result

    def _display_image(self, face_list, img_array):
        width = 8
        height = 8
        rows = 4
        cols = 4
        axes = []
        fig = plt.figure()
        for i in range(rows*cols):
            if i < len(face_list):
                # print (f'{i}:{len(img_array)}')
                axes.append(fig.add_subplot(rows, cols, i+1))
                subplot_title = (face_list[i])
                axes[-1].set_title(subplot_title)  
                plt.imshow(img_array[i])
        fig.tight_layout()
        plt.show()
    
    def detect_person(self, net, device):
        face_list = []
        img_array = []
        logger.info('Extracting faces')
        self.extract_faces()
        for path in self.extracted_faces:
            # Resize Image
            img = cv2.imread(path)
            resize_img = cv2.resize(img, (32,32))
            img_array.append(resize_img)
            # Convert np.array image to PIL
            img_PIL = Image.fromarray(resize_img)
            # Detect face
            face_result = self._classify_face(img_PIL, net, device)
            face_list.append(face_result)
        logger.info(f'Detected images of : {face_list}')
        self._display_image(face_list, img_array)
