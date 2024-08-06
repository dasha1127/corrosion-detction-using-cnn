import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import numpy as np

class Corrosion_Detection(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # imput : 3 x 224 x 224
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # output : 32 x 224 x 224
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 112 x 112

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 56 x 56

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 28 x 28

            nn.Flatten(),
            nn.Linear(256*28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2))

    def forward(self, xb):
        return self.network(xb)

def predict_single(image, model):
    xb = image.unsqueeze(0)
    preds = model(xb)
    prediction = preds[0]
    a = prediction[0].item()
    b = prediction[1].item()
    print(a, b);
    if a>b :
        return "## Corrosion Detected"
    else:
        return "## No Corrosion Detected"

st.title("Corrosion Detection")

@st.cache_resource
def load_model():
  path = '/Users/dashasdharshan/Downloads/Corrosion_Detection_Sgr/trained_model.pth'
  model = Corrosion_Detection()
  model.load_state_dict(torch.load(path, map_location='cpu'))
  return model

with st.spinner('Model is being loded..'):
  model = load_model()

file = st.file_uploader("Please upload a picture", type = ["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
  image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
  og_image = image
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((256,256)),
          transforms.CenterCrop(size=224),
          transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
  ])
  tensor = transform(image)
  prediction = predict_single(tensor, model)
  return prediction

if file is None:
  st.text("Please upload an image file")
else :
  st.image(file, use_column_width=True)
  fb = np.asarray(bytearray(file.read()), dtype=np.uint8)
  image = cv2.imdecode(fb, 1)
  predictions = import_and_predict(image, model)
  st.markdown(predictions)
