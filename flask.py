from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64

import torchvision.transforms as transforms

app = Flask(__name__)

model = torch.load('Downloads\model.onnx')
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    
    img = Image.open(file.stream)
    
  
    img_tensor = transform(img).unsqueeze(0) 
    
    with torch.no_grad():
        prediction = model(img_tensor)
 
    _, predicted_class = torch.max(prediction, dim=1)

    
    return jsonify({
        "Predicted Class": predicted_class.item()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

