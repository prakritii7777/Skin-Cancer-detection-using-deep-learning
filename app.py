from flask import Flask, render_template, request
import torch
from torchvision import transforms, models
from PIL import Image
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mapping
class_mapping = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
class_names = {v: k for k, v in class_mapping.items()}

# Load model
def load_model(model_path, device, num_classes=4):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model('best_resnet50.pth', device)

# Predict function
def classify_image(image_path, model, class_names, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

    predicted_class = class_names[predicted_idx]
    return predicted_class, confidence

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_class, confidence = classify_image(file_path, model, class_names, device)
            return render_template('index.html',
                                   image_url=file_path,
                                   predicted_class=predicted_class,
                                   confidence=round(confidence * 100, 2))
    return render_template('index.html', image_url=None)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
