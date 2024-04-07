from flask import send_from_directory
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# import sys
# sys.path.insert(1, 'SWIN-LSTM/')


from swinlstm import swin_t, LSTMSWINModel

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(img):
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Image data is empty or invalid.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_pil = Image.fromarray((img * 255).astype('uint8'), 'RGB')
    img = transform(img_pil)
    img = img.unsqueeze(0).unsqueeze(0)

    return img

def predict(img, model, device):
    img_tensor = preprocess_image(img)
    img_tensor = img_tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)

    probabilities = torch.softmax(outputs, dim=1)
    class_names = ['glioma', 'meningioma','notumor','pituitary']
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name, probabilities.cpu().numpy()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = Image.open(filepath).convert("RGB")
        img_array = np.array(img) / 255.0
        predicted_class, probabilities = predict(img_array, lstm_swin_model, device)
        max_probability = np.max(probabilities)

        image_url = url_for('uploaded_file', filename=filename)

        return render_template('result.html', predicted_class=predicted_class, max_probability=max_probability, image_url=image_url)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    swin_backbone = swin_t()
    sample_input = torch.randn(1, 3, 224, 224)
    swin_output = swin_backbone(sample_input)
    lstm_input_size = swin_output.size(-1)

    lstm_hidden_size = 128
    num_classes = 4
    lstm_swin_model = LSTMSWINModel(swin_backbone, lstm_input_size, lstm_hidden_size, num_classes)
    model_path = "SWIN-LSTM-4C-model100.pth"
    lstm_swin_model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_swin_model = lstm_swin_model.to(device)
    lstm_swin_model.eval()
    app.run(debug=True)
