from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
from cellpose import models
import matplotlib.pyplot as plt
from skimage.color import label2rgb

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'static/uploads/'
MODEL_PATH = 'cyto2_model_1402_withoutaug'  # Adjust this to your actual model file

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed image formats
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

def allowed_file(filename):
    """Check if the file has a valid extension."""
    return filename.lower().endswith(VALID_EXTENSIONS)

def is_valid_image(image_path):
    """Check if the file is a valid image and dimensions are ≤ 1024x1024."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, "Error: Unable to load image. Ensure it is a valid image file."
    
    h, w = img.shape
    if h > 1024 or w > 1024:
        return False, f"Error: Image dimensions ({w}x{h}) exceed 1024x1024."

    return True, img

def save_mask(mask, img_path):
    """Save the predicted mask with labels where each cell gets a different color."""
    base_name, ext = os.path.splitext(img_path)
    mask_path = f"{base_name}_mask.png"  # Save as PNG
    
    # Convert the mask into a colored label image
    color_mask = label2rgb(mask, bg_label=0)  # Assign unique colors to each cell
    
    # Save the colorized mask using Matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(color_mask)
    plt.axis('off')
    plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return mask_path

def evaluate(model_path, img_path, flow_threshold):
    """Load model, preprocess image, evaluate, and save mask."""
    print(f"Loading model from: {model_path}")

    # Load the trained model
    trained_model = models.CellposeModel(pretrained_model=model_path, gpu=False)

    # Validate and load image
    is_valid, img = is_valid_image(img_path)
    if not is_valid:
        raise ValueError(img)  # img contains the error message

    # Resize the image to 512x512
    img_resized = cv2.resize(img, (512, 512))

    print("Running model evaluation...")
    trained_results = trained_model.eval(
        img_resized, 
        channels=[0, 0], 
        diameter=50, 
        flow_threshold=flow_threshold
    )

    # Extract the mask from results
    masks = trained_results[0]  # First item in the result is the mask

    # Save the predicted mask as a labeled image
    mask_path = save_mask(masks, img_path)
    return mask_path

    
@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = None
    segmented_image = None
    uploaded_filename = None  # Store the file name

    if request.method == 'POST':
        # If a new file is uploaded, save it
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if allowed_file(file.filename):
                filename = file.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                uploaded_image = filename  # Save filename for rendering
                uploaded_filename = filename  # Store filename to display
            else:
                return render_template('index.html', error="Invalid file type")

        # Keep using the previously uploaded image and filename if no new file is selected
        elif request.form.get('existing_image'):
            uploaded_image = request.form.get('existing_image')
            uploaded_filename = request.form.get('existing_filename')  # Keep filename

        # Get flow threshold from form
        flow_threshold = request.form.get('flow_threshold', 0.4)
        try:
            flow_threshold = float(flow_threshold)
            if flow_threshold <= 0 or flow_threshold >= 1:
                return render_template('index.html', error="Flow threshold must be between 0 and 1.", uploaded_image=uploaded_image, uploaded_filename=uploaded_filename)
        except ValueError:
            return render_template('index.html', error="Invalid flow threshold value", uploaded_image=uploaded_image, uploaded_filename=uploaded_filename)

        # Perform segmentation if an uploaded image exists
        if uploaded_image:
            filepath = os.path.join(UPLOAD_FOLDER, uploaded_image)
            segmented_image = evaluate(MODEL_PATH, filepath, flow_threshold)
            segmented_image = os.path.basename(segmented_image)

        return render_template('index.html', uploaded_image=uploaded_image, uploaded_filename=uploaded_filename, segmented_image=segmented_image, flow_threshold=flow_threshold)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
