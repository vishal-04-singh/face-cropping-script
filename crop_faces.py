import cv2
import os
import numpy as np
import sys
import shutil

# Configuration
input_base_dir = "input_folders"  # Base directory containing all input folders
output_base_dir = "cropped_results"  # Base directory for all output folders
min_dim = 800  # Increased minimum width or height of output image for better quality
side_padding_ratio = 0.9  # 90% padding on left/right sides of face
top_padding_ratio = 0.6  # 60% padding above the face
bottom_padding_ratio = 0.8  # 80% padding below the face
confidence_threshold = 0.8  # Minimum confidence for DNN face detection
output_quality = 100  # JPEG quality (0-100, maximum quality)
sharpen_amount = 0  # No sharpening

# Initialize face detector (DNN-based for better accuracy)
# Initialize variables
use_dnn = False
face_cascade = None

try:
    # Use more accurate DNN-based face detector
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    
    # Check if models exist, if not create the directory
    os.makedirs("models", exist_ok=True)
    
    # If models don't exist, fall back to Haar cascade
    if not (os.path.exists(modelFile) and os.path.exists(configFile)):
        print("DNN face detector models not found. Using Haar cascade instead.")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_detector = face_cascade
        use_dnn = False
    else:
        face_detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        use_dnn = True
except:
    print("Failed to load DNN face detector. Falling back to Haar cascade.")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_detector = face_cascade
    use_dnn = False

# Initialize facial landmark detector for better alignment
try:
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel("models/lbfmodel.yaml")
    use_landmarks = True
except:
    print("Facial landmark detector not available. Proceeding without landmark alignment.")
    use_landmarks = False

def detect_faces_dnn(image):
    """
    Detect faces using DNN model (more accurate than Haar cascade)
    """
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    face_detector.setInput(blob)
    detections = face_detector.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            
            # Convert to x, y, w, h format
            faces.append((x1, y1, x2-x1, y2-y1, confidence))
    
    return faces

def detect_faces_haar(image, gray=None):
    """
    Detect faces using Haar cascade
    """
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Convert to same format as DNN output (x,y,w,h,conf)
    return [(x, y, w, h, 1.0) for x, y, w, h in faces]

def detect_facial_landmarks(image, face_rect):
    """
    Detect facial landmarks to improve cropping alignment
    """
    x, y, w, h = face_rect[:4]
    
    # Ensure rect is valid
    x = max(0, x)
    y = max(0, y)
    right = min(image.shape[1], x + w)
    bottom = min(image.shape[0], y + h)
    w = right - x
    h = bottom - y
    
    if w <= 0 or h <= 0:
        return None
    
    # Prepare face rect in format expected by landmark detector
    faces = np.array([[x, y, w, h]], dtype=np.int32)
    success, landmarks = landmark_detector.fit(image, faces)
    
    if not success or len(landmarks) == 0:
        return None
        
    return landmarks[0][0]  # Return landmarks for the first face

def get_optimal_crop_from_landmarks(image, landmarks):
    """
    Calculate optimal crop region based on facial landmarks
    """
    # Extract key points for eyes, nose, and mouth
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    nose_tip = landmarks[30]
    mouth_center = np.mean(landmarks[48:68], axis=0)
    
    # Calculate face center and dimensions based on landmarks
    eyes_center = (left_eye + right_eye) / 2
    face_height = np.linalg.norm(mouth_center - eyes_center) * 2.7  # Approximate face height
    eye_distance = np.linalg.norm(right_eye - left_eye)
    face_width = eye_distance * 2.5  # Approximate face width
    
    # Calculate crop coordinates with proper padding
    center_x, center_y = eyes_center[0], (eyes_center[1] + nose_tip[1]) / 2
    
    crop_width = face_width * (1 + 2 * side_padding_ratio)
    crop_height_top = face_height * (0.5 + top_padding_ratio)  # 0.5 of face height above eyes
    crop_height_bottom = face_height * (0.7 + bottom_padding_ratio)  # 0.7 of face height below eyes
    
    x1 = int(max(0, center_x - crop_width / 2))
    y1 = int(max(0, center_y - crop_height_top))
    x2 = int(min(image.shape[1], center_x + crop_width / 2))
    y2 = int(min(image.shape[0], center_y + crop_height_bottom))
    
    return (x1, y1, x2, y2)

def enhance_image(image):
    """
    Apply image enhancements to improve quality
    """
    # Apply mild sharpening if sharpen_amount > 0
    if sharpen_amount > 0:
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1], 
                           [-1, 9 + sharpen_amount, -1], 
                           [-1, -1, -1]])
        
        # Normalize kernel to avoid brightness changes
        kernel = kernel / kernel.sum()
        
        # Apply sharpening filter
        image = cv2.filter2D(image, -1, kernel)
    
    # Apply minor contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on luminance channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

def preserve_image_quality(original_path, image):
    """
    Determine best format for saving based on original image
    """
    ext = os.path.splitext(original_path)[1].lower()
    
    # If original is png, preserve transparency and use png
    if ext == '.png':
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        if has_alpha:
            return '.png', []  # Return png extension and empty params
    
    # Default to high quality jpeg for other formats
    return '.jpg', [int(cv2.IMWRITE_JPEG_QUALITY), output_quality]

def resize_high_quality(image, new_width, new_height):
    """
    High quality image resizing using multi-step approach
    """
    # Get current dimensions
    h, w = image.shape[:2]
    
    # If we need to upscale, use a more advanced scaling approach
    if new_width > w or new_height > h:
        # For upscaling, use a multi-scale approach for better quality
        # First resize to 2x the original size with CUBIC interpolation
        scale_factor = min(2.0, min(new_width/w, new_height/h))
        if scale_factor > 1:
            inter_w = int(w * scale_factor)
            inter_h = int(h * scale_factor)
            image = cv2.resize(image, (inter_w, inter_h), interpolation=cv2.INTER_CUBIC)
    
    # Final resize to target dimensions with LANCZOS4 for best quality
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

def crop_face(image_path, show_detection=False):
    """
    Main function to detect and crop faces from an image
    """
    try:
        # Read image at highest quality
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Can't read image: {image_path}")
            return None
            
        # Validate image dimensions
        if img.shape[0] < 10 or img.shape[1] < 10:
            print(f"Error: Image too small: {image_path}")
            return None
            
        # Create copy for visualization if needed
        if show_detection:
            vis_img = img.copy()
        
        # Convert to grayscale for face detection
        if len(img.shape) == 3 and img.shape[2] == 4:  # Handle alpha channel
            gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = []
        if use_dnn:
            faces = detect_faces_dnn(img)
        else:
            faces = detect_faces_haar(img, gray)
        
        if len(faces) == 0:
            print(f"No face found in: {image_path}")
            return None
            
        # Find the largest face or highest confidence face
        if use_dnn:
            # Sort by confidence for DNN detector
            face_rect = max(faces, key=lambda f: f[4])
        else:
            # Sort by area for Haar detector
            face_rect = max(faces, key=lambda f: f[2] * f[3])
        
        x, y, w, h = face_rect[:4]
        
        # Try to use facial landmarks for better alignment if available
        if use_landmarks:
            landmarks = detect_facial_landmarks(gray, face_rect)
            if landmarks is not None:
                x1, y1, x2, y2 = get_optimal_crop_from_landmarks(img, landmarks)
                
                if show_detection:
                    # Draw landmarks
                    for (x_l, y_l) in landmarks:
                        cv2.circle(vis_img, (int(x_l), int(y_l)), 2, (0, 255, 0), -1)
            else:
                # Fallback to basic padding if landmarks detection fails
                pad_w = int(w * side_padding_ratio)
                pad_top = int(h * top_padding_ratio)
                pad_bottom = int(h * bottom_padding_ratio)
                
                x1 = max(x - pad_w, 0)
                y1 = max(y - pad_top, 0)
                x2 = min(x + w + pad_w, img.shape[1])
                y2 = min(y + h + pad_bottom, img.shape[0])
        else:
            # Basic padding without landmarks
            pad_w = int(w * side_padding_ratio)
            pad_top = int(h * top_padding_ratio)
            pad_bottom = int(h * bottom_padding_ratio)
            
            x1 = max(x - pad_w, 0)
            y1 = max(y - pad_top, 0)
            x2 = min(x + w + pad_w, img.shape[1])
            y2 = min(y + h + pad_bottom, img.shape[0])
        
        # Draw detection rectangle if requested
        if show_detection:
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Face rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)      # Crop rectangle
            debug_dir = os.path.join(os.path.dirname(image_path), "debug")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"debug_{os.path.basename(image_path)}"), vis_img)
        
        # Crop the image
        cropped = img[y1:y2, x1:x2].copy()
        
        # Ensure minimum dimensions while preserving aspect ratio
        h_crop, w_crop = cropped.shape[:2]
        if h_crop < min_dim or w_crop < min_dim:
            # Calculate scale while preserving aspect ratio
            scale = max(min_dim / h_crop, min_dim / w_crop)
            new_w = int(w_crop * scale)
            new_h = int(h_crop * scale)
            cropped = resize_high_quality(cropped, new_w, new_h)
        
        # No enhancement applied - as requested
        
        return cropped
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_folder(input_folder, output_folder, debug_mode=False):
    """
    Process all images in a single folder
    """
    os.makedirs(output_folder, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return 0, 0
    
    print(f"Found {len(image_files)} image files in {input_folder}")
    
    for filename in image_files:
        path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}")
        
        face_img = crop_face(path, show_detection=debug_mode)
        
        if face_img is not None:
            # Determine best format and quality settings for saving
            out_ext, params = preserve_image_quality(path, face_img)
            
            # Generate output filename, potentially changing extension
            base_name = os.path.splitext(filename)[0]
            out_filename = base_name + out_ext
            save_path = os.path.join(output_folder, out_filename)
            
            # Save cropped image to the output folder with optimal quality settings
            if params:
                cv2.imwrite(save_path, face_img, params)
            else:
                cv2.imwrite(save_path, face_img)
                
            print(f"✓ Successfully cropped: {save_path}")
            success_count += 1
        else:
            print(f"✗ Failed to crop: {filename}")
            fail_count += 1
    
    return success_count, fail_count

def main():
    # Process command-line arguments
    debug_mode = "--debug" in sys.argv
    high_quality_mode = "--high-quality" in sys.argv or "-hq" in sys.argv
    
    # Adjust quality settings if high quality mode is requested
    global output_quality, sharpen_amount, min_dim
    if high_quality_mode:
        output_quality = 100  # Maximum JPEG quality
        sharpen_amount = 0    # No sharpening
        min_dim = 1200        # Higher minimum dimension
        print("High quality mode enabled")
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get all folders in the input directory
    input_folders = [f for f in os.listdir(input_base_dir) 
                    if os.path.isdir(os.path.join(input_base_dir, f))]
    
    if not input_folders:
        print(f"No folders found in {input_base_dir}")
        return
    
    print(f"Found {len(input_folders)} folders to process")
    
    # Process each folder
    total_success = 0
    total_fail = 0
    
    for folder_name in input_folders:
        input_folder = os.path.join(input_base_dir, folder_name)
        output_folder = os.path.join(output_base_dir, folder_name)
        
        print(f"\n{'='*50}")
        print(f"Processing folder: {folder_name}")
        print(f"{'='*50}")
        
        success, fail = process_folder(input_folder, output_folder, debug_mode)
        total_success += success
        total_fail += fail
        
        print(f"\nFolder summary for {folder_name}:")
        print(f"  Success: {success}")
        print(f"  Failed: {fail}")
    
    # Print overall summary
    total = total_success + total_fail
    if total > 0:
        print(f"\n{'='*50}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*50}")
        print(f"  Total folders processed: {len(input_folders)}")
        print(f"  Total images processed: {total}")
        print(f"  Success: {total_success} ({total_success/total*100:.1f}%)")
        print(f"  Failed: {total_fail} ({total_fail/total*100:.1f}%)")
        print(f"Cropped images saved to: {os.path.abspath(output_base_dir)}")
    else:
        print("\nNo images were processed.")

if __name__ == "__main__":
    main()