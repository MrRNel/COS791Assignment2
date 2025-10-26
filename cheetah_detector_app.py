import gradio as gr
from ultralytics import YOLO
import uuid
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Model loaded here on startup
MODEL_PATH = "cheetah_detection/best_run/best_cheetah_model.pt"
model = YOLO(MODEL_PATH)

def detect_cheetahs(input_media, input_type, model_name, image_size, confidence_threshold):
    """Run inference on images or videos"""
    if input_media is None:
        return None, None, "Please upload an image or video"

    try:
        if input_type == "Image":
            # Image processing happens here
            img_path = input_media.name if hasattr(input_media, 'name') else input_media
            results = model(
                img_path,
                conf=confidence_threshold,
                imgsz=int(image_size),
                iou=0.3,
                max_det=20,
                verbose=False
            )

            # Get results
            result = results[0]

            # Load original image
            original_img = cv2.imread(img_path)
            if original_img is not None:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            else:
                # Fallback to YOLO plot if image can't be loaded
                original_img = result.plot()

            # Draw boxes on original image
            if result.boxes is not None and original_img is not None:
                h, w = original_img.shape[:2]

                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()

                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label with background
                    label = f"Cheetah {conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2

                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                    # Position label inside box if it would go outside image bounds
                    label_y = y1 - 10 if y1 - 30 > 0 else y1 + 25
                    label_x = max(x1, 5)

                    # Ensure label doesn't go outside image
                    if label_x + text_width > w:
                        label_x = w - text_width - 5
                    if label_y < text_height + 5:
                        label_y = text_height + 5

                    # Draw label background
                    cv2.rectangle(original_img, 
                                 (label_x - 2, label_y - text_height - 5), 
                                 (label_x + text_width + 2, label_y + baseline + 2), 
                                 (0, 0, 0), -1)

                    # Draw label
                    cv2.putText(original_img, label, (label_x, label_y),
                              font, font_scale, (0, 255, 0), thickness)

            annotated_image = original_img

            # Count and report results
            num_detections = len(result.boxes) if result.boxes is not None else 0
            result_text = f"Detections: {num_detections}\n"

            if num_detections > 0:
                confidences = result.boxes.conf.cpu().numpy()
                result_text += f"Confidences: {[round(c, 3) for c in confidences]}\n"
                result_text += f"Average confidence: {np.mean(confidences):.3f}"
            else:
                result_text = "No cheetahs detected"

            return annotated_image, None, result_text

        else:  # Video input
            # Video processing happens here
            video_path = input_media.name if hasattr(input_media, 'name') else input_media

            # Create unique output dir to avoid conflicts
            unique_id = uuid.uuid4().hex[:8]
            unique_name = f"output_{unique_id}"

            results = model(
                video_path,
                conf=confidence_threshold,
                imgsz=int(image_size),
                iou=0.3,
                max_det=20,
                verbose=False,
                save=True,
                save_txt=True,
                project="video_detection",
                name=unique_name,
                stream=False  # Process all frames
            )

            # Find the saved video
            output_dir = Path(f"video_detection/{unique_name}")

            video_file = None
            if output_dir.exists():
                # Grab any video file from the output dir
                video_files = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.avi")) + list(output_dir.glob("*.mkv"))
                if video_files:
                    video_file = max(video_files, key=lambda x: x.stat().st_mtime)

            if video_file and video_file.exists():
                abs_path = video_file.resolve()

                # Count total detections across frames
                detection_count = 0
                if results and len(results) > 0:
                    detection_count = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)

                return None, str(abs_path), f"Processed!\nDetections: {detection_count} across all frames"

            error_msg = f"Video output not found"
            return None, None, error_msg

    except Exception as e:
        return None, None, f"Error: {str(e)}"

def detect_cheetahs_video(input_video, model_name, image_size, confidence_threshold):
    """
    Detect cheetahs in video frames.

    Args:
        input_video: Path to video file
        model_name: Model name
        image_size: Image size for inference
        confidence_threshold: Confidence threshold for detections

    Returns:
        Output video path with detections
    """
    if input_video is None:
        return None

    try:
        # Run inference on video
        results = model(
            input_video,
            conf=confidence_threshold,
            imgsz=int(image_size),
            iou=0.3,
            max_det=20,
            verbose=False,
            save=True,
            project="video_detection",
            name="cheetah_video"
        )

        # Find output video
        output_dir = Path("video_detection/cheetah_video")
        if output_dir.exists():
            video_files = list(output_dir.glob("*.mp4"))
            if video_files:
                return str(video_files[0])

        return "Error: Video processing failed"

    except Exception as e:
        return f"Error: {str(e)}"

# Load example images for quick testing
def get_example_images():
    examples = []
    test_dir = Path("cheetah_data/cheetah_test")

    if test_dir.exists():
        all_test_images = sorted(list(test_dir.glob("*.jpg")))

        for img_path in all_test_images:
            examples.append([
                str(img_path),
                "Image",
                "best_cheetah_model.pt",
                400,
                0.25
            ])

    return examples

# Build the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="YOLOv12 Cheetah Detection") as app:
    gr.HTML(
        """
        <style>
        @keyframes rainbow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .student-id {
            font-size: 48px;
            font-weight: bold;
            background: linear-gradient(-45deg, #ff0000, #ff7f00, #00ff00, #ff0000, #ff7f00, #00ff00);
            background-size: 400% 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: rainbow 8s ease infinite;
            text-align: left;
            margin: 20px 0;
        }
        </style>
        <div style="text-align: left; padding: 20px;">
            <h1 style="margin-bottom: 10px;">🐆 YOLOv12 Cheetah Detection</h1>
            <div class="student-id">U25748956</div>
            <p style="margin-top: 10px;">Real-time cheetah detection using YOLOv12 model trained on custom cheetah dataset.</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image",
                type="filepath",
                height=400
            )
            input_video = gr.Video(
                label="Input Video",
                visible=False,
                sources=["upload", "webcam"]
            )
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Detected Cheetahs",
                type="numpy",
                height=500,
                visible=True
            )
            output_video = gr.Video(
                label="Detected Cheetahs (Video)",
                height=500,
                visible=False
            )
    
    # Settings
    with gr.Row():
        with gr.Column(scale=1):
            input_type = gr.Radio(
                choices=["Image", "Video"],
                value="Image",
                label="Input Type"
            )
            
            model_dropdown = gr.Dropdown(
                choices=["best_cheetah_model.pt"],
                value="best_cheetah_model.pt",
                label="Model"
            )
        
        with gr.Column(scale=2):
            image_size = gr.Slider(
                minimum=320,
                maximum=1024,
                step=32,
                value=400,
                label="Image Size",
                info="Recommended: 400"
            )
            
            confidence_threshold = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                step=0.05,
                value=0.25,
                label="Confidence Threshold",
                info="Lower = more detections"
            )
    
    detect_button = gr.Button("Detect Objects", variant="primary", size="lg")
    result_text = gr.Textbox(label="Detection Results", interactive=False)
    
    def toggle_input_type(input_type_value):
        if input_type_value == "Video":
            return gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=False)
    
    def detect_with_input(img, vid, itype, model_name, img_size, conf):
        media = vid if itype == "Video" else img
        if media is None:
            return None, None, "Please upload an image or video"
        return detect_cheetahs(media, itype, model_name, img_size, conf)
    
    detect_button.click(
        fn=detect_with_input,
        inputs=[input_image, input_video, input_type, model_dropdown, image_size, confidence_threshold],
        outputs=[output_image, output_video, result_text]
    )
    
    def toggle_display(input_type_value):
        if input_type_value == "Video":
            return (gr.update(visible=False), gr.update(visible=True),
                    gr.update(visible=False), gr.update(visible=True))
        else:
            return (gr.update(visible=True), gr.update(visible=False),
                    gr.update(visible=True), gr.update(visible=False))
    
    input_type.change(
        fn=toggle_display,
        inputs=[input_type],
        outputs=[input_image, input_video, output_image, output_video]
    )
    
    gr.Markdown("## Examples")
    examples = get_example_images()
    
    gr.Examples(
        examples=examples if examples else [["No examples available"]],
        inputs=[input_image, input_type, model_dropdown, image_size, confidence_threshold],
        outputs=[output_image, result_text],
        fn=detect_cheetahs,
        cache_examples=False
    )

if __name__ == "__main__":
    import os
    
    print("Starting app...")
    print(f"Model: {MODEL_PATH}")
    
    # For Coolify: listen on all interfaces, use PORT env var or default
    server_host = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("PORT", "7860"))
    
    app.launch(
        share=False,
        server_name=server_host,
        server_port=server_port
    )

