import os
import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

LAST_CONV_LAYER = "conv5_block32_concat"
IMG_SIZE = (768, 768)
MODEL_PATH = "model_multiclass_DenseNet201.keras"
TEST_DIR = "DATA_test"
MASKS_DIR = "Marcadas"

print("Cargando modelo en modo compatibilidad Keras 2...")
model = load_model(MODEL_PATH, compile=False)
class_names = ['Effusion', 'Emphysema', 'No finding', 'Pneumonia', 'Pneumothorax']
print(f"TensorFlow Version: {tf.__version__}")

image_dict = {}
image_names = []
if os.path.exists(TEST_DIR):
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                rel_path = os.path.relpath(os.path.join(root, file), TEST_DIR)
                image_names.append(rel_path)
                image_dict[rel_path] = os.path.join(root, file)

def compute_gradcam(img_array, class_index):
  
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_channel = preds[:, class_index]
        
    grads = tape.gradient(class_channel, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0)
    cam /= tf.reduce_max(cam) + 1e-8
    return cam.numpy()

def overlay_gradcam(cam, orig, alpha=0.45):
    cam = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
    cam_uint8 = np.uint8(255 * cam)
    cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(orig, 1 - alpha, cam_color, alpha, 0)

def find_medical_mask(relative_path):
    if not relative_path: return None
    parts = relative_path.split(os.sep)
    class_name = parts[0] if len(parts) > 1 else ""
    file_name = parts[1] if len(parts) > 1 else parts[0]
    
    mask_path = os.path.join(MASKS_DIR, class_name, file_name)
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None: return mask
    return None

def load_preview_image(image_name):
    if not image_name or image_name not in image_dict: return None
    orig = cv2.imread(image_dict[image_name])
    return cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) if orig is not None else None

def predict_and_explain(image_dropdown, image_upload):
    try:
        orig = None
        mask = None
        true_name = "Subida por usuario"

        if image_dropdown and image_dropdown in image_dict:
            orig = cv2.imread(image_dict[image_dropdown])
            if orig is not None: orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            mask = find_medical_mask(image_dropdown)
            true_name = image_dropdown.split(os.sep)[0]
        elif image_upload is not None:
            orig = np.array(image_upload, dtype=np.uint8)
            mask = None

        if orig is None:
            return "Por favor, seleccione una imagen.", None, None

        resized = cv2.resize(orig, IMG_SIZE)
        inp = preprocess_input(resized.astype(np.float32))
        inp = np.expand_dims(inp, 0)

        preds = model.predict(inp, verbose=0)[0]
        
        sorted_indices = np.argsort(preds)[::-1]
        
        top1_idx = sorted_indices[0]
        top1_name = class_names[top1_idx]
        top1_prob = preds[top1_idx]
        
        top2_idx = sorted_indices[1]
        top2_name = class_names[top2_idx]
        top2_prob = preds[top2_idx]

        cam = compute_gradcam(inp, top1_idx)
        heatmap_overlay = overlay_gradcam(cam, orig)

        if mask is not None:
            mask_resized = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
        else:
            mask_resized = np.zeros((orig.shape[0], orig.shape[1]), dtype=np.uint8)
        
        if top1_name.lower() == true_name.lower():
            result_html = f"<span style='color: #2e7d32; font-weight: bold; background-color: #e8f5e9; padding: 2px 6px; border-radius: 4px;'>Correcto</span>"
        else:
            result_html = f"<span style='color: #c62828; font-weight: bold; background-color: #ffefe0; padding: 2px 6px; border-radius: 4px;'>Incorrecto</span>"
        
        bar_top1 = f"<div style='background-color: #e0e0e0; border-radius: 4px; width: 100%; height: 8px; margin-top: 4px; margin-bottom: 12px;'><div style='background-color: #2196f3; height: 100%; width: {top1_prob*100}%; border-radius: 4px;'></div></div>"
        bar_top2 = f"<div style='background-color: #e0e0e0; border-radius: 4px; width: 100%; height: 8px; margin-top: 4px; margin-bottom: 12px;'><div style='background-color: #9e9e9e; height: 100%; width: {top2_prob*100}%; border-radius: 4px;'></div></div>"

        diagnosis_text = (
            f"### **Informe de Diagnóstico**\n"
            f"**Clase Real:** {true_name}\n\n"
            f"**Primera Probabilidad (Principal):** {top1_name} ({top1_prob:.2%}) — {result_html}\n"
            f"{bar_top1}"
            f"Segunda Probabilidad (Secundaria): {top2_name} ({top2_prob:.2%})\n"
            f"{bar_top2}"
        )

        return diagnosis_text, heatmap_overlay, mask_resized

    except Exception as e:
        return f"Error: {str(e)}", None, None

def clear_all():
    return gr.update(value=None), gr.update(value=None), "### Esperando análisis...", None, None

with gr.Blocks(title="TFM - AI Medical Predictor & Grad-CAM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Clasificación de Radiografías de Tórax y Explicabilidad
    ### Panel interactivo con Predicción Directa, Grad-CAM y Ground Truth (5 Clases)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_selector = gr.Dropdown(choices=image_names, label="Seleccionar Imagen")
            input_img = gr.Image(label="Radiografía Original", type="numpy", height=360)
            
            with gr.Row():
                btn = gr.Button("Analizar con Inteligencia Artificial", variant="primary")
                clear_btn = gr.Button("Limpiar", variant="secondary")
            
        with gr.Column(scale=1):
            output_text = gr.Markdown(value="### Esperando análisis...")
            
            with gr.Row():
                output_cam = gr.Image(label="Explicabilidad AI", height=300)
                output_mask = gr.Image(label="Máscara Real del Médico", height=300)

    image_selector.change(fn=load_preview_image, inputs=image_selector, outputs=input_img)
    btn.click(fn=predict_and_explain, inputs=[image_selector, input_img], outputs=[output_text, output_cam, output_mask])
    clear_btn.click(fn=clear_all, inputs=[], outputs=[image_selector, input_img, output_text, output_cam, output_mask])

if __name__ == "__main__":
   
    demo.launch(server_name="0.0.0.0", server_port=10000)