from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

import tensorflow as tf
import matplotlib.cm as cm

import cv2
import numpy as np
import gradio as gr
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred):

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)

    return (2.0 * intersection + 1) / (
        K.sum(y_true) + K.sum(y_pred) + 1
    )


def iou(y_true, y_pred):

    intersection = K.sum(y_true * y_pred)

    union = (
        K.sum(y_true)
        + K.sum(y_pred)
        - intersection
    )

    return (intersection + 1) / (union + 1)


def tversky(y_true, y_pred,
            alpha=0.7,
            smooth=1):

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    true_pos = K.sum(y_true * y_pred)

    false_neg = K.sum(
        y_true * (1 - y_pred)
    )

    false_pos = K.sum(
        (1 - y_true) * y_pred
    )

    return (
        true_pos + smooth
    ) / (
        true_pos
        + alpha * false_neg
        + (1 - alpha) * false_pos
        + smooth
    )


def focal_tversky_loss(
        y_true,
        y_pred,
        gamma=0.75):

    tv = tversky(y_true, y_pred)

    return K.pow(
        (1 - tv),
        gamma
    )


model = load_model(
    "model/resunet_brain_tumor.keras",
    custom_objects={
        "dice_coef": dice_coef,
        "iou": iou,
        "tversky": tversky,
        "focal_tversky_loss": focal_tversky_loss
    },
    compile=False
)

def create_pdf_report(
        original,
        mask,
        overlay,
        gradcam,
        diagnosis,
        tumor_pixels,
        percentage,
        severity,
        confidence):

    
    cv2.imwrite("original.png",
                cv2.cvtColor(original,
                             cv2.COLOR_RGB2BGR))

    cv2.imwrite("mask.png",
                cv2.cvtColor(mask,
                             cv2.COLOR_RGB2BGR))

    cv2.imwrite("overlay.png",
                cv2.cvtColor(overlay,
                             cv2.COLOR_RGB2BGR))

    cv2.imwrite("gradcam.png",
                cv2.cvtColor(gradcam,
                             cv2.COLOR_RGB2BGR))


    pdf_path = "Brain_Tumor_Report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    story = []

    title = Paragraph(
        "Brain Tumor Segmentation Report",
        styles['Title']
    )

    story.append(title)

    story.append(Spacer(1,20))

    
    data = [

        ["Parameter","Value"],

        ["Diagnosis",
         diagnosis],

        ["Tumor Area",
         f"{tumor_pixels} pixels"],

        ["Affected Region",
         f"{percentage:.2f}%"],
        
         ["Severity Level",
          severity],

        ["Confidence Score",
         f"{confidence:.2f}%"]

    ]

    table = Table(data)

    table.setStyle(
        TableStyle([

            ('BACKGROUND',
             (0,0),
             (-1,0),
             colors.lightblue),

            ('BOX',
             (0,0),
             (-1,-1),
             1,
             colors.black),

            ('GRID',
             (0,0),
             (-1,-1),
             1,
             colors.black)

        ])
    )

    story.append(table)

    story.append(Spacer(1,25))

    story.append(
        Paragraph(
            "Original MRI",
            styles['Heading2']
        )
    )

    story.append(
        Image(
            "original.png",
            width=180,
            height=180
        )
    )

    story.append(Spacer(1,15))

    story.append(
        Paragraph(
            "Predicted Mask",
            styles['Heading2']
        )
    )

    story.append(
        Image(
            "mask.png",
            width=180,
            height=180
        )
    )

    story.append(Spacer(1,15))

    story.append(
        Paragraph(
            "Overlay Image",
            styles['Heading2']
        )
    )

    story.append(
        Image(
            "overlay.png",
            width=180,
            height=180
        )
    )

    story.append(Spacer(1,15))

    story.append(
        Paragraph(
            "Grad-CAM Heatmap",
            styles['Heading2']
        )
    )

    story.append(
        Image(
            "gradcam.png",
            width=180,
            height=180
        )
    )

    story.append(Spacer(1,20))

    story.append(
        Paragraph(
"""
Recommendations

• Consult a neurologist.

• Seek expert MRI review.

• Additional investigations may be required.

⚠ Educational purposes only.
""",
            styles['BodyText']
        )
    )

    doc.build(story)

    return pdf_path



def make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0
    ) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def overlay_gradcam(
        heatmap,
        original):

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]

    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(
        jet_heatmap
    )

    jet_heatmap = jet_heatmap.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    jet_heatmap = tf.keras.preprocessing.image.img_to_array(
        jet_heatmap
    )

    superimposed = (
        jet_heatmap*0.4
        +
        original
    )

    return superimposed.astype(np.uint8)



IMG_SIZE = 256

def predict_tumor(image):

    original = image.copy()

    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]

    heatmap = make_gradcam_heatmap(
    img,
    model,
    "conv2d_27"
    )

    gradcam_image = overlay_gradcam(
    heatmap,
    cv2.resize(original,(IMG_SIZE,IMG_SIZE))
    )

    confidence = float(np.max(pred) * 100)

    mask = (pred > 0.5).astype(np.uint8)

    mask_display = np.repeat(mask, 3, axis=-1) * 255

    overlay = cv2.resize(original, (IMG_SIZE, IMG_SIZE)).copy()

    overlay[mask.squeeze() == 1] = [255,0,0]

    tumor_pixels = int(np.sum(mask))

    percentage = (tumor_pixels/(IMG_SIZE*IMG_SIZE))*100

    # Severity estimation
    if percentage == 0:

      severity = "🟢 None"

    elif percentage < 1:

      severity = "🟢 Mild"

    elif percentage < 5:

      severity = "🟡 Moderate"

    else:

      severity = "🔴 Severe"

    if tumor_pixels > 0:

        diagnosis = """
# 🔴 Tumor Detected
"""

        diagnosis_text = "Tumor Detected"

        info = f"""
## Tumor Information

- Tumor Area: **{tumor_pixels} pixels**

- Affected Region: **{percentage:.2f}%**

- Severity Level: **{severity}**

- Confidence Score: **{confidence:.2f}%**
"""

        recommendations = """
## Recommendations

Consult a neurologist

Seek expert MRI review

Additional investigations may be required

⚠ Educational purposes only
"""

    else:

        diagnosis = """
# 🟢 No Tumor Detected
"""

        diagnosis_text = "No Tumor Detected"

        info = f"""
## Tumor Information

- Tumor Area: **0 pixels**

- Affected Region: **0%**

- Severity Level: **{severity}**

- Confidence Score: **{confidence:.2f}%**
"""

        recommendations = """
## Recommendations

No suspicious region detected

If symptoms persist, consult a medical professional

⚠ Educational purposes only
"""


    pdf_file = create_pdf_report(

    cv2.resize(original,(IMG_SIZE,IMG_SIZE)),

    mask_display,

    overlay,

    gradcam_image,

    diagnosis_text,

    tumor_pixels,

    percentage,

    severity,

    confidence

    )

    return (

        cv2.resize(original,(IMG_SIZE,IMG_SIZE)),

        mask_display,

        overlay,

        gradcam_image,

        diagnosis,

        info,

        recommendations,

        mask_display,

        overlay,

        confidence,

        pdf_file

    )




with gr.Blocks(
    theme=gr.themes.Monochrome(),
    title="NeuroVision AI"
) as demo:

    gr.Markdown("""
#  NeuroVision AI

## Brain Tumor Segmentation using ResUNet

Deep Learning Based MRI Tumor Localization and Analysis

⚠ Educational purposes only.
""")

    with gr.Tab("MRI Analysis"):

        input_image = gr.Image(
            label="Upload MRI Image",
            type="numpy",
            height=250
        )

        analyze_btn = gr.Button(
            "Analyze MRI",
            variant="primary"
        )

        gr.Markdown("## Prediction Results")

      
        with gr.Row():

            original_output = gr.Image(
                label="Original MRI",
                height=300
            )

            mask_output = gr.Image(
                label="Predicted Mask",
                height=300
            )

      
        with gr.Row():

            overlay_output = gr.Image(
                label="Overlay Image",
                height=300
            )

            gradcam_output = gr.Image(
                label="Grad-CAM Heatmap",
                height=300
            )

        gr.Markdown("---")

        with gr.Row():

            diagnosis_output = gr.Markdown()

            info_output = gr.Markdown()

        recommendations_output = gr.Markdown()

        gr.Markdown("## Confidence Score")

        confidence_output = gr.Slider(
            minimum=0,
            maximum=100,
            label="Confidence (%)",
            interactive=False
        )

        gr.Markdown("## Downloads")

        with gr.Row():

            download_mask = gr.Image(
                label="Download Predicted Mask"
            )

            download_overlay = gr.Image(
                label="Download Overlay Image"
            )

        report_output = gr.File(
            label="Download Medical Report"
        )

    with gr.Tab("About"):

        gr.Markdown("""
# Developer

### Barathwaj R S

B.Tech CSE (AI & DS)

SASTRA University

---

## Model

ResUNet

---

## Dataset

LGG MRI Segmentation Dataset

---

## Frameworks

- TensorFlow
- Keras
- OpenCV
- Gradio

---

## Features

Tumor Detection

Tumor Segmentation

Overlay Visualization

Grad-CAM Explainability

Confidence Estimation

PDF Report Generation
""")

    analyze_btn.click(
        fn=predict_tumor,
        inputs=input_image,
        outputs=[

            original_output,

            mask_output,

            overlay_output,

            gradcam_output,

            diagnosis_output,

            info_output,

            recommendations_output,

            download_mask,

            download_overlay,

            confidence_output,

            report_output

        ]
    )

if __name__ == "__main__":
    demo.launch()