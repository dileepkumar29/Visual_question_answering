import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model

processor, model = load_model()

st.title("🖼️ Visual Question Answering (VQA)")
st.write("Upload an image and ask a question about it.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Question input
    question = st.text_input("Ask a question about the image:")

    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                inputs = processor(image, question, return_tensors="pt")

                with torch.no_grad():
                    outputs = model.generate(**inputs)

                answer = processor.decode(outputs[0], skip_special_tokens=True)

            st.success(f"Answer: {answer}")