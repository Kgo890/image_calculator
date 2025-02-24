import streamlit as st
import tempfile
from predict import predict  # Import the predict function

st.title("Handwritten Equation Solver")

uploaded_file = st.file_uploader("Upload an equation image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name  # Get the temporary file path

    # Call the predict function with the correct file path
    equation, result = predict(temp_path)

    # Display results
    st.write(f"Predicted Equation: {equation}")
    st.write(f"Result: {result}")
