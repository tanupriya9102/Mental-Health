import joblib
import requests
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import requests
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import torch.utils.data
from torch.utils.data import TensorDataset
from transformers import TFRobertaModel, RobertaTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
# Set page title and icon
st.set_page_config(
    page_title="Mental Health Prediction",
    page_icon="🌈"
)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Set app-wide style
# st.markdown(
#     """
#     <style>
#         body {
#             background-color: #ff3df3;
#             color: #333;
#             font-family: 'Arial', sans-serif;
#         }
#         .st-bw {
#             background-color: #ff3df3;
#             color: #fff;
#             padding: 10px;
#             border-radius: 5px;
#         }
#         .st-cc {
#             text-align: center;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Write Title with some styling
st.title("Mental Health Prediction")
st.markdown("---")



#load model, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('Models\MH_Model.h5', custom_objects={'TFRobertaModel': TFRobertaModel})
    return model

model=load_model()




tokenizer = joblib.load('Models\MH_tokenizer.joblib')
# model = tf.keras.models.load_model('models/MH_Model.h5', custom_objects={'TFRobertaModel': TFRobertaModel})


def classify_response(question):

    encoded_question = tokenizer.encode_plus(
        question,
        max_length=256,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='tf'
    )
    input_ids = encoded_question['input_ids']
    attn_mask = encoded_question['attention_mask']

    prediction = model.predict([input_ids, attn_mask])
    predicted_class_num = tf.argmax(prediction, axis=1).numpy()[0]

    class_names = ["Depression", "Anxiety", "Frustration", "Stress"]
    predicted_class_name = class_names[predicted_class_num]

    if question=="" or question.isspace():
      return("Please enter a valid answer!!")

    else:
      return(predicted_class_name)
    

# #Write Text 
# st.write( "Do you ever worry about the security of your job?")

# #Display an Image
# # st.image("content")

# #Text Field
# response= st.text_input(" ","default value")
# st.write("Predicting Class...")
# with st.spinner("Classifying..."):
#    predicted_label=classify_response(response)
#    st.write(predicted_label)



# List of questions
questions = [
    "Do you ever worry about the security of your job?",
    "Do you feel confident at work?",
    "Have you started to avoid spending time with friends and loved ones due to extensive work?",
    "Do you feel like you have a good work-life balance here?",
    "Do you become irritable or annoyed more quickly than in the past?",
    "Do you often feel restless, on edge, or unable to relax?",
    "Do you feel comfortable talking about your mental health with others inside our organization?",
    "Is it difficult to fall asleep, get enough sleep, or wake up on time most days?",
    "Do you ever feel overworked or underworked here as an employee?",
    "Do you feel that your work is not recognized or underappreciated?"
]

# # Display questions and collect responses
# user_responses = []
# for i, question in enumerate(questions):
#     st.write(f"Q{i + 1}: {question}")
#     response = st.text_input(f"Your response to Q{i + 1}")
#     user_responses.append(response)

# # Predict label based on all responses
# st.write("Predicting Class...")
# with st.spinner("Classifying..."):
#     predicted_labels = [classify_response(response) for response in user_responses]
#     for i, predicted_label in enumerate(predicted_labels):
#         st.write(f"Q{i + 1}: {predicted_label}")
#     predicted_labels = [classify_response(response) for response in user_responses]
#     label_counts = pd.Series(predicted_labels).value_counts()

# # Plot the pie chart using Matplotlib
# st.pyplot(plt.pie(label_counts, labels=label_counts.index, autopct='%1.2f%%', startangle=90, shadow=True, explode=(0.1,) * len(label_counts)))
# st.pyplot(plt.title('Composition of Labels for All Questions'))





# # Display questions and collect responses
# user_responses = []
# for i, question in enumerate(questions):
#     st.write(f"Q{i + 1}: {question}")
#     response = st.text_input(f"Your response to Q{i + 1}")
#     user_responses.append(response)

# # Predict label based on all responses
# st.write("Predicting Class...")
# with st.spinner("Classifying..."):
#     predicted_labels = [classify_response(response) for response in user_responses]
    
#     label_counts = pd.Series(predicted_labels).value_counts()

# # Plot the pie chart using Matplotlib
# fig, ax = plt.subplots()
# ax.pie(label_counts, labels=label_counts.index, autopct='%1.2f%%', startangle=90, shadow=True, explode=(0.1,) * len(label_counts))
# ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# # Save the figure and display it in Streamlit
# st.pyplot(fig)
# st.title('Composition of Labels for All Questions')





# Display questions and collect responses
user_responses = []
predicted_labels = []
for i, question in enumerate(questions):
    st.markdown(f"**Q{i + 1}: {question}**")
    response = st.text_input(f"Your response to Q{i + 1}")
    user_responses.append(response)

    # Classify individual answers
    st.markdown(f"**Predicted Label for Q{i + 1}:**")
    with st.spinner("Classifying..."):
        predicted_label = classify_response(response)
        predicted_labels.append(predicted_label)
        st.write(predicted_label)
    st.markdown("---")

# Display composition graph at the end
st.title('Composition of Labels for All Questions')
st.markdown("---")

# Plot the pie chart using Matplotlib
label_counts = pd.Series(predicted_labels).value_counts()
fig, ax = plt.subplots()
ax.pie(label_counts, labels=label_counts.index, autopct='%1.2f%%', startangle=90, shadow=True, explode=(0.1,) * len(label_counts))
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Save the figure and display it in Streamlit
st.pyplot(fig)