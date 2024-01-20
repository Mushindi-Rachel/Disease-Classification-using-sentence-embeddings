import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer


#Loading the model
model = pickle.load(open('classification_model.sav', 'rb'))
class_model = pickle.load(open('classification_model.sav', 'rb'))
label_encoder = pickle.load(open('label_encoder.sav', 'rb'))

def main():
#Creating UI
st.title('Disease Classification App')

symptoms = st.text_area('Enter your symptoms')

#When the button is pressed, it triggers disease classification
    if st.button("Classify Disease"):
        if symptoms:
            # Encode the input symptom_text
            temp_encoding = model.encode(symptom_text)

            # Make a prediction using the loaded classifier
            temp_prediction = class_model.predict([temp_encoding])

            # Inverse transform the predicted label to get the disease name
            temp_label = label_encoder.inverse_transform(temp_prediction)

            # Display the result
            st.success(f"You might be suffering from: {temp_label[0]}")
        else:
            st.warning("Please enter symptoms.")

if __name__ == '__main__':
    main()
