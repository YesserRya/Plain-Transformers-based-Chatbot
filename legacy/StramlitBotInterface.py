import streamlit as st
from Transformer import*

# Function to generate a response from the chatbot
def generate_response(user_input):
    response = predict(user_input, loaded_model) 
    return response

# Streamlit app
def main():
   
    st.title("My Chatbot")

    user_input = st.text_input("User Input", "")
    if st.button("Send"):
        response = generate_response(user_input)
        st.text_area("Chatbot Response", value=response, height=200)

if __name__ == "__main__":
    main()

