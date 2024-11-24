import streamlit as st
import tensorflow as tf
import numpy as np
import gtts
from io import BytesIO

# Integerating RAG in This 
import streamlit as st
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

TEXT_FILE_PATH = "Document 17 (1).txt"

def text_file_to_text(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()
    return text
def text_splitter(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=600,  # This overlap helps maintain context between chunks
        length_function=len,
        separators=['\n', '\n\n', ' ', ',']
    )
    chunks = text_splitter.split_text(text=raw_text)
    return chunks
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(question, retriever):
    cohere_llm = Cohere(model="command", temperature=0.1, cohere_api_key='sRmFY97EVTJa7VaaaQha5oH7lScl1rxTZv8x6KrV')

    prompt_template = """Answer the question as precisely as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" "\n\n
                    Context: \n {context} \n\n
                    Question: \n {question} \n
                    Answer:"""

    prompt = PromptTemplate.from_template(template=prompt_template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)

#Tensorflow Model Prediction
def model_prediction(test_image):
    model=tf.keras.models.load_model('trained_plant_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions) 
    return result_index

#SideBar
st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])


#HomePage
if app_mode=="Home":
    st.header("Plant Disease Recognition System")
    image_path="/Users/vinayak/Desktop/app/Screenshot 2024-11-24 at 10.31.02 AM.png"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

    #About Page
elif app_mode=="About":
        st.header("About")
        st.markdown(""" #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)
#Prediction Page
elif app_mode=="Disease Recognition":
      st.header("Disease Recognition")
      test_image=st.file_uploader("Input a image")
      if st.button("Show Image"):
            st.image(test_image,use_container_width=True)
     #Predict Buttom
      if st.button("Predict"):
            st.write("Our Prediction")
            result_index=model_prediction(test_image)
            #Define Class
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))

if app_mode=="Disease Recognition":
    question = st.text_input("Ask a question:")
    if st.button("Ask"):
            with st.spinner("Let me help you :)"):
                # Open and process the text file
                raw_text = text_file_to_text(TEXT_FILE_PATH)
                text_chunks = text_splitter(raw_text)
                vectorstore = get_vector_store(text_chunks)
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

                if question:
                    answer = generate_answer(question, retriever)
                    st.write(answer)
                    tts=gtts.gTTS(answer) #gTSS is a class
                    tts.save("hello.mp3")
                    st.audio("hello.mp3")

            


        



