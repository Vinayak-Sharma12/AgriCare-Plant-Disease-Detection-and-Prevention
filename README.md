# AgriCare-Plant-Disease-Detection-and-Prevention
## **1. Overview**
The Plant Disease Recognition and Assistance System uses Deep learning and AI-powered tools to help identify plant diseases from images. This application allows users to upload images of plant leaves and predict potential diseases based on a pre-trained model. Additionally, the system integrates RAG (Retrieval-Augmented Generation) to answer user questions based on a document containing helpful agricultural information.The model have 95+% accuracy in detecting 

 DataSet Link -> https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
 
 Trained Model ->https://drive.google.com/drive/folders/12A5jDLj3RWhr12egESEhbmQVlMaMTUZx?usp=sharing
## **2.Feature**
- **Disease Recognition**: Upload images of plant leaves to detect diseases.
- **Question Answering**: Ask questions related to plant health, and get context-based answers from a reference document.
 -**Text-to-Speech**: Converts the answers into speech for better accessibility.

**How It Works**

**Disease Recognition**

**Upload Image**: Upload an image of a plant leaf.

**Prediction**: The system will process the image and use a pre-trained model to detect diseases.

**Result** : The system will display the predicted disease and provide further recommendations.

**Question Answering**

**Ask a Question**: Type in a question related to plant health.

**Answer Generation**: The system will retrieve context from a document and generate a precise answer.

**Text-to-Speech**: The answer is converted into speech for convenience.

Project Components

Disease Recognition Model

The app uses a Keras-based deep learning model trained to detect 38 different plant diseases. The model is built using TensorFlow and Keras.

RAG (Retrieval-Augmented Generation)

The RAG functionality is powered by LangChain with Cohere's LLM and FAISS for vector search. It processes the text file to split it into chunks, stores embeddings, and answers questions using context-based retrieval.

Text-to-Speech Integration

The app also integrates gTTS (Google Text-to-Speech) to convert the generated answers into audio for better user interaction.

## **3. Input / Output**
<img src="https://i.postimg.cc/tJzNMmqF/Screenshot-2024-11-24-at-1-14-24-PM.png" width="60%" height="60%">
<img src="https://i.postimg.cc/L6HT1RDD/Screenshot-2024-11-24-at-1-14-52-PM.png" width="60%" height="60%">
<img src="https://i.postimg.cc/wjMQwVSs/Screenshot-2024-11-24-at-1-15-20-PM.png" width="60%" height="60%">
<img src="https://i.postimg.cc/d0TmY0Qs/Screenshot-2024-11-24-at-1-15-36-PM.png" width="60%" height="60%">


