
import torch
import numpy as np
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"using device: {device}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from transformers import AutoModelForCausalLM, AutoTokenizer


Phi_model_name = "microsoft/Phi-4-mini-instruct"
Phi_tokenizer = AutoTokenizer.from_pretrained(Phi_model_name)
Phi_model = AutoModelForCausalLM.from_pretrained(Phi_model_name, device_map="auto", torch_dtype=torch.float16)


gpt2_model_name = "gpt2"  # GPT-2 is used here for demonstration
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''

AI-Powered Document Search and Summarization System.
This project covers document ingestion, embedding generation, vector database indexing, search retrieval, text summarization, and evaluation.

'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''
Full Project Code: AI-Powered Document Search and Summarization System


Step 1: Install Dependencies
Make sure you have the required libraries installed.

pip install sentence-transformers faiss-cpu transformers pdfplumber python-docx rouge-score nltk

sentence-transformers is used for efficient text embeddings.

faiss-cpu is an optimized tool for searching vectorized documents quickly.

transformers helps load models like BART and T5 for summarization.

pdfplumber and python-docx are used to extract text from PDFs and Word documents.

'''


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
Document Ingestion
Extract text from PDFs and Word documents.

This part extracts text from different document types (PDFs and Word files).

pdfplumber loops through PDF pages and extracts text.

python-docx extracts text from .docx files by iterating through paragraphs.

The final extracted text is stored as a list, where each entry contains the text from a document.
'''

'''
Load and Process Large Text File
This function reads a large text file and splits it into manageable sections.

'''

import pdfplumber
import docx


full_text = None
question = None
sentences = None

# -----------------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# -----------------------------------------
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs])


# -----------------------------------------
def extract_text_from_text_files(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()
    

# -----------------------------------------

from tkinter import Tk
from tkinter.filedialog import askopenfilename


def select_filename():
    """Selects a file name for saving the extracted text"""
    
    # display a Windows file dialogbox to select a file
    Tk().withdraw()  # Hide the root window

    # Prompt the user to select a file
    filename = askopenfilename(title="Select a file", filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf"), ("Word files", "*.docx")])
    
    if not filename or filename == "":
        return None

    filename = filename.replace("/", "\\")  # Replace forward slashes with backslashes for Windows compatibility
    print(f"\nSelected file: {filename}")
    filename = filename.lower()  # Convert to lowercase for case-insensitive comparison
    
    # Check if the file is a supported type
    if filename.endswith('.pdf') or filename.endswith('.docx') or filename.endswith('.txt'):
        return filename
    else:
        print("Unsupported file type. Please select a .txt, .pdf, or .docx file.")
        return None  # Return None or handle the error as needed
                


# -----------------------------------------

# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def load_file():

    global full_text
    global sentences

    while True:
    
        filename = select_filename()  # Select a file name for saving the extracted text
        
        if not filename or filename == "":
            print("No file selected. Try again...")
            continue

        if filename.endswith('.pdf'):
            full_text = extract_text_from_pdf(filename)
        elif filename.endswith('.docx'):
            full_text = extract_text_from_docx(filename)
        elif filename.endswith('.txt'):
            full_text = extract_text_from_text_files(filename)
        else:
            print("\nUnsupported file type. Please select a .txt, .pdf, or .docx file.")
            continue

        break
            
    sentences = sent_tokenize(full_text)  # Splits text into sentences

    print("\nDocument was loaded ...\n")
    input("To watch its content - Press ENTER to continue...")

    print("\nDocument content: (Based on tokenization)")
    print("-------------------------------------------\n")

    # print all the sentences in the document
    for i, sentence in enumerate(sentences):
        print(f"{i + 1}: {sentence}")
    
    
    input("\n\nPress ENTER to continue...")
    print("\nDocument statistics:")
    print("----------------------\n")

    print(f"Document name: {filename}")
    print(f"Document length: {len(full_text)} characters")
    print(f"Document length: {len(full_text.split())} words")
    print(f"Document length: {len(full_text.split('.'))} sentences")
    full_text_split_newline = full_text.split('\n')
    print(f"Document length: {len(full_text_split_newline)} lines")
    
    print("\n")
    input("Press ENTER to continue...")
    



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def create_embeddings_and_index():
    
    global embed_model
    global index
    global sentences_embeddings

    '''


    Convert Text to Embeddings
    Generate text embeddings using sentence-transformers.

    This step converts each document's text into numerical representations (embeddings).

    sentence-transformers provides a pretrained model (all-MiniLM-L6-v2) to generate embeddings.

    The encode function transforms each document into high-dimensional vectors.

    These embeddings allow for similarity-based retrieval.

    '''

    print("\nCreating FAISS index and embeddings ...")

    input("\nPress ENTER to continue...")


    from sentence_transformers import SentenceTransformer

    # Load embedding model
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Convert each text chunk into an embedding
    sentences_embeddings = [embed_model.encode(sentence) for sentence in sentences]


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    '''
    Store Embeddings in FAISS
    Create an efficient vector storage for search.

    FAISS indexes embeddings efficiently for fast searches.

    The embeddings are converted into a NumPy array (vectors).

    IndexFlatL2(dimension) creates a FAISS index, storing the embeddings using L2 (Euclidean) distance for similarity searches.

    The embeddings are added to FAISS with index.add(vectors).

    '''

    import faiss

    dimension = len(sentences_embeddings[0])  # Size of embedding vectors
    index = faiss.IndexFlatL2(dimension)  # L2 distance-based FAISS index

    vectors = np.array(sentences_embeddings).astype('float32')  # Convert to numpy array
    index.add(vectors)  # Add embeddings to FAISS index

    print("\nFAISS index created and embeddings added.")
    print(f"Number of embeddings in FAISS index: {index.ntotal}")
    print(f"FAISS index dimension: {index.d}")
    print()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def display_sentences():
    """Display the sentences in the document"""
    print("Document Sentences:")
    for i, sentence in enumerate(sentences):
        print(f"{i + 1}: {sentence}")
    print("End of Document Sentences")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
Search Query and Retrieve Relevant Sections
Retrieve the most relevant content using similarity search.

A user enters a search query, which is converted into an embedding (query_embedding).

The FAISS index searches for the closest embedding to the query using index.search().

distances stores similarity scores.

best_match_idx returns the document index that best matches the search query.

'''

# -----------------------------------------


def retrieve_relevant_sentence_L2_distance_based_FAISS(query, threshold=0.8):
    """Finds relevant text chunk for a given query, filters weak matches"""

    
    query_embedding = embed_model.encode(query)
    query_vector = np.array([query_embedding]).astype('float32')

    distances, best_match_idx = index.search(query_vector, 1)  # Find best match

    # Check the similarity score (FAISS uses L2 distance, lower is better)
    if distances[0][0] > threshold:  # Ignore weak matches
        return None  # No relevant result found
    
    return sentences[best_match_idx[0][0]]  # Return matching text chunk

# -----------------------------------------

import torch
from sentence_transformers.util import cos_sim

def retrieve_relevant_sentence_cosine_similarity(query, threshold=0.6):

    """Uses cosine similarity for better relevance filtering"""
    query_embedding = embed_model.encode(query)
    scores = cos_sim(query_embedding, sentences_embeddings)  # Compare query vs document

    best_match_idx = torch.argmax(scores).item()
    best_match_score = scores[0][best_match_idx].item()

    # If similarity score is below threshold, reject the result
    if best_match_score < threshold:
        return None

    return sentences[best_match_idx]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generate_answer(model, tokenizer, prompt):
    
    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)

    # Generate the model's output
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            temperature=0.1,  # Low temperature for more deterministic outputs
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,  # Discourage repetition
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the model's reply (removing the prompt)
    reply = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]  #  + len("Answer : ")
    
    return reply.strip()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def single_run(model, tokenize, input_text):
  
  prompt = input_text
  model_output = generate_answer(model, tokenize, prompt)
  return model_output


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def check_if_a_document_was_loaded():
    
    """Check if a document was loaded before searching"""
    if not full_text:
        print("\n===> ERROR: You need to load a document first.")
        return False
    return True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def define_question():
    
    """Define a question to ask regarding the document"""

    global question

    # the user will enter a question
    while True:
        question = input("\nPlease ask your question: ")
        if question:
            break
        print("\n===> ERROR: You need to enter a valid question.  Please enter your qeustion again.")
        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def check_if_a_question_was_asked():

    """Check if a question was asked before searching"""
    if not question:
        print("\n===> ERROR: You need to ask a question first.")
        return False
    return True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import requests

HUGGINGFACE_API_KEY = "YOUR_HUGGINGFACE_API_KEY"

HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Hosted LLMs:
# model_name = "TinyLlama/TinyLlama-1B"
# model_name = "tiiuae/falcon-7b-instruct"
# model_name = "cutycat2000/MeowGPT-2"
# model_name = "EleutherAI/gpt-j-6B"
model_name = "facebook/bart-large-cnn"  # initial and default model (the user can change this at runtime)
# model_name = "microsoft/Phi-4-mini-instruct"


def query_model_on_a_hosted_LLM(text, model_name):

    data = {"inputs": text}
        
    ENDPOINT = f"https://api-inference.huggingface.co/models/{model_name}"
    print("\nThe endpoint is:")
    print(ENDPOINT)
    print()

    count = 0
    try:
        while True:
            count += 1
            if count > 20:
                print("Too many errors. Exiting...")
                return None
            # Send the request to the Hugging Face API
            response = requests.post(ENDPOINT, headers=HEADERS, json=data)
            if response.status_code == 200:
                break
            else:
                print("Error code: ", response.status_code, " - ", response.text)
                print("Retrying...")
                time.sleep(1)
                
    except requests.exceptions.RequestException as e:
        print("Error code: ", response.status_code, " - ", response.text)
        print(f"Error: {e}. Retrying...")
        return None

    reply = response.json()

    return reply[0]['summary_text']


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def summarize_on_a_hosted_LLM(model_name):

    
    prompt = f"Can you please summarize the following text:\n\n{full_text}\n\n"
    data = {"inputs": prompt}
        
    ENDPOINT = f"https://api-inference.huggingface.co/models/{model_name}"
    print("\nThe endpoint is:")
    print(ENDPOINT)
    print()

    count = 0
    try:
        while True:
            count += 1
            if count > 20:
                print("Too many errors. Exiting...")
                return None
            # Send the request to the Hugging Face API
            response = requests.post(ENDPOINT, headers=HEADERS, json=data)
            if response.status_code == 200:
                break
            else:
                print("Error code: ", response.status_code, " - ", response.text)
                print("Retrying...")
                time.sleep(1)
                
    except requests.exceptions.RequestException as e:
        print("Error code: ", response.status_code, " - ", response.text)
        print(f"Error: {e}. Retrying...")
        return None

    reply = response.json()

    return reply[0]['summary_text']

    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def search_using_phi():

    """Search for a specific sentence using Phi model"""

    prompt = f"Based on the following document, answer the question:\n\n{full_text}\n\nQuestion: {question}"
    out = single_run(Phi_model, Phi_tokenizer, input_text=prompt)
    return out

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def search_using_gpt2():

    """Search for a specific sentence using GPT-2 model"""

    prompt = f"Based on the following document, answer the question:\n\n{full_text}\n\nQuestion: {question}"
    
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")

    output = gpt2_model.generate(**inputs)
    
    response = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

    reply = response[len(gpt2_tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]  #  + len("Answer: ") + 2

    return reply.strip()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def just_ask_a_general_question(question):
    """Ask a question to the model without any context"""

    prompt = f"Question: {question}"
    out = single_run(Phi_model, Phi_tokenizer, input_text=prompt)
    return out


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def summarize_with_Phi():
    """Summarize the document using Phi model"""

    prompt = f"Can you please summarize the following text:\n\n{full_text}\n\n"
    out = single_run(Phi_model, Phi_tokenizer, input_text=prompt)
    return out

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def print_separator():
    print("---------------------------------------------------------------------------------")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def get_user_menu_choice():

    print ("\n====================================================================")
    print("Please choose an option from the menu.  (Press ENTER to continue)")
    input()

    print ("")
    print ("  MENU:")
    print ("==============================================================")
    print ("|  ")
    print ("|  DOCUMENT:")
    print ("|  ----------------")
    print ("|  (L) Load: Load a document into the system")
    print ("|  ")
    print ("|  ")
    print ("|  DOCUMENT SEARCH:")
    print ("|  -----------------")
    print ("|  (A) Ask a question regarding the document")
    print ("|  ")
    print ("|  (1) Search: Looking for a specific sentence - using FAISS L2 distance")
    print ("|  (2) Search: Looking for a specific sentence - using cosine similarity")
    print ("|  ")
    print ("|  (3) Search: Searching all file text - using local LLM - Phi")
    print ("|  (4) Search: Searching all file text - using local LLM - GPT2")
    print ("|  ")
    print ("|  ")
    print ("|  SUMMARIZE:")
    print ("|  ----------")
    print ("|  (S) Summarize: Summarize the document - using local LLM - Phi")
    print ("|  (M) Summarize: Summarize the document - using a hosted LLM")
    print ("|  ")
    print ("|  ")
    print ("|  ASK THE LLM A GENERAL QUESTION:")
    print ("|  --------------------------------")
    print ("|  (G) Ask the LLM a general question (not regarding the document) - Local LLM - Phi")
    print ("|  (H) Ask the LLM a general question (not regarding the document) - using a hosted LLM") 
    print ("|  ")
    print ("|  ")
    print ("|  MANAGE:")
    print ("|  ----------")
    print ("|  (Q) *** QUIT ***")
    print ("|  ")
    
    return input("\nWhat do you want to do ? ").lower()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def main():

    global model_name

    while True:

        while True:        
            selection = get_user_menu_choice()
            if selection and (selection in "1234lasqghm"):
                break
        
        # quit
        if selection == 'q':
            print("")
            print ("\nQuiting the program.\n\n")
            break

        # load a file
        if selection == 'l':
            load_file()
            create_embeddings_and_index()
            # display_sentences()

        # ask a question
        if selection == 'a':
            define_question()

        
        # FAISS L2 distance
        if selection == '1':
            if check_if_a_document_was_loaded() and check_if_a_question_was_asked():
                retrieved_text = retrieve_relevant_sentence_L2_distance_based_FAISS(question)
                if retrieved_text is None:
                    print(f"\nRegarding your question:")
                    print(question)
                    print("No relevant text was found.")
                else:
                    print(f"\nRegarding your question:")
                    print(question)
                    print("\nThis relevant text was found:")
                    print(retrieved_text)
                    print()
                

        # cosine similarity
        if selection == '2':
            if check_if_a_document_was_loaded() and check_if_a_question_was_asked():
                retrieved_text = retrieve_relevant_sentence_cosine_similarity(question)
                if retrieved_text is None:
                    print(f"\nRegarding your question:")
                    print(question)
                    print("No relevant text was found.")
                else:
                    print(f"\nRegarding your question:")
                    print(question)
                    print("\nThis relevant text was found:")
                    print(retrieved_text)
                    print()
                

        # LLM - Phi
        if selection == '3':
            if check_if_a_document_was_loaded() and check_if_a_question_was_asked():
                answer = search_using_phi()
                print(f"\nRegarding your question:")
                print(question)
                print("\nThis is the answer:")
                print(answer)
                print()

        # LLM - GPT2
        if selection == '4':
            if check_if_a_document_was_loaded() and check_if_a_question_was_asked():
                answer = search_using_gpt2()
                print(f"\nRegarding your question:")
                print(question)
                print("\nThis is the answer:")
                print(answer)
                print()


        # summarize
        if selection == 's':
            if check_if_a_document_was_loaded():
                answer = summarize_with_Phi()
                print(f"\nThis is the summary of the document:\n")
                print(answer)
                print()

        
        # simply ask a general question - local LLM - Phi
        if selection == 'g':

            # the user will enter a question
            while True:
                general_question = input("\nPlease ask a general question: ")
                if general_question:
                    break
                print("\n===> ERROR: You need to enter a valid question.  Please enter your qeustion again.")
            
            
            answer = just_ask_a_general_question(general_question)
            print_separator()
            print(f"\nRegarding your question, this is the answer:")
            print(f"---------------------------------------------:\n")
            print(answer)
            print()


        # simply ask a general question - hosted LLMs - using Hugging Face API
        if selection == 'h':

            # the user will enter a question
            while True:
                general_question = input("\nPlease ask the LLM a general question: ")
                if general_question:
                    break
                print("\n===> ERROR: You need to enter a valid question.  Please enter your qeustion again.")
            
            
            # the user will enter a model name
            print(f"\nThe current LLM being used is:\n{model_name}")
            print("\nIf you wish, you may enter a different model name that would become the default.")
            a_new_model_name = input("\nOtherwise, you can just press ENTER: ")
            
            if a_new_model_name and a_new_model_name != "":
                model_name = a_new_model_name
                print(f"\nThe new LLM would now be: {model_name}")

            answer = query_model_on_a_hosted_LLM(general_question, model_name)
            
            print_separator()
            print(f"\nRegarding your question, this is the answer:")
            print(f"---------------------------------------------\n")
            print(answer)
            print()


        # summarize the document - hosted LLMs - using Hugging Face API
        if selection == 'm':

            if check_if_a_document_was_loaded():
                # the user will enter a model name
                print(f"\nThe current LLM being used is:\n{model_name}")
                print("\nIf you wish, you may enter a different model name that would become the default.")
                a_new_model_name = input("\nOtherwise, you can just press ENTER: ")
                
                if a_new_model_name and a_new_model_name != "":
                    model_name = a_new_model_name
                    print(f"\nThe new LLM would now be: {model_name}")

                answer = summarize_on_a_hosted_LLM(model_name)
                
                print_separator()
                print(f"\nRegarding your document, this is the summary:")
                print(f"-----------------------------------------------\n")
                print(answer)
                print()
    

    
            




main()




