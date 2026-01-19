# AI Chatbot Interview & Explanation Guide

## 1. Project Overview
**What is this project?**
This is a custom retrieval-based Intent Classification Chatbot built from scratch using Python. It uses Deep Learning (PyTorch) to classify user input into actionable "intents" defined in a JSON file.

**Core Tech Stack:**
- **Python**: The programming language.
- **PyTorch**: The deep learning framework used to build and train the neural network.
- **NLTK (Natural Language Toolkit)**: Used for Natural Language Processing (NLP) tasks like tokenization and stemming.
- **NumPy**: Used for efficient numerical arrays (handling the "Bag of Words").

---

## 2. How It Works (The "Pipeline")
If an interviewer asks "How does your chatbot work?", describe this 3-step pipeline:

### Step 1: NLP Pre-processing (`nltk_utils.py`)
Computers don't understand text; they understand numbers. We convert text into numbers using:
1.  **Tokenization**: Breaking a sentence into individual words (tokens).
    *   *Example*: "How are you?" -> `["How", "are", "you", "?"]`
2.  **Stemming**: reducing words to their root form to handle variations.
    *   *Example*: "organizing", "organizes", "organized" -> all become `organ`.
    *   *Why?* It reduces the vocabulary size the model needs to learn.
3.  **Bag of Words (BoW)**: The vectorization technique.
    *   We create a vocabulary of *all* unique words in our dataset.
    *   We create a vector (array) of 0s. If a word from the sentence exists in our vocabulary, we put a `1` at that index.
    *   *Result*: A fixed-size array of 0s and 1s representing the sentence.

### Step 2: Training (`train.py` & `model.py`)
1.  **The Data**: `intents.json` contains "patterns" (input examples) and "tags" (labels).
2.  **The Model**: A Feed-Forward Neural Network (`NeuralNet` class in `model.py`).
    *   **Input Layer**: Size = number of unique words (vocabulary size).
    *   **Hidden Layers**: Two fully connected layers with ReLU activation (non-linearity).
    *   **Output Layer**: Size = number of distinct intents/tags (soft prediction).
3.  **Loss Function**: `CrossEntropyLoss` (standard for multi-class classification).
4.  **Optimizer**: `Adam` (adaptive learning rate optimization).
5.  **Output**: We save the trained model weights to `data.pth`.

### Step 3: Inference / Chatting (`chat.py`)
1.  Load the trained model (`data.pth`) and the `intents.json` file.
2.  Take User Input -> Tokenize -> Stem -> Bag of Words.
3.  Pass this vector into the Neural Network.
4.  The network outputs probabilities for each tag (e.g., specific: 90%, greeting: 5%, goodbye: 5%).
5.  If the highest probability is > 75%, pick a random response from that tag. Otherwise, say "I don't understand".

---

## 3. Key Concepts to Know for Interviews

**Q: Why did you use a custom network instead of ChatGPT/LLM?**
**A:** "I wanted to understand the fundamentals of NLP and how text classification works under the hood. Also, this approach is lightweight, offline, and gives me 100% control over the exact responses, which is critical for specific business use cases (like customer support FAQ) where hallucinations are not acceptable."

**Q: What is the activation function you used?**
**A:** "I used **ReLU (Rectified Linear Unit)** in the hidden layers because it introduces non-linearity and is computationally efficient. It helps the model learn complex patterns."

**Q: How do you handle words the bot hasn't seen before?**
**A:** "The 'Bag of Words' approach ignores unknown words because they aren't in the pre-defined vocabulary list. If a sentence contains *only* unknown words, the resulting vector is all zeros, and the model will likely output low confidence, triggering the fallback 'I don't understand' response."

**Q: What is a 'Bag of Words'?**
**A:** "It's a representation of text that describes the occurrence of words within a document. It ignores grammar and word order but captures word multiplicity. It turns variable-length text into fixed-length vectors suitable for a Neural Network."

---

## 4. Code Structure Breakdown

*   **`intents.json`**: The knowledge base. Modify this to add new skills.
*   **`nltk_utils.py`**: The toolkit. Contains `tokenize`, `stem`, and `bag_of_words` functions.
*   **`model.py`**: The brain. Defines the `NeuralNet` class (Input -> Linear -> Relu -> Linear -> Relu -> Linear).
*   **`train.py`**: The teacher. Loops through epochs, calculates loss, updates weights, and saves the file.
*   **`chat.py`**: The interface. Uses the saved brain to talk to humans.
