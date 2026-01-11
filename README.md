# AI Chatbot with PyTorch

A custom AI Chatbot built with Python, PyTorch, and NLTK. It uses a Feed Forward Neural Network to understand natural language and strictly follows user-defined intents.

## Features
- **Customizable Intents**: Easily modify the `intents.json` file to change the bot's responses and personality.
- **Deep Learning**: Uses a PyTorch Neural Network trained on your specific patterns.
- **Natural Language Processing**: implementation of tokenization and stemming using NLTK.
- **Easy to Run**: Includes a simple batch script to get started immediately.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vaibhav1817/ai-chatbot.git
   ```
2. Install dependencies:
   ```bash
   pip install torch nltk numpy
   ```
   *(Note: You may need to install the specific version of PyTorch for your system from [pytorch.org](https://pytorch.org/))*

3. Download NLTK data (run in python):
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage

### 1. Chatting
To start chatting with the bot, simply run the included batch file:
```bash
run.bat
```
Or run the python script directly:
```bash
python chat.py
```

### 2. Training
If you modify `intents.json`, you need to retrain the model:
```bash
python train.py
```
This will generate a new `data.pth` file.

## Customization
Edit `intents.json` to add new tags, patterns (input patterns to match), and responses.

Example structure:
```json
{
  "tag": "greeting",
  "patterns": ["Hi", "Hello"],
  "responses": ["Hi there!", "Hello!"]
}
```

## Credits
Developed by Vaibhav.
