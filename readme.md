# 🦙💬 Llama Streamlit Chat

A powerful, feature-rich chatbot application built with Streamlit and Meta's Llama-3-8B-Instruct model. Features multi-session chat management, automatic conversation naming, and real-time streaming responses.

## ✨ Features

- 🤖 **Powered by Llama 3**: Uses Meta's Llama-3-8B-Instruct model via HuggingFace
- 💬 **Multi-Session Management**: Create and manage multiple chat conversations
- 🏷️ **Smart Auto-Naming**: Conversations automatically name themselves based on content
- ⚡ **Real-time Streaming**: Watch responses generate token-by-token
- 💾 **Persistent Memory**: Each chat maintains its own conversation history
- ✏️ **Rename & Delete**: Easily manage your chat sessions
- 🎨 **Clean UI**: Beautiful, intuitive interface built with Streamlit

## 📋 Prerequisites

- Python 3.8 or higher
- HuggingFace account and API token
- Basic knowledge of Python and command line

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/tarunpatelofficial/llama-streamlit-chat.git
cd llama-streamlit-chat
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the root directory:
```bash
HF_TOKEN=your_huggingface_token_here
```

To get your HuggingFace token:
- Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
- Create a new token with read access
- Copy and paste it into your `.env` file

## 📦 Requirements

Create a `requirements.txt` file with the following:

```txt
streamlit>=1.28.0
python-dotenv>=1.0.0
langchain>=0.1.0
langchain-huggingface>=0.0.1
langgraph>=0.0.1
```

## 🎮 Usage

1. **Run the application**
```bash
streamlit run app.py
```

2. **Open your browser**

The app will automatically open at `http://localhost:8501`

3. **Start chatting!**
- Type your message in the chat input
- Watch the AI respond in real-time
- Create new chats with the "➕ New Chat" button
- Rename chats with the ✏️ button
- Delete unwanted chats with the 🗑️ button

## 🏗️ Project Structure

```
llama-streamlit-chat/
│
├── app.py              # Main application file
├── .env                # Environment variables (create this)
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore file
```

## 🎯 How It Works

### Multi-Session Management
- Each chat session maintains its own conversation history
- Sessions are stored in Streamlit's session state
- Unique chat IDs ensure proper context management

### Auto-Naming System
- After the first message exchange, the chat analyzes the conversation
- Extracts key topics using keyword extraction
- Automatically generates a concise, descriptive title
- Updates the sidebar with the new name

### Streaming Responses
- Uses LangChain's streaming capabilities
- Displays tokens as they're generated
- Provides real-time feedback to users

## ⚙️ Configuration

You can customize the chatbot behavior by modifying the `SYSTEM_PROMPT` in `app.py`:

```python
SYSTEM_PROMPT = """
You are a helpful, friendly, and knowledgeable AI assistant.
# Customize your bot's personality here
"""
```

You can also adjust model parameters:

```python
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.1,        # Adjust randomness (0.0-1.0)
    max_new_tokens=512,     # Maximum response length
)
```

## 🐛 Troubleshooting

### "Failed to initialize chatbot"
- Make sure your HF_TOKEN is correctly set in the `.env` file
- Verify you have accepted the Llama model license on HuggingFace
- Check your internet connection

### Slow responses
- The model runs on HuggingFace's servers
- Response time depends on server load
- Consider using a smaller model or local deployment for faster responses

### Chat not streaming
- Ensure you're using the latest version of the code
- Check that all dependencies are properly installed
- Try refreshing the browser

## ⭐ Show your support

Give a ⭐️ if this project helped you!

---

Made with ❤️ and 🦙