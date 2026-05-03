# ✍️ AI Technical Blog Writer

An intelligent **AI-powered technical blog generator** built with LangGraph, Streamlit and Groq LLMs.

---

## 🚀 Features

* 🔀 **Smart Routing**

  * Decides whether research is needed before writing

* 🔍 **Automated Research**

  * Uses Tavily API for real-time web search

* 📋 **Structured Planning**

  * Generates a detailed blog outline with sections and goals

* ✏️ **Parallel Content Generation**

  * Each section is written independently using LangGraph workers

* 🔗 **Final Compilation**

  * Combines all sections into a clean Markdown blog

* 📥 **Download Option**

  * Export blog as a `.md` file

---

## 🧠 Tech Stack

* **Frontend:** Streamlit
* **LLM:** Groq (LLaMA models)
* **Orchestration:** LangGraph
* **Search:** Tavily API
* **Validation:** Pydantic

---

## 📂 Project Structure

```
app.py   # Main Streamlit application
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-TECHNICAL-BLOG-WRITING-AGENT.git
cd AI-TECHNICAL-BLOG-WRITING-AGENT
```

### 2. Install dependencies

```bash
pip install streamlit langgraph langchain-groq langchain-community \
            tavily-python pydantic python-dotenv
```

---

## 🔑 API Keys Required

You need:

* Groq API Key → https://console.groq.com
* Tavily API Key → https://tavily.com

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🖥️ Usage

1. Enter your own API keys in the sidebar
2. Input a blog topic
3. Click **"Generate Blog"**
4. View, copy, or download the generated blog

---

## 🧩 How It Works

### 1. Router

Determines whether the topic requires external research

### 2. Research (Optional)

Fetches relevant web results using Tavily

### 3. Orchestrator

Creates a structured blog plan with multiple sections

### 4. Workers

Generate each section in parallel using LLMs

### 5. Reducer

Combines all sections into a final blog

---

## ⚠️ Limitations

* Dependent on API rate limits (Groq free tier)
* Quality depends on prompt + model performance
* Research accuracy depends on Tavily results

---

## 💡 Future Improvements

* Add image generation 
* Add blog editing interface
* Improve citation formatting
* Add memory/history for past blogs

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open issues or suggest improvements.

---

## ⭐ Support

If you like this project, consider giving it a star ⭐ on GitHub!
