# Python Generative AI & LangChain Internship 🚀

Welcome to my **Python Generative AI & LangChain Internship Portfolio!**

This repository showcases four hands-on projects developed during my Generative AI internship. Each project focuses on applying Python, Large Language Models (LLMs), LangChain, prompt engineering, structured outputs, Retrieval-Augmented Generation (RAG), and AI agents to solve practical problems.

## 👩‍💻 About Me

I am a Computer Science and Mathematics graduate interested in Artificial Intelligence, Machine Learning, and Generative AI.

Through hands-on projects, I am exploring how to build practical AI-powered applications using Python, LLMs, LangChain, and modern AI development tools.

## 🛠️ Technologies Used

- Python 3.12
- LangChain
- Large Language Models (LLMs)
- OpenRouter API
- Pydantic
- Prompt Engineering
- Structured JSON Outputs
- Vector Embeddings
- Retrieval-Augmented Generation (RAG)
- Streamlit
- Gradio
- AI Agents
- uv (Python Package & Environment Management)
- Git & GitHub

## 📚 Internship Projects

### Project 1: Hirelyzer – AI-Powered Resume Parser

**Objective:** Build an AI-powered resume parsing system that extracts structured information from resumes in multiple formats and validates the output against a defined schema.

**Key Features:**
- Resume processing from PDF, DOCX, TXT, and image formats
- LLM-powered information extraction
- Structured JSON generation
- Pydantic schema validation
- Retry and output repair mechanisms
- Evaluation of extraction results

**Project Files:**
- [Project Documentation](01_Hirelyzer/README.md)

### Project 2: DocQA – Document Question Answering with RAG

**Objective:** Develop a Retrieval-Augmented Generation (RAG) application that answers user questions using information retrieved from uploaded documents.

**Key Features:**
- Document ingestion and processing
- Text chunking and preprocessing
- Vector embeddings
- Semantic retrieval
- Context-aware answer generation
- Source page citations
- Refusal when answers are not supported by the documents

**Project Files:**
- [Project Documentation](02_DocQA_RAG/README.md)

### Project 3: O3 – CartSense: Autonomous AI Shopping Agent

**Objective:** Build an intelligent tool-calling AI agent that helps users make informed product purchasing decisions by gathering requirements, searching the web, and calculating the actual cost of ownership.

**Key Features:**
- LangChain tool-calling agent
- Interactive user requirement gathering
- Web search integration
- Product research
- Cost-of-ownership calculations
- Multi-tool execution
- CLI-based interaction

**Project Files:**
- [Project Documentation](03_CartSense/README.md)

### Project 4: DocQA – Interactive Streamlit Application

**Objective:** Extend the DocQA RAG system into an interactive Streamlit application that allows users to communicate with their documents through a chat interface.

**Key Features:**
- Interactive Streamlit interface
- Document-based question answering
- Streaming responses
- Conversation-based interaction
- Integration with the Task 2 RAG pipeline

**Project Files:**
- [Project Documentation](04_DocQA_Streamlit/README.md)

## 📂 Repository Structure

```text
Datheon_Python_-_Generative_AI_-_LangChain_Internship/
│
├── 01_Hirelyzer/
│   ├── README.md
│   └── ...
│
├── 02_DocQA_RAG/
│   ├── README.md
│   └── ...
│
├── 03_CartSense/
│   ├── README.md
│   └── ...
│
├── 04_DocQA_Streamlit/
│   ├── README.md
│   └── ...
│
├── pyproject.toml
├── uv.lock
└── README.md
```

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/sumithaManoharan/Datheon_Python_-_Generative_AI_-_LangChain_Internship.git
```

**2. Navigate to the project directory**

```bash
cd Datheon_Python_-_Generative_AI_-_LangChain_Internship
```

**3. Install dependencies**

This repository uses `uv` for Python dependency and environment management.

```bash
uv sync
```

**4. Configure environment variables**

Each project has its own `.env.example` file.

Copy the relevant example file into `.env` inside the project directory and configure the required API keys.

Refer to the individual project README files for detailed setup and execution instructions.

## 🔐 API Key Configuration

Some projects require API keys to access external LLM providers and services.

- Configure the required API keys in the respective project's `.env` file.
- Never commit `.env` files or expose API keys in your repository.
- Refer to each project's documentation for its specific configuration requirements.

## 🎯 Learning Outcomes

Through these projects, I gained practical experience in:

- Building Python applications powered by Large Language Models.
- Designing prompts and generating structured LLM outputs.
- Extracting and validating information using Pydantic.
- Implementing Retrieval-Augmented Generation pipelines.
- Working with document processing, embeddings, and semantic search.
- Building autonomous AI agents with tool-calling capabilities.
- Integrating LLMs with external tools and APIs.
- Developing interactive AI applications using Streamlit and Gradio.
- Managing Python dependencies and project environments using uv.

## 👩‍💻 Author

**Sumitha Manoharan**

GitHub: [@sumithaManoharan](https://github.com/sumithaManoharan)

---

⭐ Thank you for visiting my Python Generative AI & LangChain Internship Portfolio!

