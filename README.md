# ResearchMentor AI

An AI-powered research mentoring application that guides law and social science students through the research process step by step.

## About

**ResearchMentor AI** is designed to act as a research mentor rather than simply generating research content for students.

The application encourages students to develop their own ideas and provides AI-guided feedback throughout three stages of the research process:

- **Module 1 — Research Discovery:** Develop and refine a research topic and question.
- **Module 2 — Literature Review:** Analyse research papers, identify patterns, and develop a research gap.
- **Module 3 — Research Design:** Select and justify an appropriate research methodology.

The AI follows a mentoring approach:

> **Block irrelevant responses. Guide weak responses. Encourage strong responses.**

## Key Features

- 🤖 AI-powered research supervision
- 🧠 Encourages independent student thinking
- 📚 Literature review guidance
- 🔍 Research gap identification
- ⚖️ Designed for law and social science research
- 📝 Research methodology guidance
- 📄 PDF research paper support
- 💾 Multiple research project management
- 🔐 API key support through Streamlit secrets or user input

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- Pandas
- PyPDF

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ResearchMentor-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Add your Google API key through Streamlit secrets:

```toml
GOOGLE_API_KEY = "your_api_key_here"
```

Alternatively, the application supports API key input within the app.

### 4. Run the application

```bash
streamlit run app.py
```

## Project Structure

```text
ResearchMentor-AI/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Project Goal

Traditional AI tools often generate research content directly for students.

**ResearchMentor AI takes a different approach.**

Instead of replacing the student's thinking, it aims to guide students through:

**Idea → Question → Literature → Research Gap → Methodology**

## Hackathon Vision

> **Don't let AI do the student's research. Let AI help the student learn how to research.**
