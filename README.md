"LexResearch AI — Evidence-Grounded Legal Research & Case Preparation Copilot"

PRODUCT VISION

Build an interactive AI-powered legal research workspace that helps lawyers and law students move from an unstructured legal problem to a structured, evidence-grounded case research report.

The application must NOT behave like a generic legal chatbot.

Its primary purpose is to:

- structure legal problems
- analyze uploaded legal documents
- retrieve relevant passages
- identify legal issues
- compare authorities
- detect contradictions
- build arguments and counterarguments
- verify citations
- identify unsupported claims
- generate a structured research report

The AI must act as a research assistant and analytical advisor, not as a replacement for a qualified lawyer.

CORE PRINCIPLE

Every important AI-generated legal claim must be grounded in retrieved source material whenever source material is available.

The application must never invent:

- cases
- statutes
- legal provisions
- quotations
- citations
- court decisions
- facts

If evidence is unavailable, explicitly say:

"Insufficient evidence in the available sources."

TECHNOLOGY

Build the application using:

- Python
- Streamlit
- OpenAI Agents SDK for agent orchestration
- MCP for modular tool integration
- RAG architecture
- PDF/document parsing
- embeddings
- vector database suitable for a local hackathon deployment
- an LLM provider with API-key configuration through the application settings
- modular architecture
- clean error handling
- environment-variable support as an optional deployment method

Do not hardcode API keys.

The application should allow the user to enter/configure the required API key through a secure settings interface where practical.

APPLICATION WORKFLOW

Create exactly six major workflow modules.

MODULE 1 — CASE INTAKE & ISSUE FORMULATION

User provides:

- case title
- jurisdiction
- practice area
- client position
- case facts
- known legal issues
- available documents

Create a Case Intake Agent.

The agent should:

- summarize user-provided facts
- separate facts from assumptions
- identify primary legal issues
- identify secondary legal issues
- generate research questions
- identify missing information
- identify potentially relevant legal domains

Display:

Primary Issues
Secondary Issues
Missing Information
Research Questions

Do not present AI assumptions as established facts.

MODULE 2 — LEGAL RESEARCH & DOCUMENT ANALYSIS

Allow users to upload:

- PDF judgments
- statutes
- regulations
- contracts
- legal articles
- research documents
- case briefs
