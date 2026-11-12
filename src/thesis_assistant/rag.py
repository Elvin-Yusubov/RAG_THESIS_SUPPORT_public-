"""RAG backend shared by the Streamlit app and the Telegram bot.

A single LangChain agent owns the conversation. Retrieval is exposed as a
*tool*, so the model decides per request whether the answer depends on an
institutional rule (search the index) or not (draft, rewrite, write a review).

Public API:
    run_llm      blocking call returning the complete answer and its sources
    stream_llm   generator yielding text chunks, then a final result dict

Heavy resources (embeddings, vector store, chat model, agent) are created on
first use and cached, so importing this module has no network side effects.
"""

from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pydantic import SecretStr

from thesis_assistant import config
from thesis_assistant.retrieval import rerank_hybrid
from thesis_assistant.vision import describe_docx_images, format_image_snippets

try:  # private, git-ignored prompts take precedence
    from thesis_assistant.prompts import SYSTEM_PROMPT
except ImportError:
    from thesis_assistant.prompts_default import SYSTEM_PROMPT

ChatHistory = List[Dict[str, str]]
StreamItem = str | Dict[str, Any]


# --- Cached resources --------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    """Return the Pinecone store, using the same embedding setup as ingestion."""
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=SecretStr(config.OPENROUTER_API_KEY) if config.OPENROUTER_API_KEY else None,
        base_url=config.OPENROUTER_BASE_URL,
    )
    return PineconeVectorStore(index_name=config.INDEX_NAME, embedding=embeddings)


@lru_cache(maxsize=1)
def get_agent():
    """Build the tool-using agent once and reuse it for every request."""
    model = init_chat_model(
        config.CHAT_MODEL,
        model_provider=config.CHAT_MODEL_PROVIDER,
        api_key=config.DEEPSEEK_API_KEY,
    )
    return create_agent(model, tools=[retrieve_context], system_prompt=SYSTEM_PROMPT)


# --- Retrieval tool ------------------------------------------------------------------

def format_retrieved_documents(documents: List[Document]) -> str:
    """Serialize chunks for the model, including source, category and page.

    Category and page let the model cite precisely; rules often differ between
    document types (for example bachelor vs master templates).
    """
    return "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')} "
            f"(category: {doc.metadata.get('category', 'Unknown')}, "
            f"page: {doc.metadata.get('page_number', '?')})\n\n"
            f"Content: {doc.page_content[: config.MAX_CHARS_PER_CHUNK]}"
        )
        for doc in documents
    )


def search_documents(query: str, mode: Optional[str] = None, top_k: Optional[int] = None) -> List[Document]:
    """Search the index. Shared by the agent tool and the evaluation script.

    Args:
        query: Natural-language query.
        mode: "dense" or "hybrid"; defaults to ``config.RETRIEVAL_MODE``.
        top_k: Number of chunks to return; defaults to ``config.RETRIEVAL_TOP_K``.
    """
    mode = mode or config.RETRIEVAL_MODE
    top_k = top_k or config.RETRIEVAL_TOP_K
    # Hybrid over-fetches candidates, then lexical fusion picks the best top_k.
    fetch_k = max(config.RETRIEVAL_FETCH_K, top_k) if mode == "hybrid" else top_k
    candidates = get_vectorstore().as_retriever(search_kwargs={"k": fetch_k}).invoke(query)
    if mode == "hybrid":
        return rerank_hybrid(query, candidates, top_k)
    return candidates


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant thesis-support content (guidelines, methodology docs, templates, submission rules) to help answer user queries."""
    documents = search_documents(query)
    # The second value is the tool "artifact": the raw Documents, which the UIs
    # use to list sources without parsing the model's text.
    return format_retrieved_documents(documents), documents


# --- Conversation assembly ---------------------------------------------------------------

def trim_history(chat_history: Optional[ChatHistory]) -> ChatHistory:
    """Keep only the most recent MAX_HISTORY_TURNS user/assistant exchanges."""
    if not chat_history:
        return []
    return chat_history[-config.MAX_HISTORY_TURNS * 2 :]


def build_user_message(query: str, document_text: Optional[str] = None) -> str:
    """Compose the user message, prepending an uploaded document if present.

    The document is truncated to MAX_DOCUMENT_CHARS and the model is told when
    that happened. It applies to this call only and is never stored.
    """
    if not document_text:
        return query

    truncated = document_text[: config.MAX_DOCUMENT_CHARS]
    note = (
        "\n\n[Note: the document below was truncated to fit context limits.]"
        if len(document_text) > config.MAX_DOCUMENT_CHARS
        else ""
    )
    return (
        f"[Tələbənin yüklədiyi sənədin mətni:]\n\n{truncated}{note}\n\n"
        f"---\n\n[Tələbənin sualı/tapşırığı:]\n{query}"
    )


def _build_messages(
    query: str, chat_history: Optional[ChatHistory], document_text: Optional[str]
) -> ChatHistory:
    return trim_history(chat_history) + [
        {"role": "user", "content": build_user_message(query, document_text)}
    ]


# --- Public API ------------------------------------------------------------------------------

def run_llm(
    query: str,
    chat_history: Optional[ChatHistory] = None,
    document_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer a query and return the complete result (blocking).

    Args:
        query: The user's question or task.
        chat_history: Prior ``{"role", "content"}`` messages, used to resolve
            follow-ups such as "what about the introduction?".
        document_text: Text of a student-uploaded .docx, used for this call only.

    Returns:
        ``{"answer": str, "context": List[Document]}`` where ``context`` holds
        every chunk the retrieval tool returned.
    """
    messages = _build_messages(query, chat_history, document_text)
    response = get_agent().invoke({"messages": messages})

    context_docs: List[Document] = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and isinstance(message.artifact, list):
            context_docs.extend(message.artifact)

    return {"answer": response["messages"][-1].content, "context": context_docs}


def stream_llm(
    query: str,
    chat_history: Optional[ChatHistory] = None,
    document_text: Optional[str] = None,
) -> Iterator[StreamItem]:
    """Streaming variant of :func:`run_llm`.

    Yields text chunks as the model produces them, then a final
    ``{"answer": str, "context": List[Document]}`` dict as the last item.

    Because retrieval is a tool call, there is a pause before text starts:
    the model first decides to search and receives the results, then the
    answer itself streams token by token.
    """
    messages = _build_messages(query, chat_history, document_text)

    full_answer = ""
    context_docs: List[Document] = []

    # stream_mode="messages" also emits tool-call and tool-result messages;
    # only AI text is forwarded, and tool artifacts are collected as context.
    for chunk, _metadata in get_agent().stream({"messages": messages}, stream_mode="messages"):
        if isinstance(chunk, ToolMessage):
            if isinstance(chunk.artifact, list):
                context_docs.extend(chunk.artifact)
            continue

        text = getattr(chunk, "content", None)
        if text:
            full_answer += text
            yield text

    yield {"answer": full_answer, "context": context_docs}


# --- Uploaded documents ----------------------------------------------------------------------

def extract_docx_text(file_path: str) -> str:
    """Extract the text of a student's uploaded .docx as conversation context.

    This is a one-off read for the current request; the content is never
    added to the shared Pinecone index. Embedded images (diagrams, charts,
    screenshots) are described by the vision fallback and appended to the
    text when that fallback is enabled.
    """
    documents = Docx2txtLoader(file_path).load()
    text = "\n\n".join(doc.page_content for doc in documents)

    snippets = describe_docx_images(file_path)
    if snippets:
        text = (text.strip() + "\n\n" + format_image_snippets(snippets)).strip()
    return text
