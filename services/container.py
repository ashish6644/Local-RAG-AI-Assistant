"""Builds the object graph once per process."""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import Settings, ensure_directories
from core.embeddings import OllamaEmbedder
from core.ollama_client import OllamaClient
from core.rag_chain import RAGPipeline
from core.retriever import Retriever
from core.vector_store import VectorStore
from database.connection import Database
from database.repository import Repository
from services.chat_service import ChatService
from services.document_service import DocumentService
from services.ingestion_service import IngestionService
from services.model_service import ModelService


@dataclass
class Services:
    settings: Settings
    repo: Repository
    client: OllamaClient
    models: ModelService
    chat: ChatService
    documents: DocumentService
    ingestion: IngestionService
    store: VectorStore


def build_services(settings: Settings) -> Services:
    ensure_directories(settings)
    db = Database(settings.db_path)
    db.init_db()
    repo = Repository(db)
    client = OllamaClient(settings.ollama_base_url)
    embedder = OllamaEmbedder(client)
    store = VectorStore(settings.chroma_dir, settings.collection_name)
    pipeline = RAGPipeline(client, Retriever(store, embedder))
    ingestion = IngestionService(settings, repo, embedder, store)
    return Services(
        settings=settings, repo=repo, client=client, models=ModelService(client),
        chat=ChatService(repo, pipeline), documents=DocumentService(settings, repo, store, ingestion),
        ingestion=ingestion, store=store,
    )
