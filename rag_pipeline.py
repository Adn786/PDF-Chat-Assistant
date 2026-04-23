"""
RAG Pipeline Module with MMR (Maximal Marginal Relevance)
Handles: PDF loading, chunking, embeddings, vector storage, retrieval, and answer generation
"""

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import chromadb
import tempfile
import os


class RAGPipeline:
    def __init__(self, groq_api_key):
        """Initialize RAG Pipeline with Groq API key"""
        self.groq_api_key = groq_api_key

        # Initialize embeddings (local, no API needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Initialize ChromaDB (in-memory)
        self.chroma_client = chromadb.Client()
        self.collection = None

        # Store embeddings in memory to avoid re-computing on every query
        self.all_chunks_data = []

        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-120b",
            temperature=0.3
        )

    def load_and_chunk_pdf(self, pdf_path, chunk_size=1000, chunk_overlap=150):
        """Load PDF and split into chunks"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            return chunks
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")

    def create_embeddings_and_store(self, chunks, status_callback=None):
        """Create embeddings and store in ChromaDB + in-memory cache"""
        try:
            collection_name = "pdf_documents"

            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(name=collection_name)
            except:
                pass

            self.collection = self.chroma_client.create_collection(name=collection_name)

            # Clear in-memory cache
            self.all_chunks_data = []

            total = len(chunks)

            for i, chunk in enumerate(chunks):
                if status_callback:
                    status_callback(f"Embedding chunk {i+1} of {total}...")

                chunk_embedding = self.embeddings.embed_documents([chunk.page_content])[0]
                chunk_embedding_np = np.array(chunk_embedding)

                self.collection.add(
                    ids=[f"chunk_{i}"],
                    documents=[chunk.page_content],
                    embeddings=[chunk_embedding],
                    metadatas=[{"chunk_id": i}]
                )

                # Cache in memory
                self.all_chunks_data.append({
                    'id': f"chunk_{i}",
                    'doc': chunk.page_content,
                    'embedding': chunk_embedding_np
                })

            if status_callback:
                status_callback(f"✅ All {total} chunks embedded and stored in ChromaDB!")

            return total

        except Exception as e:
            raise Exception(f"Error creating embeddings: {str(e)}")

    def retrieve_with_mmr(self, query, k=10, mmr_lambda=0.5):
        """
        Retrieve top-k chunks using MMR (Maximal Marginal Relevance)
        Uses cached embeddings — no re-embedding of chunks.
        """
        try:
            if not self.all_chunks_data:
                raise Exception("No PDF loaded. Please upload a PDF first.")

            # Only embed the query (chunks are already cached)
            query_embedding = np.array(self.embeddings.embed_documents([query])[0])

            # Calculate query similarity for all cached chunks
            chunks_with_scores = []
            for chunk in self.all_chunks_data:
                similarity = np.dot(chunk['embedding'], query_embedding) / (
                    np.linalg.norm(chunk['embedding']) * np.linalg.norm(query_embedding) + 1e-10
                )
                chunks_with_scores.append({**chunk, 'similarity': similarity})

            # MMR Selection
            retrieved_chunks = []
            selected_ids = set()

            for _ in range(min(k, len(chunks_with_scores))):
                best_mmr_score = -float('inf')
                best_chunk = None

                for chunk in chunks_with_scores:
                    if chunk['id'] in selected_ids:
                        continue

                    relevance = chunk['similarity']

                    if len(retrieved_chunks) == 0:
                        diversity = 1.0
                    else:
                        similarities_to_selected = [
                            np.dot(chunk['embedding'], sel['embedding']) / (
                                np.linalg.norm(chunk['embedding']) *
                                np.linalg.norm(sel['embedding']) + 1e-10
                            )
                            for sel in retrieved_chunks
                        ]
                        diversity = 1 - min(similarities_to_selected)

                    mmr_score = (1 - mmr_lambda) * relevance + mmr_lambda * diversity

                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_chunk = chunk

                if best_chunk:
                    retrieved_chunks.append(best_chunk)
                    selected_ids.add(best_chunk['id'])

            return [chunk['doc'] for chunk in retrieved_chunks]

        except Exception as e:
            raise Exception(f"Error in MMR retrieval: {str(e)}")

    def generate_answer(self, query, retrieved_chunks):
        """Generate answer using Groq LLM with retrieved chunks as context"""
        try:
            context = "\n\n---CHUNK---\n\n".join(retrieved_chunks)

            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are an expert assistant. Answer the following question ONLY using the provided document context.

INSTRUCTIONS:
1. Read the context carefully
2. Answer directly and thoroughly
3. Use exact information from the context
4. If you find relevant information, include it
5. ONLY say "I don't have enough information" if the context contains NO relevant content

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
            )

            chain = prompt_template | self.llm
            response = chain.invoke({
                "context": context,
                "question": query
            })

            return response.content

        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")

    def process_pdf(self, pdf_file, status_callback=None):
        """
        Complete pipeline: Load PDF -> Chunk -> Embed -> Store
        Returns number of chunks created
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name

            if status_callback:
                status_callback("📄 Loading and chunking PDF...")

            chunks = self.load_and_chunk_pdf(tmp_path)

            if status_callback:
                status_callback(f"✅ PDF chunked into {len(chunks)} chunks. Now embedding...")

            num_chunks = self.create_embeddings_and_store(chunks, status_callback)

            os.unlink(tmp_path)

            return num_chunks

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def chat(self, user_query):
        """
        Main chat function: Retrieve relevant chunks and generate answer
        """
        try:
            retrieved_chunks = self.retrieve_with_mmr(user_query, k=10)
            answer = self.generate_answer(user_query, retrieved_chunks)
            return answer

        except Exception as e:
            raise Exception(f"Error in chat: {str(e)}")