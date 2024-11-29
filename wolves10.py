import os
import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pdfplumber
import pytesseract
from PIL import Image
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import re
import pandas as pd
import time

# Configurações Globais
TIPOS_ARQUIVOS_VALIDOS = ['Pdf']

PARAMS_MODELO = {'max_tokens': 2048}

CHUNK_CONFIG = {
    'tamanho': 2000,
    'sobreposicao': 200,
    'separadores': ["\n\n", "\n", ". ", " ", ""]
}

ID_PATTERNS = {
    'processo': r'\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}',
    'protocolo': r'protocolo\s*[nº]?\s*\d+[-./]\d+',
    'documento': r'(?i)doc(?:umento)?\s*[nº]?\s*\d+[-./]\d+',
    'oficio': r'(?i)of[íi]cio\s*[nº]?\s*\d+[-./]\d+',
    'registro': r'(?i)registro\s*[nº]?\s*\d+[-./]\d+',
    'decisao': r'(?i)decisão\s*[nº]?\s*\d+[-./]\d+',
    'despacho': r'(?i)despacho\s*[nº]?\s*\d+[-./]\d+',
    'sentenca': r'(?i)sentença\s*[nº]?\s*\d+[-./]\d+',
    'id_geral': r'ID\s*[-:]?\s*[A-Z0-9]+[-./]?[A-Z0-9]*'
}

def progress_bar(placeholder, fase, progresso):
    """Atualiza a barra de progresso visual."""
    barras_total = 20
    barras_completas = int(progresso * barras_total)
    barra = f"[{'▓' * barras_completas}{'░' * (barras_total - barras_completas)}] {int(progresso * 100)}%"
    placeholder.markdown(f"{barra}\n{fase}")

@dataclass
class ChunkMetadata:
    chunk_id: str
    page_number: int
    position: int
    document_ids: List[str]
    content_type: str
    references: List[str]
    timestamp: str
    original_text: str

class DocumentProcessor:
    def __init__(self):
        self.chunks_metadata = {}
        self.found_ids = set()

    def extract_document_ids(self, text: str) -> List[str]:
        found_ids = []
        for id_type, pattern in ID_PATTERNS.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                found_ids.append(f"{id_type}:{match.group()}")
        return found_ids

    def process_text(self, text: str, progress_placeholder) -> Dict:
        try:
            pages = text.split("=== Página")
            total_pages = len(pages) - 1
            all_processed_chunks = {}
            total_chunks = 0
            progress_bar(progress_placeholder, "Extração", 0.2)
            for page_num, page_content in enumerate(pages[1:], 1):
                progress = 0.2 + (page_num / total_pages * 0.3)
                progress_bar(progress_placeholder, "Extração", progress)
                paragraphs = [p for p in page_content.split('\n\n') if p.strip()]
                for para_num, paragraph in enumerate(paragraphs):
                    if len(paragraph.strip()) < 50:
                        continue
                    document_ids = self.extract_document_ids(paragraph)
                    self.found_ids.update(document_ids)
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_CONFIG['tamanho'],
                        chunk_overlap=CHUNK_CONFIG['sobreposicao'],
                        separators=CHUNK_CONFIG['separadores']
                    )
                    chunks = splitter.split_text(paragraph)
                    for chunk_num, chunk in enumerate(chunks):
                        total_chunks += 1
                        chunk_id = f"P{page_num}_Para{para_num + 1}_C{chunk_num + 1}"
                        metadata = ChunkMetadata(
                            chunk_id=chunk_id,
                            page_number=page_num,
                            position=total_chunks,
                            document_ids=document_ids,
                            content_type="corpo",
                            references=self.extract_document_ids(chunk),
                            timestamp=datetime.now().isoformat(),
                            original_text=chunk
                        )
                        all_processed_chunks[chunk_id] = asdict(metadata)
            progress_bar(progress_placeholder, "Análise", 0.7)
            time.sleep(0.5)
            progress_bar(progress_placeholder, "Organização", 1.0)
            time.sleep(0.3)
            return all_processed_chunks
        except Exception as e:
            st.error(f"Erro durante o processamento: {str(e)}")
            return {}

class EnhancedOracle:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.memoria = ConversationBufferMemory()
        self.suggested_prompts = []

    def process_document(self, arquivo, progress_placeholder) -> str:
        try:
            text = self.extract_from_pdf(arquivo, progress_placeholder)
            if not text:
                return None
            processed_chunks = self.processor.process_text(text, progress_placeholder)
            return processed_chunks
        except Exception as e:
            st.error(f"Erro durante o processamento: {str(e)}")
            return None

    def extract_from_pdf(self, arquivo, progress_placeholder) -> Optional[str]:
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
                temp.write(arquivo.getbuffer())
                temp_path = temp.name
            text = ""
            with pdfplumber.open(temp_path) as pdf:
                total_pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages, 1):
                    progress = i / total_pages * 0.2
                    progress_bar(progress_placeholder, "Extração", progress)
                    content = page.extract_text()
                    if not content or len(content.strip()) < 100:
                        image = page.to_image()
                        content = pytesseract.image_to_string(image.original, lang='por')
                    if content:
                        text += f"\n=== Página {i} ===\n{content}"
            os.unlink(temp_path)
            return text
        except Exception as e:
            st.error(f"Erro ao extrair texto do PDF: {str(e)}")
            return None

def create_enhanced_prompt(documento: str) -> str:
    prompt_template = """Você é um assistente especializado chamado Artur que DEVE seguir estas regras RIGOROSAMENTE:

    ####
    {documento}
    ####
    """
    return prompt_template.format(documento=documento)

def main():
    st.set_page_config(page_title="Artur - Analisador de Processo", layout="wide")
    if 'oracle' not in st.session_state:
        st.session_state['oracle'] = EnhancedOracle()
        st.session_state['previous_file'] = None
    with st.sidebar:
        st.header("⚙️ Configurações")
        arquivo = st.file_uploader("Upload do arquivo PDF", type=['pdf'])
        if st.button("Processar Documento", use_container_width=True):
            if arquivo:
                progress_placeholder = st.empty()
                documento_processado = st.session_state['oracle'].process_document(arquivo, progress_placeholder)
                if documento_processado:
                    prompt = create_enhanced_prompt(str(documento_processado))
                    chat = ChatOpenAI(
                        model="gpt-4o-mini",
                        api_key=os.getenv("OPENAI_API_KEY"),
                        max_tokens=PARAMS_MODELO['max_tokens']
                    )
                    template = ChatPromptTemplate.from_messages([
                        ("system", prompt),
                        ("human", "{query}")
                    ])
                    st.session_state['chain'] = template | chat
                    st.success("✅ Documento processado com sucesso!")
                progress_placeholder.empty()
            else:
                st.error("Por favor, faça o upload de um PDF.")

if __name__ == "__main__":
    main()
