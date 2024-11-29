import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pdfplumber
import pytesseract
from PIL import Image
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set
import hashlib
import re
import pandas as pd
import time

# Configurações Globais
TIPOS_ARQUIVOS_VALIDOS = ['Pdf']

# Configurações de Modelo (apenas GPT-4o-mini)
CONFIG_MODELOS = {
    'OpenAI': {
        'modelos': ['gpt-4o-mini'],  # Apenas um modelo disponível
        'chat': ChatOpenAI
    }
}

# Parâmetros específicos do modelo
PARAMS_MODELOS = {
    'gpt-4o-mini': {'max_tokens': 2048},  # Apenas configuração para o modelo usado
}


# Padrões para identificação de IDs no documento
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

CHUNK_CONFIG = {
    'tamanho': 2000,
    'sobreposicao': 200,
    'separadores': ["\n\n", "\n", ". ", " ", ""]
}

def progress_bar(placeholder, fase, progresso):
    """Atualiza a barra de progresso visual."""
    barras_total = 20
    barras_completas = int(progresso * barras_total)
    
    barra = f"[{'▓' * barras_completas}{'░' * (barras_total - barras_completas)}] {int(progresso * 100)}%"
    
    placeholder.markdown(f"""
    {barra}
    {fase}
    """)
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
        self.document_index = {}
        self.found_ids = set()
        
    def extract_document_ids(self, text: str) -> List[str]:
        found_ids = []
        for id_type, pattern in ID_PATTERNS.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                found_ids.append(f"{id_type}:{match.group()}")
        return found_ids

    def process_text(self, text: str, doc_type: str, progress_placeholder) -> Dict:
        try:
            pages = text.split("=== Página")
            total_pages = len(pages) - 1
            
            all_processed_chunks = {}
            total_chunks = 0
            
            # Fase de extração
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
                        separators=CHUNK_CONFIG['separadores']  # Corrigido de 'separadores'
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
                            content_type=self.identify_content_type(chunk),
                            references=self.extract_document_ids(chunk),
                            timestamp=datetime.now().isoformat(),
                            original_text=chunk
                        )
                        
                        all_processed_chunks[chunk_id] = asdict(metadata)
            
            # Fase de análise
            progress_bar(progress_placeholder, "Análise", 0.7)
            time.sleep(0.5)
            
            # Fase de organização
            progress_bar(progress_placeholder, "Organização", 1.0)
            time.sleep(0.3)
            
            return all_processed_chunks
            
        except Exception as e:
            st.error(f"Erro durante o processamento: {str(e)}")
            return {}

    def identify_content_type(self, text: str) -> str:
        text_lower = text.lower()
        if "conclusão" in text_lower or "conclusao" in text_lower:
            return "conclusão"
        elif "sentença" in text_lower:
            return "sentença"
        elif "decisão" in text_lower or "despacho" in text_lower:
            return "decisão"
        elif "audiência" in text_lower:
            return "audiência"
        elif "manifestação" in text_lower:
            return "manifestação"
        elif "petição" in text_lower:
            return "petição"
        elif "certidão" in text_lower:
            return "certidão"
        elif any(header in text_lower for header in ["título", "capítulo", "seção"]):
            return "cabeçalho"
        return "corpo"

class EnhancedOracle:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.memoria = ConversationBufferMemory()
        self.suggested_prompts = []
        self.grau_jurisdicao = None

    def _generate_first_degree_prompts(self) -> List[str]:  # Movido para dentro da classe
        prompts = [
            "📋 Resumo dos principais atos do processo",
            "⚖️ Análise das últimas decisões",
            "📅 Status dos prazos e intimações",
            "👥 Situação das audiências marcadas",
            "📄 Manifestações pendentes",
            "🔍 Diligências em andamento",
            "⚡ Pedidos urgentes não apreciados",
            "📊 Análise de provas produzidas",
            "⚖️ Status das medidas cautelares",
            "📝 Resumo das últimas petições",
            "🔎 Perícias e laudos técnicos",
            "👨‍⚖️ Despachos pendentes de cumprimento",
            "📅 Próximos atos processuais",
            "⚖️ Questões preliminares pendentes",
            "📊 Resumo das testemunhas ouvidas"
        ]
        return prompts

    def _generate_second_degree_prompts(self) -> List[str]:  # Movido para dentro da classe
        prompts = [
            "📋 Resumo do recurso",
            "⚖️ Análise de admissibilidade",
            "📊 Principais teses recursais",
            "🔍 Jurisprudência citada",
            "📝 Status do julgamento",
            "👥 Manifestações das partes",
            "⚡ Pedidos de efeito suspensivo",
            "📄 Análise das contrarrazões",
            "⚖️ Parecer do MP",
            "📊 Precedentes aplicáveis",
            "🔎 Questões preliminares",
            "📑 Documentos novos",
            "⚖️ Análise da decisão recorrida",
            "🏛️ Jurisprudência do tribunal",
            "📌 Pedidos de preferência"
        ]
        return prompts

    def _organize_prompts_by_group(self, prompts: List[str]) -> List[str]:
        grupos = {
            "Prazos e Urgências": [],
            "Análise Processual": [],
            "Específicos do Caso": []
        }
        
        for prompt in prompts:
            if any(word in prompt.lower() for word in ["prazo", "urgente", "pendente"]):
                grupos["Prazos e Urgências"].append(prompt)
            elif any(word in prompt.lower() for word in ["análise", "resumo", "status"]):
                grupos["Análise Processual"].append(prompt)
            else:
                grupos["Específicos do Caso"].append(prompt)
        
        organized_prompts = []
        for grupo, items in grupos.items():
            if items:
                organized_prompts.extend(items)
        
        return organized_prompts
    def generate_suggested_prompts(self, processed_text: str) -> List[str]:
        if self.grau_jurisdicao == "1º Grau":
            prompts = self._generate_first_degree_prompts()
        else:
            prompts = self._generate_second_degree_prompts()
        
        prompts = prompts[:15]  # Limita a 15 prompts
        return self._organize_prompts_by_group(prompts)

    def process_document(self, arquivo, progress_placeholder) -> str:
        try:
            texto = self.extract_from_pdf(arquivo, progress_placeholder)
            if not texto:
                return None
            
            processed_chunks = self.processor.process_text(texto, 'Pdf', progress_placeholder)
            
            # Organiza o resultado
            structured_text = []
            content_types = {
                "decisão": [], "sentença": [], "audiência": [],
                "manifestação": [], "petição": [], "outros": []
            }
            
            for chunk_id, metadata in processed_chunks.items():
                content_type = metadata['content_type']
                if content_type in content_types:
                    content_types[content_type].append(metadata)
                else:
                    content_types["outros"].append(metadata)

            for content_type, chunks in content_types.items():
                if chunks:
                    structured_text.append(f"\n=== {content_type.upper()} ===")
                    for metadata in sorted(chunks, key=lambda x: x['page_number']):
                        chunk_header = f"\n--- Página {metadata['page_number']} ---"
                        if metadata['document_ids']:
                            chunk_header += f"\nReferências: {', '.join(metadata['document_ids'])}"
                        chunk_header += "\nConteúdo:"
                        structured_text.append(f"{chunk_header}\n{metadata['original_text']}")
            
            final_text = "\n\n".join(structured_text)
            self.suggested_prompts = self.generate_suggested_prompts(final_text)
            
            return final_text
            
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
                    progress = i / total_pages * 0.2  # Primeiros 20%
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

    1. Você tem acesso ao seguinte documento dividido em partes identificadas:
    
    ####
    {documento}
    ####

    2. REGRAS FUNDAMENTAIS:
    - Todas as informações fornecidas DEVEM vir exclusivamente do documento analisado
    - NUNCA faça suposições ou crie informações não presentes no documento
    - Sempre cite as referências (IDs) ao fornecer informações
    
    3. AO ANALISAR O DOCUMENTO:
    - Identifique e cite os IDs mencionados no conteúdo
    - Use os IDs como referência principal
    - Se não houver ID específico, use a página como referência
    
    4. AO RESPONDER:
    - Comece indicando a fonte da informação (ID ou página)
    - Use tabelas quando apropriado
    - Indique claramente quando uma informação não for encontrada
    
    5. Se uma informação não estiver no documento, responda "Esta informação não consta no documento analisado"."""

    return prompt_template.format(documento=documento)
def main():
    st.set_page_config(
        page_title="Artur - Analisador de Processo",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Configuração do estilo
    st.markdown("""
        <style>
        .main-header {text-align: center; padding: 1rem;}
        .progress-bar {padding: 0.5rem; margin: 1rem 0;}
        .stButton > button {font-size: 0.85em; padding: 0.3rem;}
        .sidebar-content {padding: 1rem 0;}
        div[data-testid="stSidebarNav"] {
            background-image: url("https://cdn.midjourney.com/5062038c-895e-4e32-8976-a0e85f062538/0_0.png");
            background-repeat: no-repeat;
            padding-top: 120px;
            background-position: 20px 20px;
            background-size: 200px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if 'oracle' not in st.session_state:
        st.session_state['oracle'] = EnhancedOracle()
        st.session_state['previous_file'] = None
    
    with st.sidebar:
        st.image("https://cdn.midjourney.com/5062038c-895e-4e32-8976-a0e85f062538/0_0.png", 
                 width=250)
        st.markdown("##### Desenvolvido por João Valério")
        
        st.header("⚙️ Configurações")
        
        # Instruções de uso
        with st.expander("📖 Como usar"):
            st.markdown("""
            1. Selecione o grau de jurisdição
            2. Faça upload do arquivo PDF
            3. Configure o modelo de análise
            4. Clique em 'Processar Documento'
            5. Use as sugestões ou faça suas perguntas
            """)
        
        # Seleção do grau de jurisdição
        grau_jurisdicao = st.radio(
            "Selecione o grau de jurisdição:",
            ["1º Grau", "2º Grau"],
            key="grau_jurisdicao"
        )
        
        arquivo = st.file_uploader(
            "Upload do arquivo PDF",
            type=['pdf']
        )
        
        # Verifica se um novo arquivo foi carregado
        if arquivo is not None and st.session_state['previous_file'] != arquivo.name:
            st.session_state['previous_file'] = arquivo.name
            st.session_state['oracle'] = EnhancedOracle()
            st.session_state['oracle'].grau_jurisdicao = grau_jurisdicao
            if 'chain' in st.session_state:
                del st.session_state['chain']
        
        modelo_config = st.selectbox(
            "Provedor",
            CONFIG_MODELOS.keys()
        )
        
        modelo = st.selectbox(
            "Modelo",
            CONFIG_MODELOS[modelo_config]['modelos']
        )
        
        api_key = st.text_input(
            f"API Key ({modelo_config})",
            type="password"
        )
        
        if st.button("Processar Documento", use_container_width=True):
            if arquivo:
                progress_placeholder = st.empty()
                with st.spinner():
                    documento_processado = st.session_state['oracle'].process_document(
                        arquivo, progress_placeholder
                    )
                    
                    if documento_processado:
                        prompt = create_enhanced_prompt(documento_processado)
                        max_tokens = PARAMS_MODELOS.get(modelo, {}).get('max_tokens', 2048)
                        chat = CONFIG_MODELOS[modelo_config]['chat'](
                            model=modelo,
                            api_key=api_key,
                            max_tokens=max_tokens
                        )
                        
                        template = ChatPromptTemplate.from_messages([
                            ("system", prompt),
                            ("human", "{query}")
                        ])
                        
                        chain = template | chat
                        st.session_state['chain'] = chain
                        st.success("✅ Documento processado com sucesso!")
                progress_placeholder.empty()
            else:
                st.error("Por favor, faça o upload de um PDF.")
    
    # Área principal
    st.title("Artur - Analisador de Processo")
    st.markdown("### Sistema Avançado de Análise de Documentos Jurídicos")
    
    if 'chain' in st.session_state:
        st.header("💬 Análise do Processo")
        
        # Estilo CSS para os prompts
        st.markdown("""
            <style>
            div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
                padding: 0.2rem;
            }
            .stButton > button {
                font-size: 0.75em;
                padding: 0.3rem;
                min-height: 0px;
                margin: 0.1rem 0;
                line-height: 1.2;
                width: 100%;
                height: auto;
                white-space: normal;
                background-color: #f0f2f6;
                border: 1px solid #e0e3e9;
            }
            .stButton > button:hover {
                background-color: #e0e3e9;
                border-color: #c0c3c9;
            }
            .grupo-prompt {margin: 1rem 0;}
            </style>
        """, unsafe_allow_html=True)
        
        if st.session_state['oracle'].suggested_prompts:
            st.markdown("##### 🤔 Sugestões de análise", 
                       help="Clique em qualquer sugestão para analisar")
            
            num_cols = 3
            cols = st.columns(num_cols)
            for idx, prompt in enumerate(st.session_state['oracle'].suggested_prompts):
                with cols[idx % num_cols]:
                    if st.button(prompt, key=f"prompt_{idx}", use_container_width=True):
                        st.session_state['current_query'] = prompt
        
        chat_container = st.container()
        user_input = st.chat_input("Faça sua pergunta sobre o processo...")
        
        if 'current_query' in st.session_state:
            user_input = st.session_state['current_query']
            del st.session_state['current_query']
        
        with chat_container:
            for msg in st.session_state['oracle'].memoria.buffer_as_messages:
                with st.chat_message(msg.type):
                    st.write(msg.content)
        
        if user_input:
            with st.chat_message("human"):
                st.write(user_input)
            
            with st.chat_message("ai"):
                full_response = ""
                response_container = st.empty()
                
                try:
                    for chunk in st.session_state['chain'].stream({
                        "query": user_input
                    }):
                        if hasattr(chunk, 'content'):
                            chunk_content = chunk.content
                        else:
                            chunk_content = str(chunk)
                            
                        full_response += chunk_content
                        response_container.markdown(full_response)
                    
                    st.session_state['oracle'].memoria.chat_memory.add_user_message(user_input)
                    st.session_state['oracle'].memoria.chat_memory.add_ai_message(full_response)
                
                except Exception as e:
                    st.error(f"Erro ao processar a resposta: {str(e)}")
    else:
        st.info("👈 Comece fazendo upload de um PDF na barra lateral.")

if __name__ == "__main__":
    main()
