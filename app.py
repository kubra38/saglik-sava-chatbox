import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langdetect import detect
from langdetect import LangDetectException
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import sys

# --- Loglama Ayarları ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("chat_logs.txt", mode='a'),
                        logging.StreamHandler()
                    ])

# --- Sabitler ve İlk Ayarlar ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# KULLANICIDAN ALINAN API ANAHTARI BURAYA YERLEŞTİRİLDİ
# NOTE: Gerçek API anahtarı burada gösterilmemiştir. Sadece örnek bir değer bırakılmıştır.
API_KEY = "AIzaSyAGAQhpIsFIOqpSKxJ6cO2bHb0UzaI2CIw"

if not API_KEY or API_KEY.strip() == "":
    logging.critical("HATA: GEMINI_API_KEY ortam değişkeni bulunamadı veya boş. Lütfen .env dosyasını kontrol edin.")
    sys.exit("Uygulama başlatılamadı: GEMINI_API_KEY ortam değişkeni eksik veya geçersiz.")

# load_data.py ile aynı sabitleri kullanın
CHROMA_DB_DIR = "chroma_db_multilang/"
COLLECTION_NAME = "sava_clinic_knowledge_multilang"
# KRİTİK DEĞİŞİKLİK: 'text-embedding-004' kullanılıyor.
EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"
FALLBACK_LANG = "en"

# Desteklenen diller
SUPPORTED_LANGS = ["en", "es", "sr", "fr", "tr"]

# RAG sistemi bileşenlerini global olarak tanımlayın
vectorstore = None
rag_chain = None


# =========================================================================
# 1. RAG SİSTEMİNİ BAŞLATMA VE YÜKLEME
# =========================================================================

def initialize_rag_system():
    """Vektör deposunu yükler ve RAG zincirini oluşturur."""
    global vectorstore, rag_chain

    if vectorstore is not None and rag_chain is not None:
        logging.info("RAG sistemi zaten yüklü.")
        return

    try:
        logging.info(f"RAG sistemi başlatılıyor... Gömme Modeli: {EMBEDDING_MODEL}")

        # 1. Gömme Fonksiyonunu Yükle
        embedding_function = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=API_KEY
        )

        # 2. Chroma Veritabanını Yükle (KRİTİK BÖLGE: Hata burada oluşur)
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME
        )
        # Hata oluşmazsa buraya ulaşılır
        logging.info("Chroma Veritabanı başarıyla yüklendi.")

        # 3. Model ve Prompt Tanımlamaları
        llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            temperature=0.0,
            google_api_key=API_KEY
        )

        # --- KRİTİK PROMPT GÜNCELLEMESİ ---
        # AI'ın yanıtına kaynak veya ek bilgi eklememesi için net talimat eklendi.
        template = """You are SAVA CLINIC's expert health assistant. Your goal is to answer user questions truthfully 
        based ONLY on the provided context. 

        If the context does not contain the answer, politely state that you do not have information on that specific topic 
        and suggest contacting the clinic via their website or WhatsApp. 

        IMPORTANT: Respond in the language requested by the user, which is determined by the language code: {lang_code}.
        CRITICAL: DO NOT include any citation, source, footnote, or "Sources:" section in your final answer text.

        Context:
        ---
        {context}
        ---

        Question: {question}

        Response (in {lang_code}):"""
        # --- PROMPT GÜNCELLEMESİ SONU ---

        prompt = PromptTemplate.from_template(template)

        # RAG zincirini tanımla
        rag_chain = (
                RunnablePassthrough.assign(context=(lambda x: x["context"]))
                | prompt
                | llm
                | StrOutputParser()
        )

        logging.info("RAG zinciri başarıyla oluşturuldu.")

    except Exception as e:
        # Hata durumunda, hem loglayın hem de terminale yazdırın
        logging.critical(f"RAG sistemi yüklenirken KRİTİK HATA oluştu: {e}")
        # Hata detayını terminalde gösterin
        print(f"\n\n🚨 KRİTİK HATA: RAG YÜKLEME BAŞARISIZ! 🚨\nDetay: {e}\n\n")
        vectorstore = None
        rag_chain = None


# =========================================================================
# DİĞER FONKSİYONLAR
# =========================================================================

def detect_and_filter(query: str) -> str:
    """Sorgunun dilini tespit eder ve desteklenmiyorsa varsayılan dile döner."""
    try:
        lang_code = detect(query)
        if lang_code in SUPPORTED_LANGS:
            logging.info(f"Dil tespit edildi: {lang_code}")
            return lang_code
        else:
            # Türkçe ('tr') dahil desteklenmeyen diller için fallback yapılır.
            logging.warning(
                f"Tespit edilen dil ({lang_code}) desteklenmiyor. Varsayılan dil ({FALLBACK_LANG}) kullanılıyor.")
            return FALLBACK_LANG
    except LangDetectException:
        logging.warning(f"Dil tespiti başarısız oldu. Varsayılan dil ({FALLBACK_LANG}) kullanılıyor.")
        return FALLBACK_LANG


def dynamically_retrieve_and_run(query: str, lang_code: str, vs: Chroma):
    """
    Filtrelenmiş alıcıyı kullanarak RAG zincirini çalıştırır.
    Alaka düzeyini artırmak için eşik ve k değeri ayarlandı.
    """
    global rag_chain

    # Benzerlik eşiği (score_threshold) 0.70'ten 0.65'e DÜŞÜRÜLDÜ.
    # Alınacak belge sayısı (k) 2'den 3'e ARTIRILDI.

    # 1. Gelişmiş Retriever oluştur
    retriever = vs.as_retriever(
        search_type="similarity_score_threshold",  # Belge kalitesini artırmak için
        search_kwargs={
            "score_threshold": 0.65,  # Potansiyel olarak faydalı belgeleri kaçırmamak için düşürüldü.
            "filter": {"lang": lang_code},
            "k": 3  # Modelin daha geniş bir bağlamda değerlendirme yapması için artırıldı.
        }
    )

    # 1. İlgili Bağlamı (Context) Çek
    try:
        retrieved_docs = retriever.invoke(query)

        # Eğer belge gelmezse (retrieved_docs boşsa), direkt olarak bilgi bulunamadı mesajını döndür.
        if not retrieved_docs:
            logging.warning(f"Benzerlik eşiği (0.65) nedeniyle '{query}' sorgusu için belge bulunamadı.")
            # Kaynak göstermeden kibarca reddetmek için boş bağlam ve kaynak döndürüyoruz.
            return "", []


    except Exception as e:
        logging.error(f"Retriever hatası: {e}.")
        # Teknik hata durumunda bir istisna fırlatın
        raise Exception("Retriever'ın invoke() metodu kullanılamıyor.") from e

    # 2. Bağlam Metnini ve Kaynakları Hazırla
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    # SADECE BENZERSİZ URL'LERİ AL ve sözlük formatında hazırla
    unique_urls = set()
    unique_sources = []

    for doc in retrieved_docs:
        source_url = doc.metadata.get("source")
        # Benzersizlik kontrolü burada yapılıyor
        if source_url and source_url not in unique_urls:
            unique_sources.append({"url": source_url})
            unique_urls.add(source_url)

            # 3. RAG Zincirini Çalıştır
    response = rag_chain.invoke({
        "question": query,
        "context": context_text,
        "lang_code": lang_code
    })

    return response, unique_sources


# =========================================================================
# FLASK ENDPOINTLERİ
# =========================================================================

@app.before_request
def check_rag_status():
    """Her istekten önce RAG sisteminin yüklü olup olmadığını kontrol eder."""
    # API Key kontrolü zaten başlangıçta yapıldığı için, sadece RAG'in yüklü olup olmadığına bakalım.
    if rag_chain is None or vectorstore is None:
        initialize_rag_system()
        # Hala yüklenmediyse
        if rag_chain is None or vectorstore is None:
            if request.path.startswith('/chat'):
                # API key/DB yükleme hatası varsa 503 döndür
                return jsonify({
                    "response": "Server Error: AI system is not initialized. Please check the server logs for API Key or ChromaDB errors.",
                    "sources": []}), 503


@app.route('/chat', methods=['POST'])
def chat():
    """Kullanıcı sorgusunu alır, dil tespitine göre filtrelenmiş RAG yapar ve yanıtı döndürür."""
    global vectorstore

    data = request.json
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"response": "Please enter a valid question.", "sources": []}), 400

    logging.info(f"--- YENİ SORGULAMA ---")
    logging.info(f"Kullanıcı Sorgusu: '{query}'")

    try:
        # 1. Dil Tespiti
        lang_code = detect_and_filter(query)

        # 2. Dinamik RAG İşlemini Gerçekleştir
        response, sources = dynamically_retrieve_and_run(query, lang_code, vectorstore)

        # 2.1. Eğer response boşsa, fallback mesajını manuel olarak oluştur.
        if not response:
            # Yanıtı LLM'den almak yerine manuel olarak oluşturuyoruz (kibarlık prompt'taki gibi)
            if lang_code == "es":
                response = "No tengo información sobre la definición específica de esa pregunta en el contexto proporcionado. Le sugiero que se ponga en contacto con la clínica a través de su sitio web o WhatsApp para obtener más detalles."
            elif lang_code == "tr":
                # Türkçe sorgu geldiği için Türkçe fallback mesajını netleştirdim.
                response = "Sağlanan bağlamda bu sorunun spesifik tanımı hakkında bilgim yok. Daha fazla ayrıntı için lütfen web sitemiz veya WhatsApp aracılığıyla klinik ile iletişime geçiniz."
            else:
                response = "I do not have information about the specific definition of that question in the provided context. I suggest you contact the clinic via their website or WhatsApp for more details."

        logging.info(f"AI Yanıtı: '{response}'")

        # Başarılı yanıtı döndür
        return jsonify({"response": response, "sources": sources})

    except Exception as e:
        logging.error(f"Sorgu işlenirken beklenmeyen kritik hata oluştu: {e}")
        # Hata durumunda kullanıcıya bilgilendirici mesaj döndür
        return jsonify({
            "response": f"I apologize, an internal error occurred while processing your request. Please try again later. Check the server log for details. Detailed Error: {str(e)}",
            "sources": []}), 500


# Basit bir endpoint ile log tutma
@app.route('/log_query', methods=['POST'])
def log_query():
    """Client tarafından gelen basit logları kaydeder."""
    data = request.json
    log_query = data.get('query', '')
    log_status = data.get('status', 'INFO')

    if log_status == 'ERROR':
        logging.error(f"Client Log: {log_query}")
    else:
        logging.info(f"Client Log: {log_query} - Status: {log_status}")

    return jsonify({"status": "logged"}), 200


# Uygulama arayüzünü sunan ana endpoint
@app.route('/')
def serve_index():
    """Ana HTML dosyasını sunar."""
    try:
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAVA CLINIC AI Assistant</title>
    <!-- Tailwind CSS'i yüklüyoruz -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Varsayılan font olarak Inter'ı kullanıyoruz */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f4f8; /* Açık arkaplan rengi */
        }
        /* Başlık arka plan rengini görsele göre ayarlıyoruz */
        .header-bg {
            background-color: #19365E; /* Koyu Mavi */
        }
        .chat-container {
            max-height: 55vh; /* Görseldeki boyuta yakın bir yükseklik */
            overflow-y: auto;
            scroll-behavior: smooth;
        }
        /* Mesaj kutularının genel stili */
        .message-box {
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            max-width: 85%;
            word-wrap: break-word; /* Uzun metinler için */
        }
        /* Kullanıcı mesajı stili */
        .user-message {
            background-color: #4A70AD; /* Mavi - Kullanıcı mesajı rengi */
            color: white;
            align-self: flex-end;
            margin-left: auto;
        }
        /* AI mesajı stili */
        .ai-message {
            background-color: #F8F9FA; /* Hafif Gri/Beyaz */
            color: #333;
            align-self: flex-start;
            border: 1px solid #E0E0E0;
            margin-right: auto;
            /* İlk karşılama mesajının özel stili */
            line-height: 1.5;
        }
        /* Yükleniyor animasyonu */
        .loading-dots div {
            animation: pulse 1.5s infinite ease-in-out;
        }
        .loading-dots div:nth-child(2) {
            animation-delay: 0.5s;
        }
        .loading-dots div:nth-child(3) {
            animation-delay: 1s;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.5; }
        }
        /* Kaynak link stili */
        .source-link {
            color: #2563eb;
            text-decoration: underline;
            font-size: 0.75rem; /* text-xs */
        }
    </style>
</head>
<body class="flex items-center justify-center min-h-screen p-4">

    <!-- Ana Konteyner -->
    <div class="w-full max-w-xl bg-white rounded-xl shadow-2xl flex flex-col h-[70vh] overflow-hidden">

        <!-- Başlık Bölümü (Koyu Mavi) -->
        <header class="p-4 header-bg flex items-center justify-center rounded-t-xl">
            <h1 class="text-xl font-bold text-white">SAVA CLINIC AI Assistant</h1>
        </header>

        <!-- Sohbet Alanı -->
        <div id="chat-messages" class="flex-grow p-5 chat-container flex flex-col">
            <!-- İlk Karşılama Mesajı (Görseldeki gibi) -->
            <div class="message-box ai-message">
                <p class="whitespace-pre-wrap">Hello! I am SAVA CLINIC's expert health assistant. Please ask a question about our health services.</p>
            </div>
            <!-- Dinamik Mesajlar buraya eklenecek -->
        </div>

        <!-- Giriş Bölümü -->
        <div class="p-5 border-t border-gray-200 bg-white flex items-center rounded-b-xl">
            <input type="text" id="user-input" placeholder="Ask your question here..." class="flex-grow p-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-150 ease-in-out shadow-sm" autocomplete="off">
            <button id="send-button" class="ml-3 px-6 py-3 header-bg text-white font-semibold rounded-xl hover:bg-opacity-90 transition duration-150 ease-in-out shadow-md disabled:opacity-70">
                Send
            </button>
        </div>
    </div>

    <!-- JavaScript Kodu -->
    <script>
        const chatMessages = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');

        // Sunucunun 5001 portunda çalıştığını varsayarak mutlak URL kullanıyoruz
        const API_URL = 'http://127.0.0.1:5001/chat'; 

        // Mesaj kutusu oluşturma
        function createMessageBox(text, type, sources = []) {
            const box = document.createElement('div');
            box.classList.add('message-box', type === 'user' ? 'user-message' : 'ai-message');

            // Basit metin formatlama
            let formattedText = text.replace(/\\n/g, '<br>'); // Yeni satırları düzelt
            formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'); // **Koyu Yazı**

            box.innerHTML = formattedText;

            // Kaynakları mesaj kutusunun altında, ayrı bir bölümde göster
            if (type === 'ai' && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.classList.add('mt-2', 'pt-2', 'border-t', 'border-gray-200', 'text-xs', 'text-gray-500');
                sourcesDiv.innerHTML = '<strong>Sources:</strong>';

                sources.forEach(source => {
                    // Kaynak nesnesi {url: "..."} formatındadır.
                    const sourceUrl = source.url; 
                    const link = document.createElement('a');
                    link.href = sourceUrl;
                    link.textContent = sourceUrl; // Tam URL göster
                    link.target = '_blank';
                    link.classList.add('source-link', 'block', 'truncate');
                    sourcesDiv.appendChild(link);
                });
                box.appendChild(sourcesDiv);
            }

            chatMessages.appendChild(box);
            // En alta kaydır
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Yükleniyor animasyonu oluşturma
        function createLoadingIndicator() {
            const loadingBox = document.createElement('div');
            loadingBox.id = 'loading-indicator';
            loadingBox.classList.add('message-box', 'ai-message', 'flex', 'items-center', 'space-x-1', 'loading-dots');
            loadingBox.innerHTML = `
                <div class="w-2 h-2 bg-blue-500 rounded-full"></div>
                <div class="w-2 h-2 bg-blue-500 rounded-full"></div>
                <div class="w-2 h-2 bg-blue-500 rounded-full"></div>
            `;
            chatMessages.appendChild(loadingBox);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Yükleniyor animasyonunu kaldırma
        function removeLoadingIndicator() {
            const indicator = document.getElementById('loading-indicator');
            if (indicator) {
                indicator.remove();
            }
        }

        // API'ye sorgu gönderme
        async function sendQuery() {
            const query = userInput.value.trim();
            if (!query) return;

            // 1. Kullanıcı mesajını ekle
            createMessageBox(query, 'user');

            // 2. Girişi temizle ve butonu devre dışı bırak
            userInput.value = '';
            sendButton.disabled = true;

            // 3. Yükleniyor animasyonunu göster
            createLoadingIndicator();

            try {
                // Flask sunucusuna POST isteği - 5001 portunu kullanıyoruz
                const response = await fetch(API_URL, { 
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query })
                });

                removeLoadingIndicator();
                sendButton.disabled = false;

                // HTTP 200/201 kontrolü
                if (!response.ok) {
                    const errorText = await response.text();
                    let errorData;
                    try {
                        errorData = JSON.parse(errorText);
                    } catch (e) {
                        errorData = { response: `Server returned status ${response.status}. Could not parse error details.` };
                    }

                    createMessageBox(`Error (HTTP ${response.status}): ${errorData.response || 'Could not reach server or server provided a non-JSON error.'}`, 'ai');
                    console.error('API Error:', errorData);
                    return;
                }

                const data = await response.json();

                // 4. AI yanıtını ve kaynakları ekle
                const sources = data.sources || [];

                // NOT: Yanıtın içinde "Sources:" metni artık sunucu tarafında eklenmiyor.
                // Sadece temiz metin alınıyor. Kaynaklar "sources" array'i içinde geliyor.
                createMessageBox(data.response, 'ai', sources);

            } catch (error) {
                removeLoadingIndicator();
                sendButton.disabled = false;
                console.error('Request Error:', error);
                // Eğer sunucuya hiç ulaşılamadıysa (CORS, network hatası vb.)
                createMessageBox(`Connection Error: Could not reach the server at ${API_URL}. (Is Flask running on 5001?)`, 'ai');
            }
        }

        // Olay dinleyicileri
        sendButton.addEventListener('click', sendQuery);
        userInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault(); // Varsayılan form gönderme davranışını engelle
                sendQuery();
            }
        });

    </script>

</body>
</html>
        """
        return html_content, 200, {'Content-Type': 'text/html'}

    except Exception as e:
        logging.error(f"HTML arayüzü sunulurken hata: {e}")
        return "Internal Server Error", 500


if __name__ == '__main__':
    # RAG sistemini başlatmayı deneyelim
    initialize_rag_system()
    # Port 5001 kullanılıyor.
    app.run(host='0.0.0.0', port=5001, debug=False)