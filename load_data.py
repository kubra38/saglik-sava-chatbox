import os
import requests
import logging
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

# --- Log Ayarları ---
# Loglama seviyesini DEBUG'a ayarlayalım ki tüm adımları görelim.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Sabitler ---
load_dotenv()
# KULLANICIDAN ALINAN API ANAHTARI BURAYA YERLEŞTİRİLDİ
API_KEY = "AIzaSyAGAQhpIsFIOqpSKxJ6cO2bHb0UzaI2CIw"
if not API_KEY:
    logging.error("HATA: GEMINI_API_KEY ortam değişkeni bulunamadı. Lütfen .env dosyasını kontrol edin.")

# Çok dilli URL'ler ve dillerin eşleşmesi (CRITICAL)
LANG_URLS = {
    "en": ["https://savaclinic.com/treatments/obesity-surgery/",
           "https://savaclinic.com/bariatric-surgery/gastric-sleeve/",
           "https://savaclinic.com/bariatric-surgery/gastric-bypass/",
           "https://savaclinic.com/treatments/obesity-surgery/gastric-bypass-revision/",
           "https://savaclinic.com/bariatric-surgery/gastric-balloon/",
           "https://savaclinic.com/treatments/plastic-surgery/",
           "https://savaclinic.com/treatments/plastic-surgery/rhinoplasty/",
           "https://savaclinic.com/treatments/plastic-surgery/face-lift/",
           "https://savaclinic.com/treatments/plastic-surgery/bichectomy/",
           "https://savaclinic.com/treatments/plastic-surgery/breast-augmentation/",
           "https://savaclinic.com/treatments/plastic-surgery/breast-reconstruction/",
           "https://savaclinic.com/treatments/plastic-surgery/breast-reduction/",
           "https://savaclinic.com/treatments/plastic-surgery/liposuction/",
           "https://savaclinic.com/treatments/plastic-surgery/tummy-tuck-abdominoplasty/",
           "https://savaclinic.com/treatments/plastic-surgery/arm-and-thigh-lift/",
           "https://savaclinic.com/treatments/plastic-surgery/mommy-makeover/"],
    "es": ["https://savaclinic.com/es/tratos/cirugia-de-la-obesidad/",
           "https://savaclinic.com/es/tratos/cirugia-de-la-obesidad/manga-gastrica/",
           "https://savaclinic.com/es/tratos/cirugia-de-la-obesidad/bypass-gastrico/",
           "https://savaclinic.com/es/tratos/cirugia-de-la-obesidad/revision-del-bypass-gastrico/",
           "https://savaclinic.com/es/tratos/cirugia-de-la-obesidad/balon-gastrico/",
           "https://savaclinic.com/es/tratos/cirugia-plastica/",
           "https://savaclinic.com/es/tratos/cirugia-plastica/rinoplastia/",
           "https://savaclinic.com/es/tratos/cirugia-plastica/estiramiento-facial/",
           "https://savaclinic.com/es/tratos/cirugia-plastica/bichectomia/",
           "https://savaclinic.com/es/tratos/cirugia-plastica/aumento-de-senos/",
           "https://savaclinic.com/es/tratos/cirugia-plastica/reconstruccion-mamaria/",
           "https://savaclinic.com/es/tratos/cirugia-plastica/reduccion-de-mama/",
           "https://savaclinic.com/es/tratos/cirugia-plastica/liposuccion/",
           "https://savaclinic.com/es/tratos/cirugia-plastica/tummy-tuck-abdominoplastia/",
           "https://savaclinic.com/es/tratos/cirugia-plastica/lifting-de-brazos-y-muslos/",
           "https://savaclinic.com/es/tratos/cirugia-plastica/mommy-makeover/"],
    "sr": ["https://savaclinic.com/sr/tretmani/operacija-gojaznosti/",
           "https://savaclinic.com/sr/tretmani/operacija-gojaznosti/gastricki-bajpas/",
           "https://savaclinic.com/sr/tretmani/operacija-gojaznosti/sleeve-gastrektomija/",
           "https://savaclinic.com/sr/tretmani/operacija-gojaznosti/gastricni-balon/",
           "https://savaclinic.com/sr/tretmani/operacija-gojaznosti/revizija-gastricnog-bajpasa/",
           "https://savaclinic.com/sr/tretmani/plasticna-operacija/",
           "https://savaclinic.com/sr/tretmani/plasticna-operacija/rinoplastika/",
           "https://savaclinic.com/sr/tretmani/plasticna-operacija/podmladjivanje-lica/",
           "https://savaclinic.com/sr/tretmani/plasticna-operacija/bihektomija/",
           "https://savaclinic.com/sr/tretmani/plasticna-operacija/povecanje-grudi/",
           "https://savaclinic.com/sr/tretmani/plasticna-operacija/rekonstrukcija-dojke/",
           "https://savaclinic.com/sr/tretmani/plasticna-operacija/smanjenje-grudi/",
           "https://savaclinic.com/sr/tretmani/plasticna-operacija/liposukcija/",
           "https://savaclinic.com/sr/tretmani/plasticna-operacija/zatezanje-stomaka-abdominoplastika/",
           "https://savaclinic.com/sr/tretmani/plasticna-operacija/lifting-ruku-i-butina/",
           "https://savaclinic.com/sr/tretmani/plasticna-operacija/mama-makeover/"],
    "fr": ["https://savaclinic.com/fr/traitements/chirurgie-de-lobesite/",
           "https://savaclinic.com/fr/chirurgie-de-lobesite/bypass-gastrique/",
           "https://savaclinic.com/fr/chirurgie-de-lobesite/revision-du-bypass-gastrique/",
           "https://savaclinic.com/fr/traitements/chirurgie-de-lobesite/sleeve-gastrique/",
           "https://savaclinic.com/fr/traitements/chirurgie-de-lobesite/ballon-gastrique/",
           "https://savaclinic.com/fr/traitements/chirurgie-plastique/",
           "https://savaclinic.com/fr/traitements/chirurgie-plastique/rhinoplastie/",
           "https://savaclinic.com/fr/traitements/chirurgie-plastique/lifting-facial/",
           "https://savaclinic.com/fr/traitements/chirurgie-plastique/bichectomie/",
           "https://savaclinic.com/fr/traitements/chirurgie-plastique/augmentation-mammaire/",
           "https://savaclinic.com/fr/traitements/chirurgie-plastique/reconstruction-mammaire/",
           "https://savaclinic.com/fr/traitements/chirurgie-plastique/reduction-mammaire/",
           "https://savaclinic.com/fr/traitements/chirurgie-plastique/liposuccion/",
           "https://savaclinic.com/fr/traitements/chirurgie-plastique/abdominoplastie-redrapage-abdominal/",
           "https://savaclinic.com/fr/traitements/chirurgie-plastique/lifting-des-bras-et-des-cuisses/",
           "https://savaclinic.com/fr/traitements/chirurgie-plastique/mommy-makeover/"]
}

CHROMA_DB_DIR = "chroma_db_multilang/"  # Farklı bir klasör kullanalım
COLLECTION_NAME = "sava_clinic_knowledge_multilang"
# KRİTİK GÜNCELLEME: app.py'deki 'text-embedding-004' ile eşleşmelidir.
EMBEDDING_MODEL = "text-embedding-004"


# =========================================================================
# 1. VERİ ÇEKME VE TEMİZLEME
# =========================================================================

def fetch_and_clean_data(url: str) -> str:
    """Belirtilen URL'den HTML çeker ve temiz metin döndürür."""
    logging.info(f"Veri çekiliyor: {url}")
    try:
        # User-Agent ekleyelim, bazı siteler botları engeller.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Sadece ana içeriği veya makale içeriğini hedefleyin
        # Bu, menü, footer gibi gürültüyü azaltır.
        main_content = soup.find('main') or soup.find('article') or soup.find(id='content')

        if main_content:
            # Metin çıkarma ve temizleme
            text = main_content.get_text()
            # Birden fazla boşluğu ve yeni satırı tek boşluğa indirgeyin
            text = ' '.join(text.split())

            # Gürültü filtrelemesi: Kısa metinleri (örneğin sadece menü isimleri) ele
            if len(text) < 100:
                logging.warning(f"Uyarı: {url} adresinden çekilen metin çok kısa ({len(text)} karakter), atlanıyor.")
                return ""

            return text
        else:
            logging.warning(f"Uyarı: {url} adresinde ana içerik bulunamadı.")
            return ""

    except requests.exceptions.RequestException as e:
        logging.error(f"Hata: {url} adresine erişilemedi: {e}")
        return ""
    except Exception as e:
        logging.error(f"Hata: {url} verileri işlenirken hata oluştu: {e}")
        return ""


# =========================================================================
# 2. METİNİ PARÇALAMA VE DİL METADATA EKLEME
# =========================================================================

def chunk_data(text: str, url: str, lang_code: str) -> list[Document]:
    """Metni Langchain Document objelerine böler ve dil kodunu metadata olarak ekler."""

    # Daha kararlı gömme için 512/100 ayarı
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        length_function=len
    )

    if not text:
        return []

    chunks = text_splitter.split_text(text)

    # Her dokümana 'source' (URL) ve 'lang' (dil kodu) metadata'sı ekliyoruz
    documents = [
        Document(page_content=chunk, metadata={"source": url, "lang": lang_code})
        for chunk in chunks
    ]
    logging.info(f"[{lang_code.upper()}] {url} için {len(documents)} adet doküman parçası oluşturuldu.")
    return documents


# =========================================================================
# 3. VERİTABANI OLUŞTURMA
# =========================================================================

def create_chroma_db(documents: list[Document]):
    """Dokümanları gömer ve Chroma veritabanını diske kaydeder."""
    if not API_KEY:
        logging.error("Veritabanı oluşturulamadı: API Anahtarı eksik.")
        return

    if not documents:
        logging.warning("Veritabanına kaydedilecek doküman bulunamadı.")
        return

    try:
        # 1. Gömme Fonksiyonunu Tanımla
        embedding_function = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=API_KEY
        )

        # 2. Chroma Veritabanını Oluştur ve Kaydet
        # Persistence'ı etkinleştirmek için "persist_directory" kullanıyoruz
        Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=CHROMA_DB_DIR,
            collection_name=COLLECTION_NAME
        )

        logging.info(f"Vektör veritabanı başarıyla oluşturuldu ve diske kaydedildi: {CHROMA_DB_DIR}")

    except Exception as e:
        logging.error(f"ChromaDB oluşturulurken kritik hata: {e}")


# =========================================================================
# ANA ÇALIŞMA FONKSİYONU
# =========================================================================

if __name__ == "__main__":
    logging.info("--- SAVA Clinic ÇOK DİLLİ RAG Veritabanı Oluşturma Başladı ---")

    all_documents = []

    # Her dildeki tüm URL'leri döngüye sok
    for lang_code, urls in LANG_URLS.items():
        logging.info(f"--- DİL: {lang_code.upper()} Verileri İşleniyor ---")
        for url in urls:
            # 1. Veriyi Çek ve Temizle
            raw_text = fetch_and_clean_data(url)

            if raw_text:
                # 2. Metni Parçalara Böl ve Dil Kodunu Ekle
                documents = chunk_data(raw_text, url, lang_code)
                all_documents.extend(documents)

    if all_documents:
        # 3. Veritabanını Oluştur
        create_chroma_db(all_documents)
    else:
        logging.error("KRİTİK HATA: Hiçbir URL'den geçerli içerik çekilemedi. Veritabanı oluşturulmadı.")

    logging.info("--- SAVA Clinic ÇOK DİLLİ RAG Veritabanı Oluşturma Tamamlandı ---")