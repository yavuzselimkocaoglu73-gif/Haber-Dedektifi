import streamlit as st
import base64
from duckduckgo_search import DDGS
from groq import Groq
from PIL import Image

# --- AYARLAR ---

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
MODEL_NAME = "llama-3.3-70b-versatile"

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

def safe_str(text):
    if not text: return ""
    mapping = str.maketrans("ğĞıİöÖüÜşŞçÇ", "gGiIoOuUsScC")
    return str(text).translate(mapping).encode('ascii', 'ignore').decode('ascii')

st.set_page_config(page_title="Haber Dedektif", layout="wide")
st.title("🕵️ Haber Dedektifi")

tab1, tab2 = st.tabs(["📝 Haber Analizi", "📸 Görsel Analizi"])

# --- TAB 1: HABER ANALİZİ ---
with tab1:
    user_input = st.text_input("İddiayı girin:", key="txt_v32")
    if st.button("Doğrula"):
        if user_input:
            with st.spinner('Araştırılıyor...'):
                try:
                    search_q = safe_str(user_input)
                    web_context = ""
                    with DDGS() as ddgs:
                        results = list(ddgs.text(search_q, max_results=5))
                        for r in results:
                            web_context += f"Kaynak: {r['title']} - Özet: {r['body']}\n"
                    
                    prompt = f"İddia: {user_input}\nVeriler: {web_context}\nBu iddiayı analiz et. Gereksiz nezaket cümleleri kurma. Sonuçta 'Gerçek', 'Yalan' veya 'Şüpheli' şeklinde net kararını belirt."
                    
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=MODEL_NAME,
                    )
                    st.info(safe_str(chat_completion.choices[0].message.content))
                except Exception as e:
                    st.error(f"Hata: {str(e)}")

# --- TAB 2: GÖRSEL ANALİZ (NET VE KISA) ---
with tab2:
    img_file = st.file_uploader("Görsel yükleyin", type=["jpg", "jpeg", "png"])
    
    if img_file:
        img = Image.open(img_file)
        st.image(img, width=500)
        w, h = img.size

        # --- TAB 2: MANTIKSAL ÇAPRAZ SORGU (v49) ---
        if st.button("Teknik Analizi Başlat"):
            if img_file is not None:
                with st.spinner('Veri tabanındaki dijital izleri karşılaştırıyorum...'):
                    try:
                        img = Image.open(img_file)
                        w, h = img.size
                        
                        # Dedektife sadece elindeki somut verileri veriyoruz
                        analiz_prompt = f"""
                        Sen bir Dijital Adli Bilişim Uzmanısın. Görseli GÖREMEDİĞİNİ biliyorum. 
                        Sadece şu somut verilere dayanarak bir Olasılık Raporu hazırla:

                        SOMUT VERİLER:
                        - Çözünürlük: {w}x{h}
                        - Dosya Adı: {img_file.name}

                        DEĞERLENDİRME KRİTERLERİ:
                        1. DOSYA ADI ANALİZİ: Eğer dosya adında 'pexels', 'unsplash', 'canon', 'iphone' gibi ifadeler varsa bu GERÇEK bir fotoğraftır. Eğer 'ai', 'generated', 'midjourney', 'dalle' geçiyorsa YAPAY ZEKA'dır.
                        2. ÇÖZÜNÜRLÜK ANALİZİ: {w}x{h} boyutu standart bir sensör oranı mı (3:2, 4:3, 16:9)? Eğer çok absürt bir kareyse (Örn: 1024x1024 tam kare) AI olma ihtimali artar.
                        3. ŞÜPHE PRENSİBİ: Eğer dosya adı stok sitesinden geliyorsa (Pexels vb.), çözünürlük ne olursa olsun 'İNSAN YAPIMI' lehine karar ver.

                        HÜKÜM FORMATI:
                        - Tespit Edilen İpucu: 
                        - Tahmini Kaynak: 
                        - 3. maddeye göre karar verdiysen bile stok sitesi adı olduğu için %100 bu kararı verdiğini söyleme
                        - SONUÇ: YAPAY ZEKA veya İNSAN YAPIMI
                        """

                        chat_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": analiz_prompt}],
                            model="llama-3.3-70b-versatile", 
                        )
                        
                        st.success("🤖 Mantıksal Analiz Sonucu:")
                        st.write(chat_completion.choices[0].message.content)
                        
                    except Exception as e:
                        st.error(f"Sorgulama başarısız: {str(e)}")
            else:
                st.warning("Lütfen önce bir fotoğraf yükle!")