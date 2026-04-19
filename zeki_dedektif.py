import streamlit as st
import base64
from duckduckgo_search import DDGS
from groq import Groq
from PIL import Image
import io

# --- AYARLAR ---
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def encode_image(image_file):
    image_file.seek(0)
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
                        results = list(ddgs.text(search_q, max_results=8))
                        for r in results:
                            web_context += f"Kaynak: {r['title']} - Özet: {r['body']}\n"
                    
                    prompt = f"""
Sen bir profesyonel gerçek-yanlış haber dedektörüsün.

İDDİA: {user_input}

WEB KAYNAKLARI:
{web_context}

GÖREV:
1. İddiayı web kaynaklarıyla karşılaştır
2. Kaynaklar güvenilir mi? (haber sitesi mi, blog mu?)
3. İddiayı destekleyen ve çürüten kaynakları ayır
4. Net bir karar ver

KURALLAR:
- Gereksiz nezaket cümleleri kurma
- Eğer web'de yeterli bilgi yoksa 'Şüpheli' de
- Kaynakları belirt

SONUÇ FORMATI:
- Kaynaklar: (hangi sitelerde geçiyor)
- Destekleyen Kanıtlar:
- Çürüten Kanıtlar:
- KARAR: GERÇEK / YALAN / ŞÜPHELİ
- Güven Skoru: %0-100
"""
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=TEXT_MODEL,
                    )
                    st.info(safe_str(chat_completion.choices[0].message.content))
                except Exception as e:
                    st.error(f"Hata: {str(e)}")

# --- TAB 2: GÖRSEL ANALİZ ---
with tab2:
    img_file = st.file_uploader("Görsel yükleyin", type=["jpg", "jpeg", "png"])
    
    if img_file:
        img = Image.open(img_file)
        st.image(img, width=500)

        if st.button("Görsel Analizi Başlat"):
            with st.spinner('Görsel inceleniyor...'):
                try:
                    # Görseli base64'e çevir
                    img_file.seek(0)
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    # Dosya uzantısına göre media type belirle
                    ext = img_file.name.split('.')[-1].lower()
                    media_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

                    analiz_prompt = """Sen bir dijital adli bilişim uzmanısın. Bu görseli dikkatle inceleyerek yapay zeka tarafından üretilip üretilmediğini tespit et.
Görselde ne gördüğünü kısaca söyle. 
Şu kriterlere göre analiz et:
- Ellerde parmak sayısı ve şekli
- Yüz simetrisi ve anatomik doğruluk  
- Arka plan tutarlılığı ve mantığı
- Işık/gölge yönü tutarlılığı
- Deri dokusu (aşırı pürüzsüz mü?)
- Yazı varsa harflerin doğruluğu
- Gözler ve yansımalar
- Fiziksel olarak imkansız objeler
- Arka planda mantıksız detaylar
- Obje ile arka planın fiziksel uyumsuzluğu

SONUÇ FORMATI:
- Dikkat Çeken İpuçları: (gözlemlerin)
- Güven Skoru: %0-100
- KARAR: YAPAY ZEKA veya GERÇEK FOTOĞRAF
"""
                    
                    chat_completion = client.chat.completions.create(
                        model=VISION_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{media_type};base64,{image_data}"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": analiz_prompt
                                    }
                                ]
                            }
                        ],
                        max_tokens=1000,
                    )

                    st.success("🔍 Görsel Analiz Sonucu:")
                    st.write(chat_completion.choices[0].message.content)

                except Exception as e:
                    st.error(f"Analiz başarısız: {str(e)}")
