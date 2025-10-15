import re
import time
import random
import threading
import joblib
import speech_recognition as sr
import pyttsx3
import pandas as pd

# ====== Load model & vectorizer ======
vectorizer = joblib.load("models/language/vectorizer.pkl")
model = joblib.load("models/language/lgbm_model.pkl")
with open("models/language/threshold.txt", "r") as f:
    chosen_threshold = float(f.read().strip())

# ====== Load responses for output ======
file_path = "data/csv/Data_tanggapan_positif.xlsx"
df_krisis = pd.read_excel(file_path, sheet_name="Krisis")
df_tidak = pd.read_excel(file_path, sheet_name="Tidak Krisis")

def preprocess_text(s):
    s = str(s).lower()
    s = re.sub(r'[^0-9a-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

CRISIS_KEYWORDS = [preprocess_text(k) for k in [
    # ✨ Frasa langsung
    "bunuh diri", "saya mau mati", "saya mati", "bunuh", "ingin mati", "ingin bunuh diri",
    "tidak ingin hidup", "sudah tidak kuat", "sudah tidak sanggup",
    "mati saja", "selesai saja", "akhiri hidup", "putus asa",
    "menyakiti diri", "mengakhiri hidup", "sudah tidak ada harapan",
    "sudah ingin mati", "capek hidup",

    # 💔 Variasi penulisan & ejaan
    "gw mau mati", "gue mau mati", "pengen mati", "pgn mati", "pingin mati",
    "udah ga kuat", "gak kuat lagi", "gk kuat", "ga kuat",
    "cape hidup", "capee hidup", "udah cape", "sudah capek",
    "gak sanggup lagi", "udah nyerah", "nyerah aja",

    # 🥀 Kalimat tidak langsung
    "hidup gak ada artinya", "hidup gak guna", "hidup sia sia",
    "aku pengen hilang", "ingin hilang", "pengen ngilang", "ingin pergi selamanya",
    "mending mati", "lebih baik mati", "biar aku mati aja",
    "ingin tidur selamanya", "ingin berhenti hidup",

    # ⚠️ Perilaku menyakiti diri
    "lukai diri", "melukai diri", "sayat", "nyakitin diri", "self harm",
    "aku menyakiti diri", "aku pengen nyakitin diri", "pengen sayat",
    "pengen nyakitin badan",

    # 😞 Kalimat hopeless
    "aku gak berharga", "aku gagal", "semuanya percuma",
    "hidup ini sia sia", "aku menyerah", "aku nyerah", "udah gak ada harapan",
    "gak ada gunanya hidup"
]]

def contains_crisis_keyword(s_proc):
    for k in CRISIS_KEYWORDS:
        if k in s_proc:
            return True, k
    return False, None

# ====== Classification ======
def classify_text(text):
    s_proc = preprocess_text(text)
    kw_match, kw = contains_crisis_keyword(s_proc)
    if kw_match:
        response = random.choice(df_krisis["Respon"].dropna().tolist())
        return {"label": "Krisis", "prob": 1.0, "reason": f"keyword:{kw}", "respon": response}

    if len(s_proc.split()) <= 2:
        return {"label": "Netral", "prob": None, "reason": "short_text", "respon": "Saya mendengarkan, silakan ceritakan"}

    v = vectorizer.transform([s_proc])
    prob = float(model.predict_proba(v)[0, 1])
    label = "Krisis" if prob >= chosen_threshold else "Tidak Krisis"
    responses = df_krisis["Respon"] if label == "Krisis" else df_tidak["Respon"]
    response = random.choice(responses.dropna().tolist()) if not responses.empty else "Saya mendengarkan kamu 💙"

    return {"label": label, "prob": prob, "reason": "model", "respon": response}

# ====== TTS ======
def speak_text(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        voices = engine.getProperty("voices")
        for v in voices:
            if "indonesia" in v.name.lower() or "id" in v.id.lower():
                engine.setProperty("voice", v.id)
                break
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# ====== Speech Recognition ======
recognizer = sr.Recognizer()
mic = sr.Microphone()

recognizer.dynamic_energy_threshold = True
recognizer.energy_threshold = 300
recognizer.pause_threshold = 0.8

print("\n🎤 Sistem siap mendengarkan (ucapkan 'exit' untuk berhenti).\n")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=1.0)
    print("✅ Kalibrasi selesai, mulai mendengarkan...")

def callback(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio, language="id-ID")
        print(f"\n🗣️ Anda berkata: {text}")

        if text.strip().lower() == "exit":
            print("🚪 Keluar dari program...")
            stop_listening(wait_for_stop=False)
            return

        result = classify_text(text)
        print("📊 Prediksi:", result["label"], f"(Prob: {result['prob']})")
        print("💡 Tanggapan:", result["respon"])
        speak_text(result["respon"])

    except sr.UnknownValueError:
        print("❌ Tidak bisa mengenali suara.")
    except sr.RequestError as e:
        print(f"⚠️ Error STT: {e}")

stop_listening = recognizer.listen_in_background(mic, callback, phrase_time_limit=10)

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("⛔ Dihentikan oleh user.")
    stop_listening(wait_for_stop=False)
