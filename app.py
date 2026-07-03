import os, json, time, re, traceback
import numpy as np
from collections import defaultdict
from flask import Flask, render_template, request, jsonify


# GEMINI API KEY 0
GEMINI_API_KEY = "gemini key"

# ── Safe imports ──────────────────────────────────────────
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Pillow not installed. Run: pip install Pillow")

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
    print("Gemini SDK ready.")
except ImportError:
    GEMINI_AVAILABLE = False
    print("Gemini SDK not installed. Run: pip install google-generativeai")

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Plant Scan disabled.")

# ── Load herbal database ──────────────────────────────────
DATABASE_FILE = 'herbal_database.json'
if os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
        herbal_data = json.load(f)
    print(f"Database loaded: {len(herbal_data)} herbs")
else:
    herbal_data = {}
    print("herbal_database.json not found!")

# ── Load plant model ──────────────────────────────────────
model = None
MODEL_PATHS = [
    'herbal_model.h5',
    'models/herbal_model.h5',
    'model/herbal_model.h5',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'herbal_model.h5'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'herbal_model.h5'),
]
if TF_AVAILABLE:
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                model = load_model(path)
                print(f"Plant model loaded: {path}")
                break
            except Exception as e:
                print(f"Failed loading {path}: {e}")
    if model is None:
        print("herbal_model.h5 not found. Put it in the same folder as app.py")

# ── Flask setup ───────────────────────────────────────────
app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)
os.makedirs('data',    exist_ok=True)
CONFIDENCE_THRESHOLD = 0.65
ALLOWED = {'png', 'jpg', 'jpeg', 'webp'}

def allowed(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED

# ── Rate limiter ──────────────────────────────────────────
req_log = defaultdict(list)
def rate_limited(ip, mx=15, win=60):
    now = time.time()
    req_log[ip] = [t for t in req_log[ip] if now - t < win]
    if len(req_log[ip]) >= mx:
        return True
    req_log[ip].append(now)
    return False

# ── Usage stats ───────────────────────────────────────────
STATS_FILE = 'data/usage_stats.json'

def load_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "total_remedy_searches": 0,
        "total_plant_scans":     0,
        "total_skin_checks":     0,
        "remedy_counts":         {},
        "plant_counts":          {},
        "skin_condition_counts": {}
    }

def save_stats(stats):
    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Stats save failed: {e}")

def record_event(category, label=None):
    stats = load_stats()
    if category == 'remedy':
        stats['total_remedy_searches'] += 1
        if label:
            stats['remedy_counts'][label] = stats['remedy_counts'].get(label, 0) + 1
    elif category == 'plant':
        stats['total_plant_scans'] += 1
        if label:
            stats['plant_counts'][label] = stats['plant_counts'].get(label, 0) + 1
    elif category == 'skin':
        stats['total_skin_checks'] += 1
        if label:
            stats['skin_condition_counts'][label] = stats['skin_condition_counts'].get(label, 0) + 1
    save_stats(stats)

# ── Symptom map ───────────────────────────────────────────
SYMPTOM_MAP = {
    "0":  ["hair","baal","dandruff","scalp","hair loss","hair fall","cough","khansi","pyaaz"],
    "1":  ["burn","jalan","aloe","skin","sunburn","wound","dry skin","cut","rash"],
    "2":  ["joint","arthritis","joint pain","rheumatoid","inflammation","swelling"],
    "3":  ["digestion","iron","anemia","hemoglobin","cholai","vitamin deficiency"],
    "4":  ["fever","bukhar","malaria","dengue","viral","immunity","kalmegh","flu"],
    "5":  ["ulcer","skin ulcer","boil","wound healing","sharifa","diabetic wound"],
    "6":  ["jackfruit","kathal","immune boost","bowel","constipation","antioxidant"],
    "7":  ["acne","pimple","fungal","skin disease","blood purifier","neem","allergy"],
    "8":  ["memory","brain","stress","anxiety","focus","brahmi","concentration"],
    "9":  ["bone","nail","silica","bamboo","bone strength"],
    "10": ["constipation","laxative","mouth ulcer","fiber","bowel movement"],
    "11": ["muscle pain","cold","chest congestion","mustard","sarson","circulation"],
    "12": ["blood sugar","weight","kohlrabi","metabolism","cardiovascular"],
    "13": ["chronic pain","aak","madar","external swelling","severe inflammation"],
    "14": ["capsaicin","chilly","mirch","pain relief","topical pain"],
    "15": ["sciatica","back pain","nerve pain","lower back","limb stiffness"],
    "16": ["papaya","papita","platelet","dengue fever","stomach","indigestion"],
    "17": ["anemia","low iron","karonda","blood pressure","acidity"],
    "18": ["glucose","blood glucose","sadabahar","sugar control"],
    "19": ["insulin","diabetes","blood sugar control","diabetic","insulin plant"],
    "20": ["cramp","muscle cramp","blocked nose","camphor","kafoor","sprain"],
    "21": ["lemon","nimbu","vitamin c","detox","liver cleanse","weight loss"],
    "22": ["acidity","nausea","vomiting","acid reflux","gas","bloating"],
    "23": ["headache","sar dard","migraine","head pain","alertness","coffee"],
    "24": ["kidney stone","pathri","asthma","wheezing","phlegm","ajwain"],
    "25": ["taro","arbi","dietary fiber","smooth digestion"],
    "26": ["body heat","burning urine","coriander","dhania","cooling"],
    "27": ["prostate","worms","parasites","pumpkin","kaddu","urinary tract"],
    "28": ["turmeric","haldi","golden milk","anti inflammatory","healing"],
    "29": ["lemongrass","sleep","insomnia","detox tea","calm"],
    "30": ["old wound","kaner","stubborn infection","wound oil"],
    "31": ["hair greying","grey hair","bhringraj","liver","hair tonic"],
    "32": ["eucalyptus","safeda","sinus","blocked nose","mucus","cold congestion"],
    "33": ["shortness of breath","bronchial","dysentery","dudhi","wheeze"],
    "34": ["diarrhea","loose stool","bleeding cut","minor cut"],
    "35": ["peepal","blood clean","chronic cough","skin cleanse"],
    "36": ["cough","khansi","urinary infection","gul e makhmal","high blood pressure"],
    "37": ["piles","hemorrhoid","rectal pain"],
    "38": ["hibiscus","gudhal","hair thinning","cholesterol","heart health"],
    "39": ["fatigue","chronic fatigue","fungal infection","jatoba","energy"],
    "40": ["blood dysentery","eczema","skin rash","ixora"],
    "41": ["jasmine","chambeli","mood","skin blemish","mental calm"],
    "42": ["bronchitis","expectorant","adusa","dry phlegm","persistent cough"],
    "43": ["antiseptic","infected wound","lantana","wound wash"],
    "44": ["henna","mehendi","cooling feet","burning feet","hair condition"],
}

def find_by_symptom(text):
    s = text.lower().strip()
    best_id, best_score = None, 0
    for cid, kws in SYMPTOM_MAP.items():
        score = sum(len(kw.split()) for kw in kws if kw in s)
        if score > best_score:
            best_score, best_id = score, cid
    if best_score == 0:
        for cid, d in herbal_data.items():
            db = f"{d.get('herb','')} {d.get('disease_type','')} {d.get('local_name','')} {' '.join(d.get('tags',[]))}".lower()
            if s in db:
                return cid
    return best_id if best_score > 0 else None

def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    clean = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    match = re.search(r'\{[\s\S]*\}', clean)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None

# ═════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('main.html')


# ── 1. Symptom Remedy ─────────────────────────────────────
@app.route('/get_remedy', methods=['POST'])
def get_remedy():
    if rate_limited(request.remote_addr):
        return jsonify({'error': 'Too many requests. Please wait.'}), 429
    symptom = (request.json or {}).get('symptom', '').strip()
    if not symptom:
        return jsonify({'error': 'Please enter a symptom'}), 400
    cid = find_by_symptom(symptom)
    if cid and cid in herbal_data:
        d = herbal_data[cid]
        record_event('remedy', d.get('herb', ''))
        return jsonify({k: d.get(k, '') for k in
            ['herb','local_name','urdu_name','disease_type',
             'recipe','recipe_ur','benefit','benefit_ur']})
    return jsonify({'error': 'No matching herb found. Try different keywords.'}), 404


# ── 2. Plant Leaf Scan ────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if rate_limited(request.remote_addr):
        return jsonify({'error': 'Too many requests.'}), 429
    if not TF_AVAILABLE or model is None:
        return jsonify({'error':
            'Plant scan unavailable. '
            'Make sure herbal_model.h5 is in the same folder as app.py.'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    file = request.files['file']
    if not file.filename or not allowed(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG or WEBP.'}), 400
    img_path = os.path.join('uploads', f"plant_{int(time.time())}.jpg")
    file.save(img_path)
    try:
        img  = keras_image.load_img(img_path, target_size=(240, 240))
        arr  = np.expand_dims(keras_image.img_to_array(img), axis=0) / 255.0
        pred = model.predict(arr)
        result     = int(np.argmax(pred))
        confidence = float(np.max(pred))
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'error': f'Low confidence ({round(confidence*100,1)}%). '
                         'Try a clearer, closer photo of the leaf.'
            }), 422
        d = herbal_data.get(str(result))
        if not d:
            return jsonify({
                'herb': f'Unknown (Class #{result})', 'local_name': 'N/A',
                'disease_type': 'Scan Result', 'recipe': 'N/A',
                'benefit': 'Not found in database.',
                'confidence': round(confidence * 100, 2)
            })
        record_event('plant', d.get('herb', ''))
        return jsonify({
            **{k: d.get(k, '') for k in
               ['herb','local_name','urdu_name','disease_type',
                'recipe','recipe_ur','benefit','benefit_ur']},
            'confidence': round(confidence * 100, 2)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


# ── 3. Skin Check ─────────────────────────────────────────
# BOTH route names work: /predict_skin AND /skin_check
# This fixes the mismatch between old HTML and new HTML
@app.route('/predict_skin', methods=['POST'])
@app.route('/skin_check',   methods=['POST'])
def skin_check():
    if rate_limited(request.remote_addr):
        return jsonify({'error': 'Too many requests.'}), 429
    if not GEMINI_AVAILABLE:
        return jsonify({'error': 'Gemini SDK not installed. Run: pip install google-generativeai'}), 503
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or not GEMINI_API_KEY.strip():
        return jsonify({'error': 'Add your Gemini API key in app.py line 11.'}), 503
    if not PIL_AVAILABLE:
        return jsonify({'error': 'Pillow not installed. Run: pip install Pillow'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    file = request.files['file']
    if not file.filename or not allowed(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG or WEBP.'}), 400

    img_path = os.path.join('uploads', f"skin_{int(time.time())}.jpg")
    file.save(img_path)

    try:
        genai.configure(api_key=GEMINI_API_KEY)

        # ── Robust image preparation ──────────────────────────
        # Fixes EXIF rotation, CMYK, RGBA, palette mode, large files
        import io
        try:
            pil_img = Image.open(img_path)

            # Convert any unusual mode to plain RGB
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # Resize if extremely large (Gemini has upload limits)
            max_dim = 1600
            w, h = pil_img.size
            if max(w, h) > max_dim:
                ratio = max_dim / max(w, h)
                pil_img = pil_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        except Exception as img_err:
            print(f"Image processing error: {img_err}")
            return jsonify({'error':
                f'Could not process this image ({str(img_err)}). '
                'Please try a different, well-lit photo.'}), 422

        # Convert to clean JPEG bytes — most reliable for Gemini SDK
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=85)
        img_bytes = buffer.getvalue()

        # Use inline_data format — correct way to send images to Gemini
        image_part = {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_bytes
            }
        }

        ai_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

        prompt = """You are Dr. Sabz, a senior dermatologist and herbal medicine consultant with 20 years of clinical experience treating patients in Pakistan and South Asia.

A patient has uploaded a photo of their skin condition. Examine it carefully — look at color, texture, pattern, spread, and any signs of infection or inflammation.

Respond warmly and in full detail as a real doctor would to a real patient.

IMPORTANT: Return ONLY a valid JSON object. Start your response with { and end with }. No markdown, no code fences, no extra text before or after.

Required JSON structure:
{
  "condition": "Exact skin condition name in English",
  "condition_ur": "اردو میں حالت کا نام",
  "severity": "Mild",
  "severity_ur": "ہلکی",
  "description": "Write 4 to 6 warm detailed sentences directly to the patient: what this condition is, why it likely happened, what they should expect, and reassurance.",
  "description_ur": "وہی تفصیل اردو میں، 4 سے 6 جملے",
  "symptoms": ["Symptom one in detail", "Symptom two", "Symptom three", "Symptom four", "Symptom five"],
  "symptoms_ur": ["پہلی علامت", "دوسری علامت", "تیسری علامت", "چوتھی علامت", "پانچویں علامت"],
  "herbal_remedies": [
    {
      "herb": "Herb common name (Botanical name)",
      "herb_ur": "اردو نام",
      "how_to_use": "Full step by step: how to prepare, how to apply, quantity, times per day, for how many days.",
      "how_to_use_ur": "اردو میں مکمل طریقہ استعمال",
      "benefit": "Exactly why this herb helps this specific condition in 2 sentences."
    },
    {
      "herb": "Second herb (Botanical name)",
      "herb_ur": "دوسری جڑی بوٹی کا نام",
      "how_to_use": "Full step by step preparation and application.",
      "how_to_use_ur": "اردو میں مکمل ہدایات",
      "benefit": "Why this herb helps in 2 sentences."
    },
    {
      "herb": "Third herb (Botanical name)",
      "herb_ur": "تیسری جڑی بوٹی کا نام",
      "how_to_use": "Full step by step preparation and application.",
      "how_to_use_ur": "اردو میں مکمل ہدایات",
      "benefit": "Why this herb helps in 2 sentences."
    }
  ],
  "do_list": ["Specific do item as full sentence", "Another do", "Another do", "Another do"],
  "do_list_ur": ["یہ کریں مکمل جملے میں", "دوسرا", "تیسرا", "چوتھا"],
  "dont_list": ["Specific avoid item as full sentence", "Another dont", "Another dont", "Another dont"],
  "dont_list_ur": ["یہ نہ کریں مکمل جملے میں", "دوسرا", "تیسرا", "چوتھا"],
  "expected_recovery": "2 sentences on realistic recovery time with consistent herbal care.",
  "expected_recovery_ur": "اردو میں 2 جملے",
  "doctor_warning": true,
  "not_skin": false
}

Rules:
- severity must be exactly: Mild, Moderate, or Severe — nothing else
- Set doctor_warning true if condition is serious, contagious, spreading, or needs medical care
- If the photo is NOT a skin condition, set not_skin to true and condition to Not a skin condition
- Start with { end with } — absolutely nothing outside the JSON"""

        response = ai_model.generate_content(
            contents=[image_part, prompt],
            generation_config={"response_mime_type": "application/json"}
        )

        print("=== GEMINI RAW RESPONSE ===")
        print(response.text[:600])
        print("===========================")

        skin_data = extract_json(response.text)

        if skin_data is None:
            print("Could not parse JSON:")
            print(response.text)
            return jsonify({'error': 'AI returned unexpected format. Please try again.'}), 500

        if not skin_data.get('not_skin'):
            record_event('skin', skin_data.get('condition', 'Unknown'))

        return jsonify(skin_data)

    except Exception as e:
        print("=== SKIN CHECK ERROR ===")
        traceback.print_exc()
        print(f"Error: {str(e)}")
        print("========================")
        return jsonify({'error': f'Skin analysis failed: {str(e)}'}), 500

    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


# ── 4. Admin Dashboard ────────────────────────────────────
@app.route('/admin')
def admin_dashboard():
    stats = load_stats()
    def top5(d): return sorted(d.items(), key=lambda x: x[1], reverse=True)[:5]
    return render_template('admin.html',
        total_remedy  = stats['total_remedy_searches'],
        total_plant   = stats['total_plant_scans'],
        total_skin    = stats['total_skin_checks'],
        top_remedies  = top5(stats['remedy_counts']),
        top_plants    = top5(stats['plant_counts']),
        top_skin      = top5(stats['skin_condition_counts']),
        model_status  = "Loaded" if model is not None else "Not found — put herbal_model.h5 next to app.py",
        gemini_status = "Configured" if (GEMINI_API_KEY != "YOUR_GEMINI_API_KEY" and GEMINI_API_KEY.strip()) else "Not configured",
        total_herbs   = len(herbal_data)
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)