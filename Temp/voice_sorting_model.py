import os, shutil, librosa, numpy as np, pygame, time, warnings, pickle, zipfile, copy
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
pygame.mixer.init()

# --- НАСТРОЙКИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "audio_input")
OUTPUT_DIR = os.path.join(BASE_DIR, "audio_sorted_results")
MODEL_BUNDLE = os.path.join(BASE_DIR, "voices_library.vmdl")

SURE_THRESHOLD = 0.995   
DOUBT_THRESHOLD = 0.40   

def get_features(path):
    try:
        y, sr = librosa.load(path, sr=16000, duration=10)
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mels=40, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except: return None

def play_audio(path, label=""):
    if path and os.path.exists(path):
        print(f">> Слушаем {label}: {os.path.basename(path)}")
        pygame.mixer.music.load(path); pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): pygame.time.Clock().tick(10)

def find_best_sample_in_folder(folder_path, target_feat):
    """Возвращает (путь_строка, сходство_число)"""
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.ogg'))]
    if not files: return None, 0.0
    best_file, max_sim = None, -1.0
    for f_name in files:
        f_p = os.path.join(folder_path, f_name)
        feat = get_features(f_p)
        if feat is not None:
            sim = float(cosine_similarity(target_feat.reshape(1,-1), feat.reshape(1,-1)).item())
            if sim > max_sim:
                max_sim = sim
                best_file = f_p
    return best_file, max_sim 

def save_bundle(kb, bundle_path):
    with zipfile.ZipFile(bundle_path, 'w') as zf:
        zf.writestr('database.pkl', pickle.dumps(kb))

def load_bundle(bundle_path):
    if not os.path.exists(bundle_path): return {}
    try:
        with zipfile.ZipFile(bundle_path, 'r') as zf:
            return pickle.loads(zf.read('database.pkl'))
    except: return {}

# --- ЗАПУСК ---
kb = load_bundle(MODEL_BUNDLE)
session_names = {} 
history = [] 

for d in [OUTPUT_DIR, INPUT_DIR]: os.makedirs(d, exist_ok=True)
files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a'))])
total_files = len(files)

f_idx = 0
while f_idx < len(files):
    f_name = files[f_idx]
    path = os.path.join(INPUT_DIR, f_name)
    feat = get_features(path)
    if feat is None: f_idx += 1; continue
        
    assigned_id, is_manual, undo_triggered = None, False, False
    kb_before = copy.deepcopy(kb)
    sn_before = copy.deepcopy(session_names)

    all_scores = []
    for v_id, data in kb.items():
        g_score = float(cosine_similarity(feat.reshape(1,-1), data['centroid'].reshape(1,-1)).item())
        e_score = float(cosine_similarity(feat.reshape(1,-1), data.get('ideal_vector', data['centroid']).reshape(1,-1)).item())
        
        s_path, s_score = None, 0.0
        if v_id in session_names:
            folder = os.path.join(OUTPUT_DIR, session_names[v_id])
            if os.path.exists(folder):
                s_path, s_score = find_best_sample_in_folder(folder, feat)
        
        all_scores.append({
            'id': v_id, 
            'score': float(max(g_score, e_score)), 
            's_score': float(s_score), 
            's_path': s_path, 
            'is_named': v_id in session_names
        })
    all_scores.sort(key=lambda x: x['score'], reverse=True)

    # 1. Автоматика
    for res in all_scores:
        if res['is_named']:
            if res['score'] >= SURE_THRESHOLD or res['s_score'] >= 0.999:
                assigned_id = res['id']
                print(f"[{f_idx+1}/{total_files}] Авто: {f_name} -> [{session_names[res['id']]}]")
                break

    # 2. Ручной перебор
    if not assigned_id:
        potential_named = [s for s in all_scores if s['score'] > DOUBT_THRESHOLD and s['is_named']]
        
        # ИСПРАВЛЕННЫЙ поиск лучшего безымянного голоса
        unnamed_candidates = [s for s in all_scores if not s['is_named']]
        best_unnamed = unnamed_candidates[0] if unnamed_candidates else None
        
        curr_p_idx = 0
        while curr_p_idx < len(potential_named):
            res = potential_named[curr_p_idx]
            v_id, display_name = res['id'], session_names[res['id']]
            
            print(f"\n" + "!"*75 + f"\nПРОГРЕСС: [{f_idx+1}/{total_files}] | ПРОВЕРКА ({curr_p_idx+1} из {len(potential_named)}): [{display_name}]")
            print(f"СХОДСТВО: Модель: {res['score']:.4f} | Папка: {res['s_score']:.4f}\n" + "!"*75)
            
            if res['s_path']: play_audio(res['s_path'], "ОБРАЗЕЦ")
            time.sleep(0.1); play_audio(path, "ТЕКУЩИЙ")
            
            is_last = (curr_p_idx == len(potential_named) - 1)
            prompt = "[y] Да | [n] Нет, дальше" if not is_last else "[y] Да | [n] Новая группа"
            ans = input(f"{prompt} | [r] Повтор | [z] Отмена: ").lower().strip()
            
            if ans == 'y':
                assigned_id, is_manual = v_id, True; break
            elif ans == 'n':
                curr_p_idx += 1; continue
            elif ans == 'r': continue
            elif ans == 'z' and history:
                kb, session_names, _, trash_path = history.pop()
                if os.path.exists(trash_path): os.remove(trash_path)
                f_idx -= 1; undo_triggered = True; break
            else: continue

        # 3. Тихое узнавание или Новая группа
        if not assigned_id and not undo_triggered:
            if best_unnamed and best_unnamed['score'] >= SURE_THRESHOLD:
                print(f"\n[{f_idx+1}/{total_files}] ТЕКУЩИЙ: {f_name}")
                play_audio(path, "ТЕКУЩИЙ")
                name = input("Введите имя для этого персонажа: ").strip() or f"hero_{best_unnamed['id']}"
                assigned_id, session_names[best_unnamed['id']], is_manual = best_unnamed['id'], name, True
            else:
                print(f"\n[{f_idx+1}/{total_files}] СОЗДАНИЕ НОВОЙ ГРУППЫ")
                play_audio(path, "ТЕКУЩИЙ")
                name = input("Имя нового персонажа: ").strip() or f"hero_{len(kb)+1}"
                assigned_id = f"v_{len(kb)+1:03d}"
                kb[assigned_id] = {'centroid': feat, 'count': 1, 'ideal_vector': feat}
                session_names[assigned_id] = name
                is_manual = True; save_bundle(kb, MODEL_BUNDLE)

    if undo_triggered: continue
    if assigned_id:
        d = kb[assigned_id]
        if is_manual and d['count'] > 0:
            if not (d['count'] == 1 and np.array_equal(d['centroid'], feat)):
                d['centroid'] = (d['centroid'] * d['count'] + feat) / (d['count'] + 1)
                d['count'] += 1
        dst = os.path.join(OUTPUT_DIR, session_names[assigned_id]); os.makedirs(dst, exist_ok=True)
        final_dst = os.path.join(dst, f_name); shutil.copy2(path, final_dst)
        history.append((kb_before, sn_before, path, final_dst))
        if is_manual: save_bundle(kb, MODEL_BUNDLE)
        f_idx += 1

# --- ФИНАЛ ---
if kb:
    print("\n" + "*"*60 + "\nФИНАЛЬНЫЙ АНАЛИЗ ЭТАЛОНОВ И СОХРАНЕНИЕ...\n" + "*"*60)
    for v_id, data in kb.items():
        name = session_names.get(v_id)
        if name and os.path.exists(os.path.join(OUTPUT_DIR, name)):
            f_p, s_v = find_best_sample_in_folder(os.path.join(OUTPUT_DIR, name), data['centroid'])
            if f_p:
                f_ft = get_features(f_p)
                if f_ft is not None:
                    data['ideal_vector'] = f_ft
                    print(f"Группа: {name:15} | Эталон: {os.path.basename(f_p)} | Точность: {float(s_v):.4f}")
    save_bundle(kb, MODEL_BUNDLE)

input("\nЗавершено. Нажмите Enter для выхода...")
