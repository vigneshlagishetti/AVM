"""
Flask Web Application - IDS in MANET (Enhanced)
=================================================
Features:
- User authentication (SQLite3)
- Dashboard with model performance
- Real-time intrusion prediction
- Prediction history & audit log
- Batch CSV prediction
- Dataset explorer (EDA)
- PDF report generation
"""

import os
import io
import csv
import json
import pickle
import sqlite3
import hashlib
import datetime
import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, send_file, Response)

app = Flask(__name__)
app.secret_key = 'avm_ids_manet_secret_key_2024'

# ============================================================
# DATABASE SETUP
# ============================================================
DB_NAME = 'signup.db'
MODELS_DIR = 'models'


def init_db():
    """Initialize the SQLite database for user authentication and prediction history."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            input_params TEXT,
            result TEXT,
            confidence REAL,
            attack_type TEXT,
            model_votes TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()


def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


# ============================================================
# LOAD TRAINED MODELS
# ============================================================
def load_models():
    """Load all trained models and artifacts."""
    models = {}
    model_files = {
        'Decision Tree': 'decision_tree.pkl',
        'XGBoost': 'xgboost.pkl',
        'Random Forest': 'random_forest.pkl',
        'SVM': 'svm.pkl',
        'Logistic Regression': 'logistic_regression.pkl',
        'MLP Classifier': 'mlp_classifier.pkl',
        'Boosted Regression Tree': 'boosted_regression_tree.pkl'
    }
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)

    artifacts = {}
    for key, fname in [('scaler', 'scaler.pkl'), ('label_encoders', 'label_encoders.pkl'),
                       ('weights', 'ensemble_weights.pkl'), ('feature_info', 'feature_info.pkl')]:
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                artifacts[key] = pickle.load(f)

    results_path = os.path.join(MODELS_DIR, 'results.json')
    results = None
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)

    return (models, artifacts.get('scaler'), artifacts.get('label_encoders'),
            artifacts.get('weights'), artifacts.get('feature_info'), results)


MODELS, SCALER, LABEL_ENCODERS, WEIGHTS, FEATURE_INFO, RESULTS = {}, None, None, None, None, None

def ensure_models_loaded():
    global MODELS, SCALER, LABEL_ENCODERS, WEIGHTS, FEATURE_INFO, RESULTS
    if not MODELS:
        MODELS, SCALER, LABEL_ENCODERS, WEIGHTS, FEATURE_INFO, RESULTS = load_models()


# ============================================================
# INPUT PARAMETERS
# ============================================================
INPUT_PARAMS = [
    {'name': 'protocol_type', 'label': 'Protocol Type', 'type': 'select',
     'options': ['tcp', 'udp', 'icmp'], 'help': 'Network protocol used'},
    {'name': 'service', 'label': 'Service', 'type': 'select',
     'options': ['http', 'smtp', 'ftp_data', 'ftp', 'ssh', 'telnet', 'private',
                 'domain_u', 'eco_i', 'other'],
     'help': 'Network service on destination'},
    {'name': 'src_bytes', 'label': 'Source Bytes', 'type': 'number', 'help': 'Bytes from source to destination'},
    {'name': 'dst_bytes', 'label': 'Destination Bytes', 'type': 'number', 'help': 'Bytes from destination to source'},
    {'name': 'wrong_fragment', 'label': 'Wrong Fragment', 'type': 'number', 'help': 'Number of wrong fragments'},
    {'name': 'num_failed_logins', 'label': 'Failed Logins', 'type': 'number', 'help': 'Failed login attempts'},
    {'name': 'num_compromised', 'label': 'Compromised Nodes', 'type': 'number', 'help': 'Compromised conditions'},
    {'name': 'root_shell', 'label': 'Root Shell', 'type': 'number', 'help': '1 if root shell obtained; 0 otherwise'},
    {'name': 'num_root', 'label': 'Num Root', 'type': 'number', 'help': 'Number of root accesses'},
    {'name': 'num_file_creations', 'label': 'File Creations', 'type': 'number', 'help': 'File creation operations'},
    {'name': 'is_guest_login', 'label': 'Guest Login', 'type': 'number', 'help': '1 if guest login; 0 otherwise'},
    {'name': 'count', 'label': 'Service Count', 'type': 'number', 'help': 'Connections to same host in 2s'},
    {'name': 'serror_rate', 'label': 'Server Error Rate', 'type': 'float', 'help': 'SYN error percentage'},
    {'name': 'dst_host_count', 'label': 'Dst Host Count', 'type': 'number', 'help': 'Same dest host connections'},
    {'name': 'dst_host_srv_count', 'label': 'Dst Host Srv Count', 'type': 'number', 'help': 'Same dest+service count'},
    {'name': 'dst_host_serror_rate', 'label': 'Dst Host Error Rate', 'type': 'float', 'help': 'SYN error %'},
    {'name': 'dst_host_same_srv_rate', 'label': 'Dst Host Same Srv Rate', 'type': 'float', 'help': 'Same service %'},
    {'name': 'dst_host_diff_srv_rate', 'label': 'Dst Host Diff Srv Rate', 'type': 'float', 'help': 'Diff service %'},
    {'name': 'dst_host_srv_diff_host_rate', 'label': 'Dst Srv Diff Host Rate', 'type': 'float', 'help': 'Diff host %'}
]


# ============================================================
# PREDICTION HELPER
# ============================================================
def run_prediction(input_data):
    """Run the adaptive voting ensemble prediction on input data dict."""
    all_features = FEATURE_INFO['all_features']
    feature_vector = np.zeros(len(all_features))

    for i, feat in enumerate(all_features):
        if feat in input_data:
            val = input_data[feat]
            if isinstance(val, str) and feat in LABEL_ENCODERS:
                le = LABEL_ENCODERS[feat]
                feature_vector[i] = le.transform([val])[0] if val in le.classes_ else 0
            else:
                feature_vector[i] = float(val)

    feature_vector = feature_vector.reshape(1, -1)
    feature_vector_scaled = SCALER.transform(feature_vector)
    selected_indices = FEATURE_INFO['selected_indices']
    feature_vector_selected = feature_vector_scaled[:, selected_indices]

    individual_predictions = {}
    for name, model in MODELS.items():
        individual_predictions[name] = int(model.predict(feature_vector_selected)[0])

    n_classes = 2
    weighted_votes = np.zeros(n_classes)
    for name, model in MODELS.items():
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_vector_selected)[0]
        else:
            proba = np.zeros(n_classes)
            proba[individual_predictions[name]] = 1.0
        weighted_votes += WEIGHTS[name] * proba

    final_prediction = int(np.argmax(weighted_votes))
    confidence = float(np.max(weighted_votes) / np.sum(weighted_votes) * 100)
    class_le = LABEL_ENCODERS['class']
    result_label = class_le.inverse_transform([final_prediction])[0]

    attack_type = None
    if result_label == 'anomaly':
        if input_data.get('serror_rate', 0) > 0.5:
            attack_type = 'DoS (Denial of Service)'
        elif input_data.get('num_failed_logins', 0) > 0 or input_data.get('root_shell', 0) > 0:
            attack_type = 'R2L (Remote to Local)'
        elif input_data.get('dst_host_diff_srv_rate', 0) > 0.5:
            attack_type = 'Probe'
        elif input_data.get('num_compromised', 0) > 0 or input_data.get('num_root', 0) > 0:
            attack_type = 'U2R (User to Root)'
        else:
            attack_type = 'Generic Attack'

    model_results = {}
    for name, pred_val in individual_predictions.items():
        model_results[name] = {
            'prediction': class_le.inverse_transform([pred_val])[0],
            'weight': round(WEIGHTS[name] * 100, 2)
        }

    return result_label, attack_type, round(confidence, 2), model_results


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        if not email or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html')
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email=? AND password=?',
                       (email, hash_password(password)))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user'] = {'id': user[0], 'name': user[1], 'email': user[2]}
            flash(f'Welcome back, {user[1]}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'error')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm = request.form.get('confirm_password', '').strip()
        if not all([name, email, password, confirm]):
            flash('Please fill in all fields.', 'error')
            return render_template('signup.html')
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('signup.html')
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                           (name, email, hash_password(password)))
            conn.commit()
            conn.close()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists.', 'error')
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/home')
def home():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    ensure_models_loaded()
    return render_template('home.html', user=session['user'], results=RESULTS, feature_info=FEATURE_INFO)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    ensure_models_loaded()
    if request.method == 'POST':
        try:
            input_data = {}
            for param in INPUT_PARAMS:
                val = request.form.get(param['name'], '0')
                if param['type'] == 'select':
                    input_data[param['name']] = val
                elif param['type'] == 'float':
                    input_data[param['name']] = float(val)
                else:
                    input_data[param['name']] = int(val)

            result_label, attack_type, confidence, model_results = run_prediction(input_data)

            # Log prediction to DB
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO predictions (user_id, input_params, result, confidence, attack_type, model_votes) VALUES (?, ?, ?, ?, ?, ?)',
                    (session['user']['id'], json.dumps(input_data), result_label, confidence,
                     attack_type, json.dumps({k: v['prediction'] for k, v in model_results.items()}))
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

            return render_template('result.html', user=session['user'], result=result_label,
                                   attack_type=attack_type, confidence=confidence,
                                   model_results=model_results, input_data=input_data)
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
    return render_template('predict.html', user=session['user'], params=INPUT_PARAMS)


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    ensure_models_loaded()
    return render_template('dashboard.html', user=session['user'], results=RESULTS, feature_info=FEATURE_INFO)


# ============================================================
# NEW: PREDICTION HISTORY
# ============================================================
@app.route('/history')
def history():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions WHERE user_id=? ORDER BY timestamp DESC LIMIT 100',
                   (session['user']['id'],))
    rows = cursor.fetchall()
    conn.close()

    predictions_list = []
    for row in rows:
        predictions_list.append({
            'id': row[0], 'timestamp': row[2], 'result': row[4],
            'confidence': row[5], 'attack_type': row[6] or '-'
        })

    total = len(predictions_list)
    threats = sum(1 for p in predictions_list if p['result'] == 'anomaly')
    safe = total - threats

    return render_template('history.html', user=session['user'], predictions=predictions_list,
                           total=total, threats=threats, safe=safe)


# ============================================================
# NEW: BATCH CSV PREDICTION
# ============================================================
@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    ensure_models_loaded()

    if request.method == 'POST':
        file = request.files.get('csv_file')
        if not file or not file.filename.endswith('.csv'):
            flash('Please upload a valid CSV file.', 'error')
            return render_template('batch.html', user=session['user'], params=INPUT_PARAMS)

        try:
            stream = io.StringIO(file.stream.read().decode('utf-8'))
            reader = csv.DictReader(stream)
            batch_results = []

            for row_num, row in enumerate(reader, 1):
                input_data = {}
                for param in INPUT_PARAMS:
                    val = row.get(param['name'], '0')
                    if param['type'] == 'select':
                        input_data[param['name']] = val
                    elif param['type'] == 'float':
                        input_data[param['name']] = float(val) if val else 0.0
                    else:
                        input_data[param['name']] = int(float(val)) if val else 0

                result_label, attack_type, confidence, model_results = run_prediction(input_data)
                batch_results.append({
                    'row': row_num, 'result': result_label,
                    'attack_type': attack_type or '-', 'confidence': confidence
                })

            threats = sum(1 for r in batch_results if r['result'] == 'anomaly')
            return render_template('batch.html', user=session['user'], params=INPUT_PARAMS,
                                   batch_results=batch_results, total=len(batch_results), threats=threats)
        except Exception as e:
            flash(f'Batch processing error: {str(e)}', 'error')

    return render_template('batch.html', user=session['user'], params=INPUT_PARAMS)


@app.route('/download-sample-csv')
def download_sample_csv():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[p['name'] for p in INPUT_PARAMS])
    writer.writeheader()
    writer.writerow({p['name']: ('tcp' if p['name'] == 'protocol_type' else 'http' if p['name'] == 'service' else '0')
                     for p in INPUT_PARAMS})
    output.seek(0)
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=sample_input.csv'})


# ============================================================
# NEW: DATASET EXPLORER
# ============================================================
@app.route('/dataset')
def dataset():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    ensure_models_loaded()
    meta = RESULTS.get('_meta', {}) if RESULTS else {}
    return render_template('dataset.html', user=session['user'], meta=meta, feature_info=FEATURE_INFO)


# ============================================================
# NEW: PDF REPORT
# ============================================================
@app.route('/download-report')
def download_report():
    if 'user' not in session:
        return redirect(url_for('login'))
    ensure_models_loaded()
    if not RESULTS:
        flash('No results available. Train models first.', 'error')
        return redirect(url_for('dashboard'))

    # Generate HTML-based report for download
    html = """<html><head><meta charset='utf-8'>
    <title>IDS MANET - Analysis Report</title>
    <style>
    body{font-family:Arial,sans-serif;margin:40px;color:#333}
    h1{color:#667eea;border-bottom:3px solid #667eea;padding-bottom:10px}
    h2{color:#764ba2;margin-top:30px}
    table{border-collapse:collapse;width:100%;margin:15px 0}
    th,td{border:1px solid #ddd;padding:10px;text-align:center}
    th{background:#667eea;color:white}
    tr:nth-child(even){background:#f5f5f5}
    .highlight{background:#e8f5e9!important;font-weight:bold}
    .meta{color:#666;font-size:14px}
    .badge{display:inline-block;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:bold}
    .badge-safe{background:#43e97b;color:white}
    .badge-danger{background:#f5576c;color:white}
    </style></head><body>"""
    html += "<h1>IDS in MANET - Analysis Report</h1>"
    html += f"<p class='meta'>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | User: {session['user']['name']}</p>"
    html += "<h2>Model Performance Comparison</h2><table><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th>Train Time</th></tr>"

    for name, res in RESULTS.items():
        if name.startswith('_'):
            continue
        is_ensemble = name == 'Adaptive Voting Ensemble'
        cls = " class='highlight'" if is_ensemble else ""
        auc_v = res.get('auc', '-')
        tt = res.get('train_time', '-')
        auc_s = f"{auc_v:.4f}" if isinstance(auc_v, float) else str(auc_v)
        tt_s = f"{tt:.3f}s" if isinstance(tt, (int, float)) else str(tt)
        html += f"<tr{cls}><td>{name}</td><td>{res['accuracy']}%</td><td>{res['precision']}%</td><td>{res['recall']}%</td><td>{res['f1_score']}%</td><td>{auc_s}</td><td>{tt_s}</td></tr>"
    html += "</table>"

    if FEATURE_INFO:
        html += "<h2>Selected Features (ABO)</h2><p>"
        html += ", ".join(FEATURE_INFO.get('selected_names', []))
        html += f"</p><p>{len(FEATURE_INFO.get('selected_names', []))} out of {len(FEATURE_INFO.get('all_features', []))} features selected.</p>"

    meta = RESULTS.get('_meta', {})
    if meta.get('dataset_stats'):
        ds = meta['dataset_stats']
        html += f"<h2>Dataset Statistics</h2><p>Samples: {ds.get('total_samples', '-')} | Features: {ds.get('total_features', '-')} | Normal: {ds.get('normal_count', '-')} | Anomaly: {ds.get('anomaly_count', '-')}</p>"

    html += "<h2>Methodology</h2><ol><li>Data Preprocessing (NSL-KDD)</li><li>Label Encoding</li><li>Z-score Normalization</li><li>SMOTE Oversampling</li><li>Artificial Butterfly Optimizer Feature Selection</li><li>7 Classifiers Training</li><li>Adaptive Voting Mechanism</li></ol>"
    html += "</body></html>"

    buffer = io.BytesIO(html.encode('utf-8'))
    buffer.seek(0)
    return send_file(buffer, mimetype='text/html', as_attachment=True,
                     download_name='IDS_MANET_Report.html')


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    init_db()
    try:
        MODELS, SCALER, LABEL_ENCODERS, WEIGHTS, FEATURE_INFO, RESULTS = load_models()
        if MODELS:
            print(f"[OK] Loaded {len(MODELS)} trained models")
        else:
            print("[!] No trained models found. Run 'python train_model.py' first.")
    except Exception as e:
        print(f"[!] Error loading models: {e}")
    print("\n>> Starting IDS Web Application...")
    print("   Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
