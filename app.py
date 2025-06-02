import io
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4' 
import re
import pandas as pd
import numpy as np
from flask import Flask, flash, render_template, request, redirect, send_file, url_for, session
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.secret_key = 'your-secret-key-here'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

users = {
    'admin': 'halo123',
}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('upload'))
        else:
            flash('Username atau password salah', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# function untuk visualisasi hasil segmentasi
def generate_visualizations(df):
    img_data = {}
    
    # Plot 1: Total Sales by Customer
    plt.figure(figsize=(5, 6))
    sales_by_customer = df.groupby('customer')['total_penjualan'].sum().sort_values(ascending=False).head(5)
    sales_by_customer.plot(kind='bar')
    plt.title('Top 5 Customers by Total Sales')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_data['sales_by_customer'] = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    
    # Plot 2: Sales by Category
    plt.figure(figsize=(10, 6))
    sales_by_category = df.groupby('kategori')['total_penjualan'].sum().sort_values(ascending=False)
    products_by_category = df['kategori'].value_counts()

    sales_chart_data = {
        'labels': sales_by_category.index.tolist(),
        'data': sales_by_category.values.tolist(),
        'colors': [
            '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', 
            '#59a14f', '#edc948', '#b07aa1', '#ff9da7'
        ]
    }
    
    product_chart_data = {
        'labels': products_by_category.index.tolist(),
        'data': products_by_category.values.tolist(),
        'colors': ['#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']
    }

    sales_by_category.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Sales Distribution by Category')
    plt.ylabel('')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_data['sales_by_category'] = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    
    # Plot 3: Payment Method Distribution
    plt.figure(figsize=(8, 5))
    payment_dist = df['jenis_pembayaran'].value_counts()
    payment_dist.plot(kind='bar')
    plt.title('Payment Method Distribution')
    plt.ylabel('Count')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_data['payment_dist'] = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    
    return img_data, sales_chart_data, product_chart_data

def preprocess_text(text):
    # 1. Remove product codes (numbers/patterns that don't carry meaning)
    text = re.sub(r'\b\d+[x×]\d+\b', '', text)  # Remove dimensions like 020x30
    text = re.sub(r'\b\d{3,}\b', '', text)     # Remove long number sequences
    
    # 2. Keep only meaningful words
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # Remove special chars/numbers
    text = ' '.join([word for word in text.split() if len(word) > 3])  # Remove short words
    
    return text.strip()

def perform_segmentation(df):
    df['jumlah'] = pd.to_numeric(df['jumlah'], errors='coerce')
    
    # persiapan data
    customer_data = df.groupby('customer_id').agg({
        'total_penjualan': 'sum',
        'jumlah': 'sum',
        'kode_item': 'nunique'
    }).reset_index()
    customer_data.columns = ['customer_id', 'total_spent', 'total_items', 'unique_products']
    
    # clean numerical values by removing thousand separators and converting to float
    for col in ['total_spent', 'total_items']:
        customer_data[col] = customer_data[col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)
    
    # normalisasi data    
    scaler = StandardScaler()
    X = scaler.fit_transform(customer_data[['total_spent', 'total_items', 'unique_products']])
    
    # penggunaan silhouette score untuk menentukan jumlah cluster yang paling optimal
    silhouette_scores = []
    for n_cluster in range(2, 8):
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        label = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, label))

# Tentukan jumlah cluster optimal berdasarkan nilai silhouette score tertinggi
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

# Penerapan KMeans dengan jumlah cluster yang optimal
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    customer_data['segment'] = kmeans.fit_predict(X)
    # segmentation plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=customer_data, x='total_spent', y='unique_products', hue='segment', palette='viridis')
    plt.title('Customer Segmentation')
    plt.xlabel('Total Spent')
    plt.ylabel('Unique Products Purchased')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    seg_plot = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    
    return customer_data, seg_plot

@app.route('/')
def index():
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():

    
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # cek file yang di upload
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
                        
            session['uploaded_file'] = filepath
            return redirect(url_for('insight'))
    
    return render_template('upload.html')

@app.route('/insight',methods=['GET', 'POST'])
def insight():   

    if 'user' not in session:
        return redirect(url_for('login')) 
    
    filepath = session.get('uploaded_file', None)
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for('upload'))
        
    # load data
    df = pd.read_csv(filepath)


    # buat visualisasi
    img_data, sales_chart_data, product_chart_data = generate_visualizations(df)
    
    # melakukan segmentasi
    customer_data, seg_plot = perform_segmentation(df)
    
    # fet top customers per segment
    top_customers = customer_data.sort_values(['segment', 'total_spent'], ascending=[True, False])

    # siapkan data untuk text analisis
    text_data = df.groupby('customer_id')['nama_item'].apply(lambda x: ' '.join(x)).reset_index()
    text_data['nama_item'] = text_data['nama_item'].apply(preprocess_text)
    # text_data = df.groupby('customer_id')['kategori'].apply(lambda x: ' '.join(x)).reset_index()

    # split data
    X_train, X_test = train_test_split(text_data, test_size=0.2, random_state=42)
    
    # text processing - BoW, Unigram, Trigram, TF-IDF
    
    # BoW
    bow_vectorizer = CountVectorizer(
        preprocessor=preprocess_text,
        stop_words=None,  
        min_df=2,         
        max_features=100  
    )

    X_train_bow = bow_vectorizer.fit_transform(X_train['nama_item'])
    
    # unigram
    unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
    X_train_unigram = unigram_vectorizer.fit_transform(X_train['nama_item'])
    
    # trigram
    trigram_vectorizer = CountVectorizer(ngram_range=(3, 3))
    X_train_trigram = trigram_vectorizer.fit_transform(X_train['nama_item'])
    
    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        stop_words=None,
        min_df=3,
        max_features=50
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['nama_item'])
    
    # get feature names for display
    bow_features = bow_vectorizer.get_feature_names_out()[:10]
    unigram_features = unigram_vectorizer.get_feature_names_out()[:10]
    trigram_features = trigram_vectorizer.get_feature_names_out()[:10]
    tfidf_features = tfidf_vectorizer.get_feature_names_out()[:10]
    
    # buat object untuk segment
    segments = []
    for segment_num in sorted(customer_data['segment'].unique()):
        segment_customers = customer_data[customer_data['segment'] == segment_num]
                
        customer_details = []
        for _, row in segment_customers.iterrows():
            customer_id = row['customer_id']
            customer_df = df[df['customer_id'] == customer_id]
            
            customer_details.append({
                'customer_id': customer_id,
                'customer_name': customer_df['customer'].iloc[0],
                'total_spent': row['total_spent'],
                'total_items': row['total_items'],
                'unique_products': row['unique_products'],
                'city': customer_df['kota'].iloc[0].split(',')[0] if pd.notna(customer_df['kota'].iloc[0]) else 'Unknown'
            })
        
        # sort customers by total spent (descending)
        customer_details.sort(key=lambda x: x['total_spent'], reverse=True)
        
        segments.append({
            'size': len(customer_details),
            'total_spent_mean': segment_customers['total_spent'].mean(),
            'unique_products_mean': segment_customers['unique_products'].mean(),
            'customers': customer_details
        })

    return render_template('insight.html', 
                         sales_chart_data=sales_chart_data,
                         product_chart_data=product_chart_data,
                         img_data=img_data,
                         seg_plot=seg_plot,
                         customer_data=top_customers.to_dict('records'),
                         bow_features=bow_features,
                         unigram_features=unigram_features,
                         trigram_features=trigram_features,
                         tfidf_features=tfidf_features,
                         columns=df.columns.tolist(),
                         segments=segments,
                         rows=df.head(10).to_dict('records'))

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/download_segments')
def download_segments():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    filepath = session.get('uploaded_file', None)
    if not filepath or not os.path.exists(filepath):
        flash('Data belum diupload atau tidak ditemukan', 'warning')
        return redirect(url_for('upload'))
    
    # Load data original
    df = pd.read_csv(filepath)
    # Dapatkan hasil segmentasi seperti di fungsi insight()
    customer_data, _ = perform_segmentation(df)
    
    # Convert ke CSV dalam memory buffer
    csv_buffer = io.StringIO()
    customer_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Kirim file sebagai attachment
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='customer_segments.csv'
    )

if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)