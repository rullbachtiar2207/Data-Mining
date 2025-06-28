import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Job Matching System - GNN & GAT",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling - Clean White Theme */
    .main {
        padding-top: 1rem;
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
    }
    
    /* Clean White Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Clean Sidebar */
    .css-1d391kg {
        background: #ffffff;
        border-right: 2px solid #e9ecef;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    
    /* Main Container - Clean */
    .block-container {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        padding: 2rem;
        margin: 1rem;
    }
    
    /* Enhanced Metrics - Clean with Subtle Gradients */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border-color: #3498db;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #3498db, #2980b9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Clean Cards with Subtle Accent */
    .job-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .job-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        background-size: 200% 200%;
        animation: gradientShift 3s ease infinite;
    }
    
    .job-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border-color: #3498db;
    }
    
    .profile-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .profile-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        border-color: #3498db;
    }
    
    /* Animation Keyframes */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.4); }
        70% { box-shadow: 0 0 0 8px rgba(52, 152, 219, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0); }
    }
    
    /* Clean Buttons with Gradient */
    .stButton > button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
        background: linear-gradient(45deg, #2980b9, #3498db);
    }
    
    /* Clean Selectbox */
    .stSelectbox > div > div {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
    }
    
    /* Page Headers - Clean */
    .page-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 12px;
        border: 1px solid #e9ecef;
    }
    
    .page-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(45deg, #2c3e50, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .page-subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        font-weight: 400;
    }
    
    /* Status Indicators - Clean */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-active { background-color: #28a745; }
    .status-pending { background-color: #ffc107; }
    .status-inactive { background-color: #dc3545; }
    
    /* Clean Progress Bars */
    .progress-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 3px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    .progress-bar {
        height: 6px;
        border-radius: 5px;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        transition: width 0.5s ease;
    }
    
    /* Clean Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    
    /* Clean Footer */
    .footer {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 2rem;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Text Colors for Clean Theme */
    h1, h2, h3, h4, h5, h6 {
        color: #7998b8 !important;
    }
    
    .stMarkdown {
        color: #495057;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .page-title { font-size: 2rem; }
        .metric-value { font-size: 2rem; }
        .block-container { margin: 0.5rem; padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# === FUNGSI HELPER UNTUK METRICS ===
def create_metric_card(label, value, icon="📊"):
    """Membuat metric card yang lebih menarik"""
    return f"""
    <div class="metric-container">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def create_progress_bar(percentage, label="Progress"):
    """Membuat progress bar yang menarik"""
    return f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: between; margin-bottom: 0.5rem;">
            <span style="color: #ecf0f1; font-weight: 500;">{label}</span>
            <span style="color: #3498db; font-weight: 600;">{percentage}%</span>
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {percentage}%;"></div>
        </div>
    </div>
    """

def create_status_badge(status, text):
    """Membuat status badge dengan styling clean"""
    status_class = f"status-{status}"
    return f"""
    <span style="display: inline-flex; align-items: center; background: #f8f9fa; 
          padding: 0.5rem 1rem; border-radius: 20px; margin: 0.25rem;
          border: 1px solid #e9ecef;">
        <span class="status-indicator {status_class}"></span>
        <span style="color: #495057; font-weight: 500;">{text}</span>
    </span>
    """

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_datasets():
    """Memuat dataset dari file CSV"""
    try:
        # Load datasets
        profiles_df = pd.read_csv('./user_profiles.csv')
        jobs_df = pd.read_csv('./job_postings.csv')
        
        # Basic data cleaning
        profiles_df = profiles_df.dropna(subset=['name', 'position'])
        jobs_df = jobs_df.dropna(subset=['title', 'description'])
        
        return profiles_df, jobs_df
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure the CSV files are in the correct location.")
        return None, None

@st.cache_data
def extract_skills_from_text(text):
    """Ekstrak skill dari teks menggunakan pattern matching"""
    if pd.isna(text):
        return []
    
    # Daftar skills umum untuk matching
    common_skills = [
        'python', 'java', 'javascript', 'sql', 'machine learning', 'data science',
        'analytics', 'tensorflow', 'pytorch', 'react', 'node.js', 'docker',
        'kubernetes', 'aws', 'azure', 'gcp', 'mongodb', 'postgresql', 'mysql',
        'git', 'agile', 'scrum', 'project management', 'leadership', 'communication',
        'excel', 'powerbi', 'tableau', 'spark', 'hadoop', 'deep learning',
        'nlp', 'computer vision', 'devops', 'ci/cd', 'microservices'
    ]
    
    text_lower = text.lower()
    found_skills = []
    
    for skill in common_skills:
        if skill.lower() in text_lower:
            found_skills.append(skill.title())
    
    return found_skills

@st.cache_data
def process_data_for_matching(profiles_df, jobs_df):
    """Memproses data untuk job matching"""
    # Extract skills dari profil
    profiles_df['extracted_skills'] = profiles_df.apply(
        lambda row: extract_skills_from_text(
            str(row.get('about', '')) + ' ' + 
            str(row.get('position', '')) + ' ' + 
            str(row.get('experience', ''))
        ), axis=1
    )
    
    # Extract skills dari job descriptions
    jobs_df['extracted_skills'] = jobs_df['description'].apply(extract_skills_from_text)
    
    # Create skill vectors for similarity calculation
    all_skills = set()
    for skills in profiles_df['extracted_skills']:
        all_skills.update(skills)
    for skills in jobs_df['extracted_skills']:
        all_skills.update(skills)
    
    all_skills = list(all_skills)
    
    return profiles_df, jobs_df, all_skills

def calculate_job_similarity_matrix(jobs_df, all_skills):
    """Menghitung similarity matrix untuk jobs berdasarkan skills"""
    # Create binary skill matrix
    skill_matrix = []
    for _, job in jobs_df.iterrows():
        skill_vector = [1 if skill in job['extracted_skills'] else 0 for skill in all_skills]
        skill_matrix.append(skill_vector)
    
    skill_matrix = np.array(skill_matrix)
    
    # Calculate cosine similarity
    if skill_matrix.shape[1] > 0:
        similarity_matrix = cosine_similarity(skill_matrix)
    else:
        similarity_matrix = np.zeros((len(jobs_df), len(jobs_df)))
    
    return similarity_matrix

def simulate_gnn_predictions(profiles_df, jobs_df, all_skills, model_type="GNN"):
    """Simulasi prediksi GNN/GAT untuk job matching"""
    # Create compatibility scores
    compatibility_scores = []
    
    for _, profile in profiles_df.iterrows():
        profile_scores = []
        profile_skills = set(profile['extracted_skills'])
        
        for _, job in jobs_df.iterrows():
            job_skills = set(job['extracted_skills'])
            
            # Calculate base compatibility
            if len(profile_skills) > 0 and len(job_skills) > 0:
                skill_overlap = len(profile_skills.intersection(job_skills))
                total_skills = len(profile_skills.union(job_skills))
                base_score = skill_overlap / total_skills if total_skills > 0 else 0
            else:
                base_score = 0
            
            # Add model-specific adjustments
            if model_type == "GAT":
                # GAT considers attention weights - simulate with location matching
                location_boost = 0.1 if (
                    pd.notna(profile.get('city')) and 
                    pd.notna(job.get('location')) and
                    str(profile.get('city', '')).lower() in str(job.get('location', '')).lower()
                ) else 0
                final_score = min(base_score + location_boost + np.random.normal(0, 0.05), 1.0)
            else:
                # Standard GNN
                final_score = min(base_score + np.random.normal(0, 0.03), 1.0)
            
            final_score = max(0, final_score)  # Ensure non-negative
            profile_scores.append(final_score)
        
        compatibility_scores.append(profile_scores)
    
    return np.array(compatibility_scores)

def cluster_jobs_by_skills(jobs_df, n_clusters=5, detect_outliers=True):
    """Clustering jobs berdasarkan skills dengan deteksi outlier"""
    # Prepare text data for clustering
    job_texts = []
    for _, job in jobs_df.iterrows():
        skills_text = ' '.join(job['extracted_skills'])
        job_text = f"{job['title']} {skills_text} {job.get('formatted_work_type', '')}"
        job_texts.append(job_text)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    try:
        X = vectorizer.fit_transform(job_texts)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # TAMBAHAN: Deteksi outlier berdasarkan jarak ke centroid
        outliers = []
        if detect_outliers:
            # Hitung jarak setiap titik ke centroid cluster-nya
            distances = []
            for i, point in enumerate(X.toarray()):
                cluster_center = kmeans.cluster_centers_[clusters[i]]
                distance = np.linalg.norm(point - cluster_center)
                distances.append(distance)
            
            distances = np.array(distances)
            
            # Outlier detection menggunakan IQR method
            Q1 = np.percentile(distances, 25)
            Q3 = np.percentile(distances, 75)
            IQR = Q3 - Q1
            outlier_threshold = Q3 + 0.4 * IQR
            
            outliers = distances > outlier_threshold
            # outlier_threshold = np.percentile(distances, 95)
            # outliers = distances > outlier_threshold
        
        return clusters, vectorizer.get_feature_names_out(), outliers, distances if detect_outliers else None
    except:
        # Fallback if TF-IDF fails
        n_jobs = len(jobs_df)
        return np.random.randint(0, n_clusters, n_jobs), [], np.array([False] * n_jobs), None

def run_batch_clustering_analysis(jobs_df, all_skills, test_configs):
    """Menjalankan batch clustering analysis dengan berbagai konfigurasi"""
    results = []
    
    for i, config in enumerate(test_configs):
        n_clusters = config['n_clusters']
        min_similarity = config['min_similarity']
        model_type = config.get('model', 'GNN')
        
        try:
            # Clustering
            clusters, feature_names, outliers, distances = cluster_jobs_by_skills(
                jobs_df, n_clusters, detect_outliers=True
            )
            
            # Calculate metrics
            n_outliers = sum(outliers)
            outlier_percentage = (n_outliers / len(jobs_df)) * 100
            avg_distance = np.mean(distances) if distances is not None else 0
            
            # Calculate similarity matrix
            similarity_matrix = calculate_job_similarity_matrix(jobs_df, all_skills)
            
            # Network statistics
            G = nx.Graph()
            for idx, job in jobs_df.iterrows():
                G.add_node(idx, cluster=clusters[idx])
            
            # Add edges based on similarity
            edges_added = 0
            for i in range(len(jobs_df)):
                for j in range(i+1, len(jobs_df)):
                    if similarity_matrix[i][j] > min_similarity:
                        G.add_edge(i, j, weight=similarity_matrix[i][j])
                        edges_added += 1
            
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
            
            # Cluster distribution
            cluster_dist = pd.Series(clusters).value_counts()
            cluster_balance = cluster_dist.std() / cluster_dist.mean() if cluster_dist.mean() > 0 else 0
            
            # Store results
            result = {
                'test_id': i + 1,
                'n_clusters': n_clusters,
                'min_similarity': min_similarity,
                'model': model_type,
                'total_outliers': n_outliers,
                'outlier_percentage': outlier_percentage,
                'avg_distance_to_centroid': avg_distance,
                'network_nodes': G.number_of_nodes(),
                'network_edges': edges_added,
                'avg_connections': avg_degree,
                'cluster_balance_score': cluster_balance,
                'largest_cluster_size': cluster_dist.max(),
                'smallest_cluster_size': cluster_dist.min(),
                'clusters': clusters,
                'outliers': outliers,
                'distances': distances,
                'similarity_matrix': similarity_matrix
            }
            
            results.append(result)
            
        except Exception as e:
            # Handle errors gracefully
            result = {
                'test_id': i + 1,
                'n_clusters': n_clusters,
                'min_similarity': min_similarity,
                'model': model_type,
                'error': str(e),
                'total_outliers': 0,
                'outlier_percentage': 0,
                'avg_distance_to_centroid': 0,
                'network_nodes': 0,
                'network_edges': 0,
                'avg_connections': 0,
                'cluster_balance_score': 0,
                'largest_cluster_size': 0,
                'smallest_cluster_size': 0
            }
            results.append(result)
    
    return results

# Sidebar untuk navigasi
st.sidebar.title("🔗 Job Matching System")
st.sidebar.markdown("### Navigation")

page = st.sidebar.selectbox(
    "Choose Page",
    ["Dashboard Overview", "Job Matching", "Graph Analysis", "Model Comparison", "Data Explorer"]
)

# Model selection
st.sidebar.markdown("### Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose Model",
    ["GNN", "GAT"],
    help="GNN: Graph Neural Network, GAT: Graph Attention Network"
)

# Load data
profiles_df, jobs_df = load_datasets()

if profiles_df is not None and jobs_df is not None:
    # Process data
    profiles_df, jobs_df, all_skills = process_data_for_matching(profiles_df, jobs_df)
    
    # Main content based on selected page
    if page == "Dashboard Overview":
        # Enhanced Page Header
        st.markdown("""
        <div class="page-header">
            <div class="page-title">🔗 Job Matching Dashboard</div>
            <div class="page-subtitle">Powered by Graph Neural Networks & Graph Attention Networks</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Key Metrics dengan Icons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("Total Profiles", len(profiles_df), "👥"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Total Jobs", len(jobs_df), "💼"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Unique Skills", len(all_skills), "🎯"), unsafe_allow_html=True)
        with col4:
            active_jobs = len(jobs_df[jobs_df['expiry'] > datetime.now().timestamp()])
            st.markdown(create_metric_card("Active Jobs", active_jobs, "✅"), unsafe_allow_html=True)

        # System Health Indicators
        # st.markdown("### 📈 System Health")
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     # Calculate system health metrics
        #     data_completeness = (len(profiles_df.dropna()) / len(profiles_df)) * 100
        #     st.markdown(create_progress_bar(int(data_completeness), "Data Completeness"), unsafe_allow_html=True)
        # with col2:
        #     skill_coverage = (len([p for p in profiles_df['extracted_skills'] if len(p) > 0]) / len(profiles_df)) * 100
        #     st.markdown(create_progress_bar(int(skill_coverage), "Skill Coverage"), unsafe_allow_html=True)
        # with col3:
        #     matching_efficiency = 87  # Simulasi nilai
        #     st.markdown(create_progress_bar(matching_efficiency, "Matching Efficiency"), unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Job Distribution by Work Type")
            work_type_counts = jobs_df['formatted_work_type'].value_counts()
            fig_work_type = px.pie(
                values=work_type_counts.values,
                names=work_type_counts.index,
                title="Job Work Types"
            )
            st.plotly_chart(fig_work_type, use_container_width=True)
        
        with col2:
            st.subheader("Top Skills in Job Market")
            all_job_skills = []
            for skills in jobs_df['extracted_skills']:
                all_job_skills.extend(skills)
            
            if all_job_skills:
                skill_counts = pd.Series(all_job_skills).value_counts().head(10)
                fig_skills = px.bar(
                    x=skill_counts.values,
                    y=skill_counts.index,
                    orientation='h',
                    title="Most Demanded Skills"
                )
                fig_skills.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_skills, use_container_width=True)
        
        # Geographic distribution
        st.subheader("Geographic Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Profile Locations**")
            if 'country_code' in profiles_df.columns:
                country_counts = profiles_df['country_code'].value_counts().head(10)
                fig_countries = px.bar(
                    x=country_counts.index,
                    y=country_counts.values,
                    title="Profiles by Country"
                )
                st.plotly_chart(fig_countries, use_container_width=True)
        
        with col2:
            st.write("**Job Locations**")
            # Extract city from location
            jobs_df['job_city'] = jobs_df['location'].str.split(',').str[0]
            city_counts = jobs_df['job_city'].value_counts().head(10)
            fig_job_cities = px.bar(
                x=city_counts.index,
                y=city_counts.values,
                title="Jobs by City"
            )
            fig_job_cities.update_xaxes(tickangle=45)
            st.plotly_chart(fig_job_cities, use_container_width=True)
    
    elif page == "Job Matching":
        st.title("🎯 Job Matching System")
        st.write(f"Using **{selected_model}** model for predictions")
        
        # Profile selection
        st.subheader("Select Profile for Matching")
        
        # Profile search and filter
        col1, col2 = st.columns([2, 1])
        with col1:
            search_name = st.text_input("Search by name:", "")
        with col2:
            position_filter = st.selectbox(
                "Filter by position:",
                ["All"] + list(profiles_df['position'].dropna().unique()[:20])
            )
        
        # Filter profiles
        filtered_profiles = profiles_df.copy()
        if search_name:
            filtered_profiles = filtered_profiles[
                filtered_profiles['name'].str.contains(search_name, case=False, na=False)
            ]
        if position_filter != "All":
            filtered_profiles = filtered_profiles[
                filtered_profiles['position'] == position_filter
            ]
        
        if len(filtered_profiles) > 0:
            selected_profile_idx = st.selectbox(
                "Choose profile:",
                range(len(filtered_profiles)),
                format_func=lambda x: f"{filtered_profiles.iloc[x]['name']} - {filtered_profiles.iloc[x]['position']}"
            )
            
            selected_profile = filtered_profiles.iloc[selected_profile_idx]

            st.subheader("Additional Information")
            col1, col2 = st.columns(2)
            with col1:
                location = st.text_input("Location:", selected_profile.get('city', ''))
                expected_salary = st.text_input("Certifications:", selected_profile.get('certifications', ''))
                experience = st.text_input("Experience:", selected_profile.get('experience', ''))
            with col2:
                education = st.text_input("Education:", selected_profile.get('education', ''))
                languages = st.text_input("Languages:", selected_profile.get('languages', ''))
                about = st.text_area("About:", selected_profile.get('about', ''))
            
            # Display profile info
            st.markdown("### Selected Profile")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="profile-card">
                <h4>{selected_profile['name']}</h4>
                <p><strong>Position:</strong> {selected_profile['position']}</p>
                <p><strong>Company:</strong> {selected_profile.get('current_company:name', 'N/A')}</p>
                <p><strong>Location:</strong> {selected_profile.get('city', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.write("**Skills:**")
                profile_skills = selected_profile['extracted_skills']
                if profile_skills:
                    skills_text = ", ".join(profile_skills)
                    st.write(skills_text)
                else:
                    st.write("No skills extracted")
            
            # Model Performance Indicator
            st.markdown("### 🤖 Model Status")
            col1, col2, col3 = st.columns(3)
            with col1:
                if selected_model == "GNN":
                    st.markdown(create_status_badge("active", "GNN Active"), unsafe_allow_html=True)
                else:
                    st.markdown(create_status_badge("inactive", "GNN Inactive"), unsafe_allow_html=True)
            with col2:
                if selected_model == "GAT":
                    st.markdown(create_status_badge("active", "GAT Active"), unsafe_allow_html=True)
                else:
                    st.markdown(create_status_badge("inactive", "GAT Inactive"), unsafe_allow_html=True)
            with col3:
                st.markdown(create_status_badge("pending", "Training Ready"), unsafe_allow_html=True)

            # Generate predictions
            if st.button("Find Matching Jobs", type="primary"):
                with st.spinner(f'Finding matches using {selected_model}...'):
                    # Get profile index in original dataframe
                    profile_original_idx = profiles_df[profiles_df['id'] == selected_profile['id']].index[0]
                    
                    # Calculate compatibility scores
                    compatibility_scores = simulate_gnn_predictions(
                        profiles_df, jobs_df, all_skills, selected_model
                    )
                    
                    # Get scores for selected profile
                    profile_scores = compatibility_scores[profile_original_idx]
                    
                    # Create results dataframe
                    results_df = jobs_df.copy()
                    results_df['compatibility_score'] = profile_scores
                    results_df = results_df.sort_values('compatibility_score', ascending=False)
                    
                    # Display top matches
                    st.subheader(f"Top Job Matches (Using {selected_model})")
                    
                    top_matches = results_df.head(10)
                    
                    for idx, (_, job) in enumerate(top_matches.iterrows()):
                        score = job['compatibility_score']
                        
                        # Color coding based on score
                        if score >= 0.7:
                            score_color = "🟢"
                        elif score >= 0.5:
                            score_color = "🟡"
                        else:
                            score_color = "🔴"
                        
                        st.markdown(f"""
                        <div class="job-card">
                        <h4>{score_color} {job['title']} - {job.get('company_name', 'Unknown Company')}</h4>
                        <p><strong>Compatibility Score:</strong> {score:.3f}</p>
                        <p><strong>Location:</strong> {job['location']}</p>
                        <p><strong>Work Type:</strong> {job['formatted_work_type']}</p>
                        <p><strong>Required Skills:</strong> {', '.join(job['extracted_skills'][:5]) if job['extracted_skills'] else 'No specific skills listed'}</p>
                        <p><strong>Description:</strong> {job['description'][:200]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Score distribution
                    st.subheader("Score Distribution")
                    fig_scores = px.histogram(
                        profile_scores,
                        nbins=20,
                        title=f"Distribution of Compatibility Scores ({selected_model})"
                    )
                    fig_scores.update_xaxes(title="Compatibility Score")
                    fig_scores.update_yaxes(title="Number of Jobs")
                    st.plotly_chart(fig_scores, use_container_width=True)
        
        else:
            st.warning("No profiles found matching the search criteria.")
    
    # MODIFIKASI: Ganti bagian "Graph Analysis" page dengan ini
    elif page == "Graph Analysis":
        st.title("🕸️ Graph-based Job Clustering")
        
        # Pilihan mode analysis
        analysis_mode = st.radio(
            "Choose Analysis Mode",
            ["Single Test", "Batch Testing"],
            horizontal=True
        )
        
        if analysis_mode == "Single Test":
            # KODE SINGLE TEST YANG SUDAH ADA (tidak berubah)
            st.subheader("Job Clustering Analysis")
            
            # Clustering parameters
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("Number of Clusters", 3, 10, 5)
            with col2:
                min_similarity = st.slider("Minimum Similarity Threshold", 0.0, 1.0, 0.3)
            
            if st.button("Generate Job Clusters", type="primary"):
                with st.spinner("Clustering jobs based on skills..."):
                    # UBAH PEMANGGILAN FUNGSI INI
                    clusters, feature_names, outliers, distances = cluster_jobs_by_skills(jobs_df, n_clusters, detect_outliers=True)
                    jobs_df['cluster'] = clusters
                    jobs_df['is_outlier'] = outliers
                    if distances is not None:
                        jobs_df['distance_to_centroid'] = distances
                    
                    # Calculate similarity matrix
                    similarity_matrix = calculate_job_similarity_matrix(jobs_df, all_skills)

                    # TAMBAHAN: Display outlier summary
                    st.subheader("Outlier Analysis")
                    n_outliers = sum(outliers)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Outliers", n_outliers)
                    with col2:
                        outlier_percentage = (n_outliers / len(jobs_df)) * 100
                        st.metric("Outlier Percentage", f"{outlier_percentage:.1f}%")
                    with col3:
                        if distances is not None:
                            avg_distance = np.mean(distances)
                            st.metric("Avg Distance to Centroid", f"{avg_distance:.3f}")
                    
                    # TAMBAHAN: Show outlier jobs
                    if n_outliers > 0:
                        with st.expander(f"View Outlier Jobs ({n_outliers} jobs)"):
                            outlier_jobs = jobs_df[jobs_df['is_outlier'] == True][['title', 'company_name', 'extracted_skills', 'distance_to_centroid']].head(10)
                            st.dataframe(outlier_jobs)
                    
                    # Display cluster summary
                    st.subheader("Cluster Summary")
                    cluster_summary = jobs_df.groupby('cluster').agg({
                        'title': 'count',
                        'formatted_work_type': lambda x: x.mode()[0] if not x.empty else 'N/A',
                        'extracted_skills': lambda x: list(set([skill for skills in x for skill in skills]))
                    }).rename(columns={'title': 'job_count'})
                    
                    for cluster_id in range(n_clusters):
                        if cluster_id in cluster_summary.index:
                            cluster_info = cluster_summary.loc[cluster_id]
                            
                            with st.expander(f"Cluster {cluster_id} ({cluster_info['job_count']} jobs)"):
                                st.write(f"**Dominant Work Type:** {cluster_info['formatted_work_type']}")
                                st.write(f"**Common Skills:** {', '.join(cluster_info['extracted_skills'][:10])}")
                                
                                # Show sample jobs from cluster
                                cluster_jobs = jobs_df[jobs_df['cluster'] == cluster_id].head(5)
                                st.write("**Sample Jobs:**")
                                for _, job in cluster_jobs.iterrows():
                                    st.write(f"• {job['title']} at {job.get('company_name', 'Unknown')}")
                    
                    # BAGIAN YANG DIUBAH: Cluster Visualization dengan PCA
                    st.subheader("Cluster Visualization")
                    
                    # Prepare feature matrix for PCA
                    try:
                        # Create TF-IDF matrix for jobs
                        job_texts = []
                        for _, job in jobs_df.iterrows():
                            skills_text = ' '.join(job['extracted_skills'])
                            job_text = f"{job['title']} {skills_text} {job.get('formatted_work_type', '')}"
                            job_texts.append(job_text)
                        
                        # TF-IDF Vectorization
                        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                        X_tfidf = vectorizer.fit_transform(job_texts)
                        
                        # Convert to dense array
                        X_dense = X_tfidf.toarray()
                        
                        # Standardize features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_dense)
                        
                        # Apply PCA
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        # Create DataFrame for visualization
                        pca_df = pd.DataFrame({
                            'PC1': X_pca[:, 0],
                            'PC2': X_pca[:, 1],
                            'Cluster': jobs_df['cluster'].values,
                            'Job_Title': jobs_df['title'].values,
                            'Company': jobs_df.get('company_name', 'Unknown').values,
                            'Work_Type': jobs_df['formatted_work_type'].values,
                            'Is_Outlier': jobs_df['is_outlier'].values,  # TAMBAHAN
                            'Distance': jobs_df.get('distance_to_centroid', 0).values  # TAMBAHAN
                        })
                        
                        # Create scatter plot
                        fig_pca = px.scatter(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            color='Cluster',
                            symbol='Is_Outlier',  # TAMBAHAN: berbeda symbol untuk outlier
                            size='Distance',      # TAMBAHAN: size berdasarkan jarak ke centroid
                            hover_data=['Job_Title', 'Company', 'Work_Type', 'Distance'],
                            title=f"Job Clusters with Outliers (PCA) - {n_clusters} Clusters",
                            labels={
                                'PC1': f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)',
                                'PC2': f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)'
                            },
                            color_continuous_scale='viridis',
                            symbol_map={True: 'diamond', False: 'circle'}  # TAMBAHAN: diamond untuk outlier
                        )
                        
                        # Update layout for better visualization
                        fig_pca.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='white')))
                        fig_pca.update_layout(
                            width=800,
                            height=600,
                            showlegend=True,
                            legend=dict(
                                title="Legend",
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.01
                            )
                        )
                        
                        st.plotly_chart(fig_pca, use_container_width=True)
                        
                        # PCA Analysis Information
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Variance Explained", f"{sum(pca.explained_variance_ratio_):.2%}")
                        with col2:
                            st.metric("PC1 Variance", f"{pca.explained_variance_ratio_[0]:.2%}")
                        with col3:
                            st.metric("PC2 Variance", f"{pca.explained_variance_ratio_[1]:.2%}")
                        
                        # Feature importance in PCA
                        st.subheader("PCA Component Analysis")
                        
                        # Get feature names
                        feature_names = vectorizer.get_feature_names_out()
                        
                        # PC1 top features
                        pc1_features = pd.DataFrame({
                            'Feature': feature_names,
                            'PC1_Weight': pca.components_[0]
                        }).sort_values('PC1_Weight', key=abs, ascending=False).head(10)
                        
                        # PC2 top features
                        pc2_features = pd.DataFrame({
                            'Feature': feature_names,
                            'PC2_Weight': pca.components_[1]
                        }).sort_values('PC2_Weight', key=abs, ascending=False).head(10)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Top Features in PC1**")
                            fig_pc1 = px.bar(
                                pc1_features,
                                x='PC1_Weight',
                                y='Feature',
                                orientation='h',
                                title="Most Important Features in PC1"
                            )
                            fig_pc1.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_pc1, use_container_width=True)
                        
                        with col2:
                            st.write("**Top Features in PC2**")
                            fig_pc2 = px.bar(
                                pc2_features,
                                x='PC2_Weight',
                                y='Feature',
                                orientation='h',
                                title="Most Important Features in PC2"
                            )
                            fig_pc2.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_pc2, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error in PCA visualization: {str(e)}")
                        st.info("Falling back to cluster distribution chart...")
                        
                        # Fallback: Original cluster distribution chart
                        cluster_dist = jobs_df['cluster'].value_counts().sort_index()
                        fig_clusters = px.bar(
                            x=cluster_dist.index,
                            y=cluster_dist.values,
                            title="Jobs per Cluster",
                            labels={'x': 'Cluster ID', 'y': 'Number of Jobs'}
                        )
                        st.plotly_chart(fig_clusters, use_container_width=True)
                    
                    # Network statistics (tetap dipertahankan)
                    st.subheader("Network Statistics")
                    
                    # Create network graph
                    G = nx.Graph()
                    
                    # Add nodes (jobs)
                    for idx, job in jobs_df.iterrows():
                        company = str(job.get('company_name', 'Unknown'))[:20]
                        G.add_node(idx, 
                                title=job['title'][:30], 
                                cluster=job['cluster'],
                                company=company)
                    
                    # Add edges based on similarity
                    for i in range(len(jobs_df)):
                        for j in range(i+1, len(jobs_df)):
                            if similarity_matrix[i][j] > min_similarity:
                                G.add_edge(i, j, weight=similarity_matrix[i][j])
                    
                    # Network statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nodes (Jobs)", G.number_of_nodes())
                    with col2:
                        st.metric("Edges (Connections)", G.number_of_edges())
                    with col3:
                        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
                        st.metric("Avg Connections", f"{avg_degree:.2f}")
                    
                    # Cluster distribution chart
                    cluster_dist = jobs_df['cluster'].value_counts().sort_index()
                    fig_clusters = px.bar(
                        x=cluster_dist.index,
                        y=cluster_dist.values,
                        title="Jobs per Cluster",
                        labels={'x': 'Cluster ID', 'y': 'Number of Jobs'}
                    )
                    st.plotly_chart(fig_clusters, use_container_width=True)

                    # TAMBAHAN: Outlier distribution chart
                    st.subheader("Outlier Distribution by Cluster")
                    outlier_by_cluster = jobs_df.groupby('cluster')['is_outlier'].agg(['sum', 'count']).reset_index()
                    outlier_by_cluster['outlier_percentage'] = (outlier_by_cluster['sum'] / outlier_by_cluster['count']) * 100
                    
                    fig_outlier_dist = px.bar(
                        outlier_by_cluster,
                        x='cluster',
                        y='outlier_percentage',
                        title="Percentage of Outliers by Cluster",
                        labels={'cluster': 'Cluster ID', 'outlier_percentage': 'Outlier Percentage (%)'}
                    )
                    st.plotly_chart(fig_outlier_dist, use_container_width=True)
                    
                    # TAMBAHAN: Distance distribution
                    if distances is not None:
                        st.subheader("Distance to Centroid Distribution")
                        fig_distance = px.histogram(
                            jobs_df,
                            x='distance_to_centroid',
                            color='is_outlier',
                            title="Distribution of Distances to Cluster Centroids",
                            labels={'distance_to_centroid': 'Distance to Centroid', 'count': 'Number of Jobs'},
                            nbins=30
                        )
                        st.plotly_chart(fig_distance, use_container_width=True)
                
        else:  # Batch Testing Mode
            st.subheader("🚀 Batch Testing Configuration")
            
            # Batch configuration
            col1, col2 = st.columns(2)
            
            with col1:
                num_tests = st.number_input("Number of Tests", min_value=2, max_value=10, value=3)
                st.info(f"Configure {num_tests} different test scenarios")
            
            with col2:
                default_model = st.selectbox("Default Model for All Tests", ["GNN", "GAT"])
                allow_model_selection = st.checkbox("Allow different models per test")
            
            # Dynamic test configuration
            st.subheader("Test Configurations")
            test_configs = []
            
            # Create tabs for each test
            tab_labels = [f"Test {i+1}" for i in range(num_tests)]
            tabs = st.tabs(tab_labels)
            
            for i, tab in enumerate(tabs):
                with tab:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        clusters = st.slider(
                            f"Number of Clusters", 
                            3, 10, 5, 
                            key=f"clusters_{i}"
                        )
                    
                    with col2:
                        similarity = st.slider(
                            f"Min Similarity Threshold", 
                            0.0, 1.0, 0.3, 
                            key=f"similarity_{i}"
                        )
                    
                    with col3:
                        if allow_model_selection:
                            model = st.selectbox(
                                "Model", 
                                ["GNN", "GAT"], 
                                index=0 if default_model == "GNN" else 1,
                                key=f"model_{i}"
                            )
                        else:
                            model = default_model
                            st.write(f"**Model:** {model}")
                    
                    test_configs.append({
                        'n_clusters': clusters,
                        'min_similarity': similarity,
                        'model': model
                    })
            
            # Run batch analysis
            if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Running batch clustering analysis..."):
                    # Run batch analysis
                    batch_results = run_batch_clustering_analysis(jobs_df, all_skills, test_configs)
                    
                    # Update progress
                    progress_bar.progress(100)
                    status_text.success(f"✅ Completed {len(batch_results)} tests!")
                
                # Display results comparison
                st.subheader("📊 Batch Results Comparison")
                
                # Results summary table
                results_df = pd.DataFrame([
                    {
                        'Test ID': r['test_id'],
                        'Clusters': r['n_clusters'],
                        'Min Similarity': r['min_similarity'],
                        'Model': r['model'],
                        'Outliers': r['total_outliers'],
                        'Outlier %': f"{r['outlier_percentage']:.1f}%",
                        'Avg Distance': f"{r['avg_distance_to_centroid']:.3f}",
                        'Network Edges': r['network_edges'],
                        'Avg Connections': f"{r['avg_connections']:.2f}",
                        'Cluster Balance': f"{r['cluster_balance_score']:.3f}",
                        'Largest Cluster': r['largest_cluster_size'],
                        'Smallest Cluster': r['smallest_cluster_size']
                    }
                    for r in batch_results if 'error' not in r
                ])
                
                if not results_df.empty:
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Comparison charts
                    st.subheader("📈 Performance Comparison Charts")
                    
                    # Create comparison metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Outlier comparison
                        fig_outliers = px.bar(
                            results_df,
                            x='Test ID',
                            y='Outliers',
                            color='Model',
                            title="Outliers Detected by Test",
                            text='Outliers'
                        )
                        fig_outliers.update_traces(textposition='outside')
                        st.plotly_chart(fig_outliers, use_container_width=True)
                    
                    with col2:
                        # Network connectivity comparison
                        fig_connections = px.bar(
                            results_df,
                            x='Test ID',
                            y='Network Edges',
                            color='Model',
                            title="Network Connections by Test",
                            text='Network Edges'
                        )
                        fig_connections.update_traces(textposition='outside')
                        st.plotly_chart(fig_connections, use_container_width=True)
                    
                    # Cluster balance comparison
                    fig_balance = px.line(
                        results_df,
                        x='Test ID',
                        y='Cluster Balance',
                        color='Model',
                        title="Cluster Balance Score (Lower = More Balanced)",
                        markers=True
                    )
                    st.plotly_chart(fig_balance, use_container_width=True)
                    
                    # Best performing test
                    st.subheader("🏆 Best Performing Tests")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Lowest outlier percentage
                        best_outlier = results_df.loc[results_df['Outliers'].idxmin()]
                        st.metric(
                            "Lowest Outliers",
                            f"Test {best_outlier['Test ID']}",
                            f"{best_outlier['Outliers']} outliers"
                        )
                    
                    with col2:
                        # Best cluster balance
                        best_balance = results_df.loc[results_df['Cluster Balance'].idxmin()]
                        st.metric(
                            "Best Balance",
                            f"Test {best_balance['Test ID']}",
                            f"{float(best_balance['Cluster Balance']):.3f} score"
                        )
                    
                    with col3:
                        # Highest connectivity
                        best_connectivity = results_df.loc[results_df['Network Edges'].idxmax()]
                        st.metric(
                            "Best Connectivity",
                            f"Test {best_connectivity['Test ID']}",
                            f"{best_connectivity['Network Edges']} edges"
                        )
                    
                    # Detailed results for each test
                    st.subheader("🔍 Detailed Test Results")
                    
                    selected_test = st.selectbox(
                        "Select Test to View Details",
                        options=range(1, len(batch_results) + 1),
                        format_func=lambda x: f"Test {x} (Clusters: {test_configs[x-1]['n_clusters']}, "
                                            f"Similarity: {test_configs[x-1]['min_similarity']}, "
                                            f"Model: {test_configs[x-1]['model']})"
                    )
                    
                    if selected_test:
                        selected_result = batch_results[selected_test - 1]
                        
                        if 'error' not in selected_result:
                            # Create temporary dataframe with results for visualization
                            temp_jobs_df = jobs_df.copy()
                            temp_jobs_df['cluster'] = selected_result['clusters']
                            temp_jobs_df['is_outlier'] = selected_result['outliers']
                            if selected_result['distances'] is not None:
                                temp_jobs_df['distance_to_centroid'] = selected_result['distances']
                            
                            # Show cluster distribution for selected test
                            cluster_dist = pd.Series(selected_result['clusters']).value_counts().sort_index()
                            fig_selected = px.bar(
                                x=cluster_dist.index,
                                y=cluster_dist.values,
                                title=f"Test {selected_test} - Jobs per Cluster",
                                labels={'x': 'Cluster ID', 'y': 'Number of Jobs'}
                            )
                            st.plotly_chart(fig_selected, use_container_width=True)
                            
                            # Show outlier information
                            if selected_result['total_outliers'] > 0:
                                outlier_jobs = temp_jobs_df[temp_jobs_df['is_outlier'] == True][
                                    ['title', 'company_name', 'extracted_skills']
                                ].head(5)
                                
                                st.write(f"**Sample Outlier Jobs ({selected_result['total_outliers']} total):**")
                                st.dataframe(outlier_jobs, use_container_width=True)
                        
                        else:
                            st.error(f"Test {selected_test} failed with error: {selected_result['error']}")
                
                else:
                    st.error("All tests failed. Please check your configurations and try again.")
    
    elif page == "Model Comparison":
        st.title("⚖️ GNN vs GAT Model Comparison")
        
        st.subheader("Model Performance Analysis")
        
        if st.button("Run Model Comparison", type="primary"):
            with st.spinner("Comparing GNN and GAT models..."):
                # Generate predictions for both models
                gnn_scores = simulate_gnn_predictions(profiles_df, jobs_df, all_skills, "GNN")
                gat_scores = simulate_gnn_predictions(profiles_df, jobs_df, all_skills, "GAT")
                
                # Calculate metrics
                gnn_avg_scores = np.mean(gnn_scores, axis=1)
                gat_avg_scores = np.mean(gat_scores, axis=1)
                
                # Display comparison metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("GNN Avg Score", f"{np.mean(gnn_avg_scores):.3f}")
                    st.metric("GNN Std Dev", f"{np.std(gnn_avg_scores):.3f}")
                
                with col2:
                    st.metric("GAT Avg Score", f"{np.mean(gat_avg_scores):.3f}")
                    st.metric("GAT Std Dev", f"{np.std(gat_avg_scores):.3f}")
                
                # Score distribution comparison
                fig_comparison = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("GNN Score Distribution", "GAT Score Distribution")
                )
                
                fig_comparison.add_trace(
                    go.Histogram(x=gnn_avg_scores, name="GNN", nbinsx=20),
                    row=1, col=1
                )
                
                fig_comparison.add_trace(
                    go.Histogram(x=gat_avg_scores, name="GAT", nbinsx=20),
                    row=1, col=2
                )
                
                fig_comparison.update_layout(
                    title_text="Model Score Distribution Comparison",
                    showlegend=False
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Direct comparison
                comparison_df = pd.DataFrame({
                    'Profile_ID': range(len(profiles_df)),
                    'Profile_Name': profiles_df['name'].values,
                    'GNN_Score': gnn_avg_scores,
                    'GAT_Score': gat_avg_scores,
                    'Difference': gat_avg_scores - gnn_avg_scores
                })
                
                st.subheader("Detailed Comparison")
                st.dataframe(comparison_df.head(20))
                
                # Performance insights
                st.subheader("Model Insights")
                
                better_gnn = len(comparison_df[comparison_df['Difference'] < 0])
                better_gat = len(comparison_df[comparison_df['Difference'] > 0])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**GNN performs better:** {better_gnn} profiles ({better_gnn/len(comparison_df)*100:.1f}%)")
                with col2:
                    st.info(f"**GAT performs better:** {better_gat} profiles ({better_gat/len(comparison_df)*100:.1f}%)")
    
    elif page == "Data Explorer":
        st.title("🔍 Data Explorer")
        
        # Dataset selector
        dataset_choice = st.selectbox("Choose Dataset", ["Profiles", "Jobs"])
        
        if dataset_choice == "Profiles":
            st.subheader("LinkedIn Profiles Dataset")
            
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Profiles", len(profiles_df))
            with col2:
                complete_profiles = len(profiles_df.dropna(subset=['name', 'position', 'about']))
                st.metric("Complete Profiles", complete_profiles)
            with col3:
                with_skills = len(profiles_df[profiles_df['extracted_skills'].apply(len) > 0])
                st.metric("Profiles with Skills", with_skills)
            
            # Display sample data
            st.subheader("Sample Data")
            display_columns = ['name', 'position', 'current_company:name', 'city', 'extracted_skills']
            available_columns = [col for col in display_columns if col in profiles_df.columns]
            st.dataframe(profiles_df[available_columns].head(10))
            
            # Skills analysis
            st.subheader("Skills Analysis")
            all_profile_skills = []
            for skills in profiles_df['extracted_skills']:
                all_profile_skills.extend(skills)
            
            if all_profile_skills:
                skill_counts = pd.Series(all_profile_skills).value_counts().head(15)
                fig_skills = px.bar(
                    x=skill_counts.values,
                    y=skill_counts.index,
                    orientation='h',
                    title="Most Common Skills in Profiles"
                )
                fig_skills.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_skills, use_container_width=True)
        
        else:  # Jobs dataset
            st.subheader("LinkedIn Job Postings Dataset")
            
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Jobs", len(jobs_df))
            with col2:
                with_salary = len(jobs_df.dropna(subset=['max_salary']))
                st.metric("Jobs with Salary Info", with_salary)
            with col3:
                remote_jobs = len(jobs_df[jobs_df['remote_allowed'] == 1])
                st.metric("Remote Jobs", remote_jobs)
            
            # Display sample data
            st.subheader("Sample Data")
            display_columns = ['title', 'company_name', 'location', 'formatted_work_type', 'extracted_skills']
            available_columns = [col for col in display_columns if col in jobs_df.columns]
            st.dataframe(jobs_df[available_columns].head(10))
            
            # Salary analysis
            if 'max_salary' in jobs_df.columns and jobs_df['max_salary'].notna().sum() > 0:
                st.subheader("Salary Distribution")
                salary_data = jobs_df.dropna(subset=['max_salary'])
                fig_salary = px.histogram(
                    salary_data,
                    x='max_salary',
                    title="Maximum Salary Distribution",
                    nbins=30
                )
                st.plotly_chart(fig_salary, use_container_width=True)

else:
    st.error("Please ensure the dataset files are available in the correct location.")
    st.info("Expected files: 'linkedinuserprofiles.csv' and 'postings.csv'")

# === TAMBAHKAN ANIMATED LOADING SPINNER ===
def show_loading_animation():
    """Menampilkan animasi loading yang menarik"""
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100px;">
        <div style="animation: spin 1s linear infinite; font-size: 3rem;">🔄</div>
        <div style="margin-left: 1rem; color: #3498db; font-weight: 600;">
            Processing with AI Magic...
        </div>
    </div>
    <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """, unsafe_allow_html=True)

# === ENHANCED FOOTER ===
# Ganti bagian footer dengan ini:
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 2rem;">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">🚀</span>
            <span style="color: #2c3e50; font-weight: 500;">Job Matching System v2.0</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">🧠</span>
            <span style="color: #2c3e50; font-weight: 500;">Powered by Neural Networks</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">⚡</span>
            <span style="color: #2c3e50; font-weight: 500;">Real-time Processing</span>
        </div>
    </div>
    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
        <span style="color: #bdc3c7; font-size: 0.9rem;">
            © 2024 Advanced Job Matching System | Built with ❤️ and AI
        </span>
    </div>
</div>
""", unsafe_allow_html=True)