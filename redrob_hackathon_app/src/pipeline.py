import gzip
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def run_ranking_pipeline(jd_text, jsonl_gz_path="candidates.jsonl.gz"):
    # 1. Load the Embedding Model (Optimized for text matching)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 2. Encode the Job Description
    jd_vector = model.encode([jd_text], convert_to_numpy=True)
    
    # 3. Stream & Process 100,000 Candidates from .jsonl.gz
    candidate_records = []
    text_profiles = []
    
    with gzip.open(jsonl_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            # Extract fields safely
            c_id = data.get("candidate_id")
            skills = data.get("skills", "")
            experience = data.get("experience", "")
            projects = data.get("projects", "")
            work_history = data.get("work_history", "")
            signals = data.get("redrob_signals", {})
            
            # Build the Profile Text exactly like your example
            profile_text = f"{experience}. Built {projects}. {skills}. Worked at {work_history}. Active on platform: {signals.get('active', False)}."
            text_profiles.append(profile_text)
            
            # Store structured data for downstream Hybrid Feature Scoring
            candidate_records.append({
                "candidate_id": c_id,
                "profile_text": profile_text,
                "tech_score": data.get("technical_skills_match_score", 0.0), # scale 0-1
                "exp_years_score": min(data.get("years_experience", 0) / 10.0, 1.0), # normalize to 0-1
                "project_score": data.get("project_quality_score", 0.0),
                "behavior_score": signals.get("engagement_score", 0.0)
            })
            
    # Convert records to a DataFrame
    df = pd.DataFrame(candidate_records)
    
    # 4. Generate Embeddings & Run FAISS Semantic Matching
    candidate_vectors = model.encode(text_profiles, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    
    # Standardize vectors for Cosine Similarity
    faiss.normalize_L2(jd_vector)
    faiss.normalize_L2(candidate_vectors)
    
    # Build a fast FAISS Index
    dimension = candidate_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(candidate_vectors)
    
    # Search the entire pool of 100k
    similarities, indices = index.search(jd_vector, len(df))
    
    # Map semantic scores back to the dataframe
    df["semantic_similarity"] = 0.0
    for idx, sim in zip(indices[0], similarities[0]):
        df.at[idx, "semantic_similarity"] = float(sim)
        
    # 5. Hybrid Scoring Weights (45%, 20%, 15%, 15%, 5%)
    df["final_score"] = (
        (df["semantic_similarity"] * 0.45) +
        (df["tech_score"] * 0.20) +
        (df["exp_years_score"] * 0.15) +
        (df["project_score"] * 0.15) +
        (df["behavior_score"] * 0.05)
    )
    
    # 6. Sort and Extract Top 100
    df_top_100 = df.sort_values(by="final_score", ascending=False).head(100).reset_index(drop=True)
    df_top_100["rank"] = df_top_100.index + 1
    
    # 7. Generate Explanatory Reasons for the CSV
    def generate_reason(row):
        return f"Strong match with a semantic alignment of {row['semantic_similarity']:.2f}. Features robust technical qualifications (Tech: {row['tech_score']:.2f}) and proven project context."
        
    df_top_100["reason"] = df_top_100.apply(generate_reason, axis=1)
    
    # Export cleanly structured CSV
    output_df = df_top_100[["rank", "candidate_id", "reason"]]
    output_df.to_csv("top_100_candidates.csv", index=False)
    
    return "top_100_candidates.csv"