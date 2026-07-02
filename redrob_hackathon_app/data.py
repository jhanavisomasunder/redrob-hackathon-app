import gzip
import json
import random

# Core skills we can mix and match to simulate a real talent pool
skills_pool = [
    "Python, FastAPI, FAISS, Vector Databases, LLMs, NLP",
    "JavaScript, React, Node.js, Express, MongoDB, Tailwind CSS",
    "Python, Django, PostgreSQL, Docker, AWS, Redis",
    "Java, Spring Boot, Kubernetes, MySQL, Microservices",
    "Python, PyTorch, Scikit-Learn, MLOps, Computer Vision"
]

company_pool = ["TechCorp", "Fintech Startup", "SaaS Scaleup", "AI Labs", "Web Agency"]

print("Generating mock talent pool matrix...")

# Let's generate 500 clean mock candidates to safely test your Top 100 script logic
with gzip.open("candidates.jsonl.gz", "wt", encoding="utf-8") as f:
    for i in range(1, 501):
        c_id = str(10000 + i)
        years_exp = random.randint(1, 12)
        
        # Make the first 5 candidates perfect matches for a Senior AI Engineer
        if i <= 5:
            skills = "Python, FastAPI, FAISS, Vector Indexes, Sentence-Transformers, LLMs"
            exp = f"{years_exp} years as a Senior ML/AI Software Engineer"
            projects = "Production scale semantic search engine and hybrid ranking vector pipeline"
            tech_score = random.uniform(0.90, 0.99)
            proj_score = random.uniform(0.88, 0.98)
        else:
            skills = random.choice(skills_pool)
            exp = f"{years_exp} years of industry experience across software engineering stacks"
            projects = "Contributed to core application backends and generic internal microservices"
            tech_score = random.uniform(0.10, 0.85)
            proj_score = random.uniform(0.20, 0.85)

        candidate = {
            "candidate_id": c_id,
            "skills": skills,
            "experience": exp,
            "projects": projects,
            "work_history": random.choice(company_pool),
            "technical_skills_match_score": tech_score,
            "years_experience": years_exp,
            "project_quality_score": proj_score,
            "redrob_signals": {
                "engagement_score": random.uniform(0.30, 0.95)
            }
        }
        f.write(json.dumps(candidate) + "\n")

print("Created valid file: candidates.jsonl.gz")