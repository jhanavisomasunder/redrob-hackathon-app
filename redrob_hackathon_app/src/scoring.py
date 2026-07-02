import re
import numpy as np
from typing import Dict, Any, List

class MultiHeuristicScoringEngine:
    """Calculates weighted profiles and executes anti-gaming/trap evaluation."""
    
    def __init__(self, target_weights: Dict[str, float] = None):
        self.weights = target_weights or {
            "experience": 0.35,
            "skills": 0.25,
            "semantic": 0.20,
            "behavioral": 0.10,
            "availability": 0.05,
            "location": 0.05
        }

    def detect_and_penalize_frauds(self, feature_map: Dict[str, Any], text_query: str) -> float:
        """Evaluates profile patterns for bad behavior, honeypots, and keyword stuffing."""
        penalty = 1.0
        doc = feature_map["search_document"].lower()
        
        # 1. Keyword Stuffing Check: Check for high repetition of terms
        words = re.findall(r'\w+', doc)
        if len(words) > 0:
            for word in set(words):
                if len(word) > 3 and words.count(word) > 15:
                    penalty *= 0.60  # Slashing 40% value
                    
        # 2. Suspicious Work Experience Check
        if feature_map["total_experience_months"] > 600: # Over 50 Years
            penalty *= 0.50
            
        # 3. Behavioral Dormancy Penalty
        if feature_map["behavioral"]["recruiter_response_rate"] < 0.10:
            penalty *= 0.85
            
        return penalty

    def compute_scores(self, candidate: Dict[str, Any], semantic_score: float, jd_keywords: List[str], target_loc: str) -> Dict[str, Any]:
        """Calculates granular scoring subcomponents matching the structural matrices required."""
        # Component 1: Skill Score
        cand_skills = [s.lower() for s in candidate["skills"]]
        matched_skills = sum(1 for kw in jd_keywords if kw.lower() in cand_skills)
        skill_score = (matched_skills / max(len(jd_keywords), 1)) * 100

        # Component 2: Experience Score (Benchmark against sweet spot, eg. 5 years = 60 months)
        exp_months = candidate["total_experience_months"]
        exp_score = min((exp_months / 60.0) * 100, 100)

        # Component 3: Behavioral Score
        b = candidate["behavioral"]
        behavioral_score = (
            (b["open_to_work"] * 30) +
            (b["recruiter_response_rate"] * 25) +
            ((b["github_score"] / 100.0) * 25) +
            (b["interview_rate"] * 20)
        )

        # Component 4: Availability / Notice Period Score
        avail_score = max(0, 100 - (b["notice_period"] * 0.8))

        # Component 5: Location Match Score
        loc_score = 100 if (b["relocate"] or target_loc.lower() in b["location"].lower()) else 40

        # Map semantic similarity score bounded dynamically 
        norm_semantic = max(0.0, min(100.0, float(semantic_score) * 100))

        # Aggregate weighted computation
        raw_total = (
            (exp_score * self.weights["experience"]) +
            (skill_score * self.weights["skills"]) +
            (norm_semantic * self.weights["semantic"]) +
            (behavioral_score * self.weights["behavioral"]) +
            (avail_score * self.weights["availability"]) +
            (loc_score * self.weights["location"])
        )

        penalty_multiplier = self.detect_and_penalize_frauds(candidate, " ".join(jd_keywords))
        final_score = round(raw_total * penalty_multiplier, 2)

        return {
            "overall_score": final_score,
            "breakdown": {
                "Experience Match": round(exp_score, 1),
                "Skill Match": round(skill_score, 1),
                "Semantic Similarity": round(norm_semantic, 1),
                "Behavioral Signals": round(behavioral_score, 1),
                "Availability": round(avail_score, 1),
                "Location Match": round(loc_score, 1)
            },
            "is_penalized": penalty_multiplier < 1.0
        }