import json
import gzip
import logging
from typing import Generator, Dict, Any

logger = logging.getLogger(__name__)

class CandidateDataLoader:
    """Efficient streaming data loader for processing massive candidate datasets with minimal footprint."""
    
    @staticmethod
    def stream_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
        """Streams records line-by-line supporting raw and compressed formats."""
        open_func = gzip.open if file_path.endswith('.gz') else open
        mode = 'rt' if file_path.endswith('.gz') else 'r'
        
        try:
            with open_func(file_path, mode, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        except Exception as e:
            logger.error(f"Error streaming data file {file_path}: {str(e)}")
            raise e

    @staticmethod
    def extract_features(candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically parses and builds flat indexes optimized for semantic retrieval and heuristic scoring."""
        # Clean skills
        skills_list = candidate.get('skills', [])
        skills_text = ", ".join(skills_list) if isinstance(skills_list, list) else str(skills_list)
        
        # Build compact history representation
        history = []
        for exp in candidate.get('work_history', []):
            history.append(f"{exp.get('role', '')} at {exp.get('company', '')} ({exp.get('duration_months', 0)} mos)")
        history_text = " | ".join(history)

        # Merge structural information for semantic search
        search_document = f"Skills: {skills_text}. Experience: {history_text}. Bio: {candidate.get('bio', '')}"
        
        return {
            "id": candidate.get("candidate_id"),
            "name": candidate.get("name", "Anonymous Candidate"),
            "search_document": search_document,
            "skills": skills_list,
            "total_experience_months": sum(int(exp.get('duration_months', 0)) for exp in candidate.get('work_history', [])),
            "behavioral": {
                "open_to_work": bool(candidate.get("open_to_work_flag", False)),
                "recruiter_response_rate": float(candidate.get("recruiter_response_rate", 0.0)),
                "github_score": float(candidate.get("github_activity_score", 0.0)),
                "interview_rate": float(candidate.get("interview_completion_rate", 0.0)),
                "notice_period": int(candidate.get("notice_period_days", 90)),
                "relocate": bool(candidate.get("willing_to_relocate", False)),
                "location": candidate.get("current_location", "")
            },
            "raw_profile": candidate
        }