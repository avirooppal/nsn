import datetime

class RecencyScorer:
    """
    Evaluates trust based on how recent the memory or observation is.
    Recent information is generally trusted more than very old information.
    """
    def score(self, timestamp_iso: str) -> float:
        try:
            # Handle standard ISO formats, including Z for UTC
            clean_ts = timestamp_iso.replace('Z', '+00:00')
            dt = datetime.datetime.fromisoformat(clean_ts)
            
            # Ensure aware datetime for comparison if dt is aware
            if dt.tzinfo is not None:
                now = datetime.datetime.now(datetime.timezone.utc)
            else:
                now = datetime.datetime.utcnow()
                
            delta = now - dt
            days_old = delta.total_seconds() / (24 * 3600)
            
            if days_old < 0: # Future timestamp?
                return 0.5
            elif days_old < 1:
                return 1.0
            elif days_old < 7:
                return 0.8
            elif days_old < 30:
                return 0.6
            elif days_old < 365:
                return 0.4
            else:
                return 0.2
        except Exception:
            return 0.5
