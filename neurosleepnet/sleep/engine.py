import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SleepEngine")

class SleepEngine:
    """
    Coordinates offline memory consolidation.
    """
    def __init__(self, memory):
        self.memory = memory
        self.is_sleeping = False

    def trigger(self):
        """
        Initiates the sleep cycle.
        """
        if self.is_sleeping:
            logger.warning("Sleep cycle is already in progress.")
            return False
            
        self.is_sleeping = True
        logger.info("Initiating sleep cycle. Offline consolidation started...")
        
        # Sleep phases (NREM, REM) will be orchestrated here in future tasks.
        
        # Simulate waking up
        self.is_sleeping = False
        logger.info("Sleep cycle complete.")
        
        return True
