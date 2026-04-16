import asyncio
import logging
import sys
import time
from .connector import FlashscoreWSClient
from .models import MatchUpdate

# Optional uvloop for maximum performance
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

async def handle_update(update: MatchUpdate):
    """
    Final output layer.
    Calculates processing latency from packet arrival to structured object.
    """
    now = time.time()
    dt_ms = (now - update.timestamp) * 1000
    
    # We output a structured summary for the user
    # In a real betting system, this would feed directly into the Markov engine.
    sys.stdout.write(
        f"\r[LATENCY: {dt_ms:5.2f}ms] {update.match_id} | "
        f"{update.p1.name} {update.p1.score} - {update.p2.score} {update.p2.name} "
        f"({update.status})\r"
    )
    sys.stdout.flush()

async def main(match_id: str):
    client = FlashscoreWSClient(match_id)
    log.info("Starting Flashscore pipeline for match: %s", match_id)
    log.info("Press Ctrl+C to terminate.")
    
    try:
        await client.connect(handle_update)
    except KeyboardInterrupt:
        client.stop()
        log.info("Pipeline terminated.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m flashscore_pipeline.main <match_id>")
        sys.exit(1)
        
    match_id = sys.argv[1]
    asyncio.run(main(match_id))
