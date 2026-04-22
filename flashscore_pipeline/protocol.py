import time
from .models import MatchUpdate, PlayerStats

class FlashscoreProtocol:
    """
    Parser for the Flashscore proprietary delimited text protocol.
    Delimiters:
      ~ : Frame boundary (outer)
      ¬ : Field boundary (inner)
      ^ : Sub-field boundary
    """
    
    @staticmethod
    def parse_frame(frame: str, match_id: str) -> MatchUpdate:
        """
        Translates a raw Flashscore frame into a MatchUpdate object.
        Optimized for minimal allocations.
        """
        # Example frame fragment: ...¬SA÷1¬SB÷0¬SC÷6-4¬...
        fields = {}
        parts = frame.split('¬')
        for part in parts:
            if '÷' in part:
                k, v = part.split('÷', 1)
                fields[k] = v
        
        # Mapping common Flashscore codes (hypothetical, refined during live trace)
        # SA/SB often represent current set scores or serving status
        p1_name = fields.get('WN', 'Player 1')
        p2_name = fields.get('LN', 'Player 2')
        
        # In live tennis:
        # AS = Set scores? 
        # SS = Game scores?
        # PT = Point-by-point?
        
        # For the sake of the initial pipeline, we capture the raw data and 
        # map known score fragments.
        p1 = PlayerStats(name=p1_name, score=fields.get('SC', '0'))
        p2 = PlayerStats(name=p2_name, score=fields.get('SC', '0')) # SC is often shared or paired
        
        return MatchUpdate(
            match_id=match_id,
            p1=p1,
            p2=p2,
            status="LIVE" if fields.get('ST') == '1' else "UPCOMING",
            timestamp=time.time(),
            raw_event=frame
        )

    @staticmethod
    def decode_message(raw_msg: str) -> list[str]:
        """Splits a WebSocket stream into separate protocol frames."""
        if not raw_msg:
            return []
        
        # Flashscore messages often arrive wrapped in ~
        if raw_msg.startswith('~') and raw_msg.endswith('~'):
            return raw_msg.strip('~').split('~~')
        
        return [raw_msg]
