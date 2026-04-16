from markov_engine import LiveMatchState

def test_markov():
    try:
        p_serve = 0.65
        p_return = 0.35
        lms = LiveMatchState(p_serve, p_return)
        prob = lms.win_probability()
        print(f"Prob: {prob}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_markov()
