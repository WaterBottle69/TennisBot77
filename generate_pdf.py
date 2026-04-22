from fpdf import FPDF

class MarkovPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Markovian Probability Engine: Tennis Betting Bot', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

pdf = MarkovPDF()
pdf.add_page()

# Section 1
pdf.chapter_title('1. Level 1: Point Win Probability (Base Features)')
pdf.chapter_body(
    "Inputs for the model are the win rates for each player depending on who is serving:\n"
    "- p_serve: Probability Player A wins a point on their own serve.\n"
    "- p_return: Probability Player A wins a point when the opponent is serving.\n\n"
    "These features are derived from H2H stats and ATP rankings."
)

# Section 2
pdf.chapter_title('2. Level 2: Game Win Probability (P_G)')
pdf.chapter_body(
    "The probability of winning a game from a specific point score (i, j) where i, j are 0, 15, 30, or 40.\n\n"
    "Recursive Equation:\n"
    "P(i, j) = p * P(i+1, j) + (1-p) * P(i, j+1)\n\n"
    "Deuce Boundary Condition:\n"
    "If the score reaches 40-40 (Deuce), the cycle of Advantage/Deuce converges analytically to:\n"
    "P(3, 3) = p^2 / (p^2 + (1-p)^2)"
)

# Section 3
pdf.chapter_title('3. Level 3: Set Win Probability (P_S)')
pdf.chapter_body(
    "Models the probability of winning a set based on games won (G1, G2). The server alternates every game.\n\n"
    "Transition Equation:\n"
    "P(G1, G2) = PG * P(G1+1, G2) + (1-PG) * P(G1, G2+1)\n\n"
    "Tiebreak Logic (P_TB):\n"
    "In a tiebreak (first to 7 points), servers alternate every 2 points. The probability of winning from 6-6 in a tiebreak is:\n"
    "PTB(6, 6) = (p_serve * p_return) / (p_serve * p_return + (1-p_serve) * (1-p_return))"
)

# Section 4
pdf.chapter_title('4. Level 4: Match Win Probability (P_M)')
pdf.chapter_body(
    "The top-level result for a Best of 3 sets match:\n"
    "P(S1, S2) = PS * P(S1+1, S2) + (1-PS) * P(S1, S2+1)\n\n"
    "- S1, S2: Current sets won by each player.\n"
    "- PS: Probability of winning the current set."
)

# Section 5
pdf.chapter_title('5. Real-Time Integration')
pdf.chapter_body(
    "- Caching: Uses @lru_cache for millisecond-latency updates.\n"
    "- Dynamic Updates: Every point won or lost triggers an instantaneous recalculation, providing a Fair Value for Kalshi betting."
)

pdf.output('/Users/omsilwal/Downloads/TennisBetting-main-3/TennisBetting-main/markov_explanation.pdf')
print("PDF generated successfully.")
