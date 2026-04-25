from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Log, Button
from textual.reactive import reactive
import asyncio
import json
import os

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
LIVE_STATE_PATH = os.path.join(BASE_DIR, "live_state.json")
BOT_LOG_PATH    = os.path.join(BASE_DIR, "bot.log")


class MatchStatusPanel(Static):
    """Displays live score and current players."""
    match_data: reactive[dict] = reactive({}, recompose=True)

    def on_mount(self) -> None:
        self.border_title = "Match Status"

    def render(self) -> str:
        if not self.match_data:
            return "Waiting for engine..."

        p_a    = self.match_data.get("player_a") or "Player A"
        p_b    = self.match_data.get("player_b") or "Player B"
        status = self.match_data.get("feed_status") or "UNKNOWN"
        score  = self.match_data.get("score") or ""
        score_line = f"\nScore: [white]{score}[/]" if score else ""

        return (
            f"[bold green]{p_a}[/] [white]vs[/] [bold blue]{p_b}[/]"
            f"{score_line}\n\n"
            f"Feed Status: [bold yellow]{status.upper()}[/]"
        )


class EngineStatsPanel(Static):
    """Displays Neural Net, XGBoost, and Markov Stats."""
    stats_data: reactive[dict] = reactive({}, recompose=True)

    def on_mount(self) -> None:
        self.border_title = "Engine Stats & Probability Charts"

    def _render_bar(self, prob: float, color: str) -> str:
        bar_len = 20
        prob    = max(0.0, min(1.0, float(prob)))
        filled  = int(round(prob * bar_len))
        empty   = bar_len - filled
        return f"[{color}]{'█' * filled}[/][#444444]{'░' * empty}[/]"

    def render(self) -> str:
        if not self.stats_data:
            return "Initializing..."

        nn_prob  = float(self.stats_data.get("nn_prob")  or 0.0)
        xgb_prob = float(self.stats_data.get("xgb_prob") or 0.0)
        p_a_live = float(self.stats_data.get("win_prob_a") or 0.0)
        p_b_live = float(self.stats_data.get("win_prob_b") or 0.0)
        mode     = self.stats_data.get("trading_mode") or "normal"

        bar_a = self._render_bar(p_a_live, "green")
        bar_b = self._render_bar(p_b_live, "blue")

        return (
            f"[b]Machine Learning Baseline[/b]\n"
            f"Neural Net (LSTM): [magenta]{nn_prob*100:.1f}%[/]\n"
            f"XGBoost: [cyan]{xgb_prob*100:.1f}%[/]\n\n"
            f"[b]Live Markov Chain DP (Real-Time)[/b]\n"
            f"Player A ({p_a_live*100:4.1f}%): {bar_a}\n"
            f"Player B ({p_b_live*100:4.1f}%): {bar_b}\n\n"
            f"Trading Mode: [b]{mode.upper()}[/b]"
        )


class FlowPanel(Static):
    """Displays Kalshi orderbook flow and Z-Scores."""
    flow_data: reactive[dict] = reactive({}, recompose=True)

    def on_mount(self) -> None:
        self.border_title = "Kalshi Tape / Flow"

    def render(self) -> str:
        if not self.flow_data:
            return "Initializing..."

        direction = self.flow_data.get("direction") or "NEUTRAL"
        vel       = float(self.flow_data.get("velocity")  or 0.0)
        z_score   = float(self.flow_data.get("z_score")   or 0.0)
        price     = float(self.flow_data.get("yes_price") or 0.5) * 100

        dir_color = "green" if direction == "CONFIRM" else ("red" if direction == "FADE" else "white")
        z_color   = "magenta" if abs(z_score) >= 2.0 else "white"

        return (
            f"Live YES Price: [yellow]{price:.1f}¢[/]\n\n"
            f"[b]Algorithmic Signals[/b]\n"
            f"Momentum: [{dir_color}]{direction}[/] ({vel:+.3f}¢/s)\n"
            f"Mispricing Z-Score: [{z_color}]{z_score:+.2f}[/]\n"
        )


class TennisBotTUI(App):
    """The main TUI Application."""

    CSS = """
    Screen {
        layout: horizontal;
        background: $surface;
        padding: 1 2;
    }

    #left_column {
        width: 60%;
        height: 100%;
        layout: vertical;
    }

    #right_column {
        width: 40%;
        height: 100%;
        layout: vertical;
    }

    .panel {
        border: round cyan;
        padding: 1;
        margin: 1;
        background: $background;
        color: $text;
    }

    #match_panel {
        height: 25%;
    }

    #flow_panel {
        height: 30%;
    }

    #log_panel {
        height: 45%;
        border: round green;
    }

    #engine_panel {
        height: 50%;
    }

    #controls_panel {
        height: 50%;
        align: center middle;
    }

    Button {
        margin: 1 2;
        min-width: 20;
    }
    """

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left_column"):
                yield MatchStatusPanel(id="match_panel", classes="panel")
                yield FlowPanel(id="flow_panel", classes="panel")
                yield Log(id="log_panel", classes="panel")
            with Vertical(id="right_column"):
                yield EngineStatsPanel(id="engine_panel", classes="panel")
                with Vertical(id="controls_panel", classes="panel"):
                    yield Static("[b]System Controls[/b]", classes="label")
                    yield Button("Start Engine", id="btn_start", variant="success")
                    yield Button("Stop Engine",  id="btn_stop",  variant="error")
                    yield Button("Toggle Mode (HF/Normal)", id="btn_mode", variant="primary")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        log_widget = self.query_one(Log)
        if event.button.id == "btn_start":
            log_widget.write_line("[bold green]> Engine started (Simulation)[/]")
        elif event.button.id == "btn_stop":
            log_widget.write_line("[bold red]> Engine stopped[/]")
        elif event.button.id == "btn_mode":
            log_widget.write_line("[bold blue]> Toggling High-Frequency Mode...[/]")

    def on_mount(self) -> None:
        self.title = "TennisBot77 TUI"
        self.last_log_size = os.path.getsize(BOT_LOG_PATH) if os.path.exists(BOT_LOG_PATH) else 0
        self._last_state_mtime: float = 0.0

        log_widget = self.query_one(Log)
        log_widget.border_title = "System Logs"
        log_widget.write_line(f"[green]TennisBot77 TUI Initialized.[/green]")
        log_widget.write_line(f"Watching: {LIVE_STATE_PATH}")
        log_widget.write_line("Awaiting connection to main.py engine...")

        self.set_interval(0.25, self.poll_state_files)
        self.set_interval(0.5,  self.poll_logs)

    async def poll_state_files(self) -> None:
        """Poll live_state.json; only parse when the file has actually changed."""
        if not os.path.exists(LIVE_STATE_PATH):
            return
        try:
            mtime = os.path.getmtime(LIVE_STATE_PATH)
            if mtime <= self._last_state_mtime:
                return
            self._last_state_mtime = mtime

            data = await asyncio.to_thread(self._read_json, LIVE_STATE_PATH)
            if data is None:
                return

            match_panel  = self.query_one("#match_panel",  MatchStatusPanel)
            engine_panel = self.query_one("#engine_panel", EngineStatsPanel)
            flow_panel   = self.query_one("#flow_panel",   FlowPanel)

            # Force reactivity: assign new dict objects so Textual detects the change
            match_panel.match_data   = dict(data)
            engine_panel.stats_data  = dict(data)
            flow_panel.flow_data     = dict(data.get("flow") or {})

            match_panel.refresh()
            engine_panel.refresh()
            flow_panel.refresh()
        except Exception as exc:
            self.query_one(Log).write_line(f"[red][TUI ERR] poll_state_files: {exc}[/red]")

    @staticmethod
    def _read_json(path: str):
        with open(path, "r") as f:
            return json.load(f)

    async def poll_logs(self) -> None:
        """Tail bot.log for new lines."""
        if not os.path.exists(BOT_LOG_PATH):
            return
        try:
            size = os.path.getsize(BOT_LOG_PATH)
            if size < self.last_log_size:
                self.last_log_size = 0
            if size <= self.last_log_size:
                return

            lines = await asyncio.to_thread(self._read_tail, BOT_LOG_PATH, self.last_log_size)
            new_size, new_lines = lines
            self.last_log_size = new_size

            log_widget = self.query_one(Log)
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                if "[ERROR]" in line:
                    log_widget.write_line(f"[red]{line}[/red]")
                elif "[WARNING]" in line:
                    log_widget.write_line(f"[yellow]{line}[/yellow]")
                elif "[FLOW WS]" in line:
                    log_widget.write_line(f"[cyan]{line}[/cyan]")
                else:
                    log_widget.write_line(line)
        except Exception as exc:
            self.query_one(Log).write_line(f"[red][TUI ERR] poll_logs: {exc}[/red]")

    @staticmethod
    def _read_tail(path: str, offset: int):
        with open(path, "r") as f:
            f.seek(offset)
            lines = f.readlines()
            return f.tell(), lines


if __name__ == "__main__":
    app = TennisBotTUI()
    app.run()
