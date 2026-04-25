from textual.app import App, ComposeResult
from textual.containers import Grid, VerticalScroll, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Log, Label, Button
from textual.reactive import reactive
import asyncio
import json
import os

class MatchStatusPanel(Static):
    """Displays live score and current players."""
    match_data = reactive({})

    def on_mount(self) -> None:
        self.border_title = "Match Status"

    def render(self) -> str:
        if not self.match_data:
            return "Waiting for engine..."
        
        p_a = self.match_data.get("player_a", "Player A")
        p_b = self.match_data.get("player_b", "Player B")
        status = self.match_data.get("feed_status", "UNKNOWN")
        
        return f"[bold green]{p_a}[/] [white]vs[/] [bold blue]{p_b}[/]\n\nFeed Status: [bold yellow]{status}[/]"

class EngineStatsPanel(Static):
    """Displays Neural Net, XGBoost, and Markov Stats."""
    stats_data = reactive({})

    def on_mount(self) -> None:
        self.border_title = "Engine Stats & Probability Charts"

    def _render_bar(self, prob: float, color: str) -> str:
        """Render a termui-style ASCII block bar chart."""
        bar_len = 20
        # Clamp probability
        prob = max(0.0, min(1.0, prob))
        filled = int(round(prob * bar_len))
        empty = bar_len - filled
        return f"[{color}]{'█' * filled}[/][#444444]{'░' * empty}[/]"

    def render(self) -> str:
        if not self.stats_data:
            return "Initializing..."
        
        nn_prob  = self.stats_data.get("nn_prob")  or 0.0
        xgb_prob = self.stats_data.get("xgb_prob") or 0.0
        p_a_live = self.stats_data.get("win_prob_a") or 0.0
        p_b_live = self.stats_data.get("win_prob_b") or 0.0
        mode = self.stats_data.get("trading_mode", "normal")
        
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
    flow_data = reactive({})

    def on_mount(self) -> None:
        self.border_title = "Kalshi Tape / Flow"

    def render(self) -> str:
        if not self.flow_data:
            return "Initializing..."
        
        direction = self.flow_data.get("direction") or "NEUTRAL"
        vel       = self.flow_data.get("velocity")  or 0.0
        z_score   = self.flow_data.get("z_score")   or 0.0
        price     = (self.flow_data.get("yes_price") or 0.5) * 100
        
        dir_color = "green" if direction == "CONFIRM" else ("red" if direction == "FADE" else "white")
        z_color = "magenta" if abs(z_score) >= 2.0 else "white"

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
                    yield Button("Stop Engine", id="btn_stop", variant="error")
                    yield Button("Toggle Mode (HF/Normal)", id="btn_mode", variant="primary")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle control button clicks."""
        log_widget = self.query_one(Log)
        if event.button.id == "btn_start":
            log_widget.write_line("[bold green]> Engine started (Simulation)[/]")
        elif event.button.id == "btn_stop":
            log_widget.write_line("[bold red]> Engine stopped[/]")
        elif event.button.id == "btn_mode":
            log_widget.write_line("[bold blue]> Toggling High-Frequency Mode...[/]")

    def on_mount(self) -> None:
        self.title = "Kolibri-Style TennisBot77 TUI"
        self.set_interval(0.1, self.poll_state_files)
        self.set_interval(0.5, self.poll_logs)
        if os.path.exists("bot.log"):
            self.last_log_size = os.path.getsize("bot.log")
        else:
            self.last_log_size = 0
        
        # Write initial instruction to log
        log_widget = self.query_one(Log)
        log_widget.border_title = "System Logs"
        log_widget.write_line("[green]TennisBot77 TUI Initialized.[/green]")
        log_widget.write_line("Awaiting connection to main.py engine...")

    def poll_state_files(self) -> None:
        """Poll the live_state JSON file for zero-latency UI updates."""
        try:
            if os.path.exists("live_state.json"):
                with open("live_state.json", "r") as f:
                    data = json.load(f)
                    
                    self.query_one("#match_panel").match_data = data
                    self.query_one("#engine_panel").stats_data = data
                    self.query_one("#flow_panel").flow_data = data.get("flow", {})
        except Exception:
            pass # Ignore read collisions

    def poll_logs(self) -> None:
        """Tail the bot.log file for real-time background prints."""
        if not os.path.exists("bot.log"):
            return
            
        try:
            size = os.path.getsize("bot.log")
            if size < self.last_log_size:
                # Log rotated
                self.last_log_size = 0
            
            if size > self.last_log_size:
                with open("bot.log", "r") as f:
                    f.seek(self.last_log_size)
                    new_lines = f.readlines()
                    self.last_log_size = f.tell()
                    
                    log_widget = self.query_one(Log)
                    for line in new_lines:
                        # Very simple styling for ERROR/WARNING/INFO
                        line = line.strip()
                        if "[ERROR]" in line:
                            log_widget.write_line(f"[red]{line}[/red]")
                        elif "[WARNING]" in line:
                            log_widget.write_line(f"[yellow]{line}[/yellow]")
                        elif "[FLOW WS]" in line:
                            log_widget.write_line(f"[cyan]{line}[/cyan]")
                        else:
                            log_widget.write_line(line)
        except Exception:
            pass

if __name__ == "__main__":
    app = TennisBotTUI()
    app.run()
