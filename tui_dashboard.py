"""
TennisBot77 Terminal Dashboard  —  tui_dashboard.py
Run:  ./start.sh tui   or   python tui_dashboard.py
"""

from __future__ import annotations
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Header, Footer, Static, Log, Sparkline
from textual.widget import Widget

TRADING_MODE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_mode.json")

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
LIVE_STATE_PATH  = os.path.join(BASE_DIR, "live_state.json")
BOT_LOG_PATH     = os.path.join(BASE_DIR, "bot.log")
TUI_EVENTS_PATH  = os.path.join(BASE_DIR, "tui_events.jsonl")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(prob: float, width: int = 22, color: str = "green") -> str:
    prob   = max(0.0, min(1.0, float(prob or 0)))
    filled = int(round(prob * width))
    return f"[{color}]{'█' * filled}[/][#555555]{'░' * (width - filled)}[/]"


def _ago(ts: float) -> str:
    d = time.time() - ts
    if d < 2:   return "[bold green]NOW[/]"
    if d < 10:  return f"[green]{d:.1f}s ago[/]"
    if d < 60:  return f"[yellow]{d:.0f}s ago[/]"
    return f"[red]{d/60:.1f}m ago[/]"


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_tail(path: str, offset: int):
    with open(path, "r", encoding="utf-8") as f:
        f.seek(offset)
        return f.tell() if f.seek(0, 2) == 0 else None, f.readlines()


# ── Panels ────────────────────────────────────────────────────────────────────

class MatchPanel(Static):
    """Big live match card: players, score, probs, live indicator."""
    data: reactive[dict] = reactive({}, recompose=True)

    def on_mount(self) -> None:
        self.border_title = "MATCH"

    def render(self) -> str:
        d = self.data
        if not d or d.get("feed_status") in (None, "scanning"):
            return ("[bold yellow]● SCANNING[/] — waiting for live market\n\n"
                    "Run [cyan]./start.sh[/] and Kalshi will be scanned automatically.")

        pa   = (d.get("player_a") or "Player A").upper()
        pb   = (d.get("player_b") or "Player B").upper()
        wa   = float(d.get("win_prob_a") or 0.5)
        wb   = float(d.get("win_prob_b") or 0.5)
        score = d.get("score") or "—"
        surf  = d.get("surface") or "Hard"
        fs    = (d.get("feed_status") or "unknown").upper()
        ticker = d.get("ticker") or ""

        live_tag = "[bold green]● LIVE[/]" if fs == "LIVE" else f"[bold yellow]◌ {fs}[/]"

        return (
            f"{live_tag}  [{surf.lower() == 'clay' and 'red' or 'cyan'}]{surf}[/]"
            f"  [dim]{ticker}[/]\n\n"
            f"[bold white]{pa}[/]\n"
            f"  {_bar(wa, 24, 'green')} [bold green]{wa*100:4.1f}%[/]\n\n"
            f"[bold white]{pb}[/]\n"
            f"  {_bar(wb, 24, 'blue')}  [bold blue]{wb*100:4.1f}%[/]\n\n"
            f"Score: [bold yellow]{score}[/]  "
            f"Updated: {_ago(float(d.get('last_update') or 0))}"
        )


class EnginePanel(Static):
    """Neural net / XGB / Markov probability bars + flow signal."""
    data: reactive[dict] = reactive({}, recompose=True)

    def on_mount(self) -> None:
        self.border_title = "ENGINE SIGNALS"

    def render(self) -> str:
        d = self.data
        if not d:
            return "Initializing engines..."

        nn   = float(d.get("nn_prob")    or 0.0)
        xgb  = float(d.get("xgb_prob")   or 0.0)
        mkv  = float(d.get("markov_prob") or float(d.get("win_prob_a") or 0.5))
        mode = (d.get("trading_mode") or "normal").upper()
        pa   = (d.get("player_a") or "A")[:12]

        flow = d.get("flow") or {}
        yp   = float(flow.get("yes_price") or d.get("yes_price") or 0.5)
        zsc  = float(flow.get("z_score")   or 0.0)
        vel  = float(flow.get("velocity")  or 0.0)
        dire = (flow.get("direction") or "NEUTRAL").upper()
        z_col = "magenta" if abs(zsc) >= 2 else ("yellow" if abs(zsc) >= 1 else "white")
        d_col = "green" if dire == "CONFIRM" else ("red" if dire == "FADE" else "white")

        edge_y = d.get("edge_yes")
        edge_n = d.get("edge_no")
        edge_line = ""
        if edge_y is not None:
            ec = "green" if (edge_y or 0) > 0.02 else "red"
            edge_line = f"\nEdge YES: [{ec}]{(edge_y or 0)*100:+.1f}%[/]"
        if edge_n is not None:
            ec2 = "green" if (edge_n or 0) > 0.02 else "red"
            edge_line += f"  Edge NO: [{ec2}]{(edge_n or 0)*100:+.1f}%[/]"

        last_act = d.get("last_action")
        act_line = f"\nLast action: [bold]{last_act}[/]" if last_act else ""

        return (
            f"[b]ML Baseline (for {pa})[/b]\n"
            f"NN LSTM:  {_bar(nn, 20, 'magenta')} [magenta]{nn*100:4.1f}%[/]\n"
            f"XGBoost:  {_bar(xgb, 20, 'cyan')}   [cyan]{xgb*100:4.1f}%[/]\n"
            f"Markov DP:{_bar(mkv, 20, 'green')}  [green]{mkv*100:4.1f}%[/]\n\n"
            f"[b]Kalshi Flow[/b]\n"
            f"YES Price: [yellow]{yp*100:.1f}¢[/]  "
            f"Z: [{z_col}]{zsc:+.2f}[/]  "
            f"[{d_col}]{dire}[/] ({vel:+.3f}¢/s)\n\n"
            f"Mode: [bold]{mode}[/]"
            f"{edge_line}{act_line}"
        )


class TradeLogPanel(Log):
    """Scrolling trade event log parsed from tui_events.jsonl + bot.log."""
    pass


class BotLogPanel(Log):
    """Filtered bot.log tail — shows key events, suppresses noise."""
    pass


class ProbSparkline(Widget):
    """Win-probability sparkline for Player A over recent ticks."""
    history: reactive[list] = reactive([], recompose=True)

    def on_mount(self) -> None:
        self.border_title = "Win Prob A — History (last 80 ticks)"

    def compose(self) -> ComposeResult:
        vals = list(self.history) or [0.5]
        yield Sparkline(vals, summary_function=max)

    def watch_history(self, vals: list) -> None:
        try:
            sp = self.query_one(Sparkline)
            sp.data = vals or [0.5]
        except Exception:
            pass


# ── Main App ──────────────────────────────────────────────────────────────────

class TennisBotTUI(App):
    """TennisBot77 terminal dashboard."""

    CSS = """
    Screen {
        background: $surface;
    }

    /* ── columns ── */
    #main_row {
        height: 1fr;
        layout: horizontal;
    }
    #left_col {
        width: 58%;
        layout: vertical;
    }
    #right_col {
        width: 42%;
        layout: vertical;
    }

    /* ── panel sizing ── */
    #match_panel   { height: 35%; border: round cyan;    padding: 0 1; margin: 0 1 0 1; }
    #spark_panel   { height: 13%; border: round #444;    padding: 0 1; margin: 0 1 0 1; }
    #tradelog      { height: 52%; border: round #22aa44; padding: 0 1; margin: 0 1 1 1; }

    #engine_panel  { height: 55%; border: round magenta; padding: 0 1; margin: 0 1 0 0; }
    #botlog        { height: 45%; border: round #3399ff; padding: 0 1; margin: 0 1 1 0; }

    Sparkline { height: 1fr; }

    /* text sizes */
    #match_panel Static { color: $text; }
    """

    BINDINGS = [
        ("d", "toggle_dark", "Dark"),
        ("q", "quit", "Quit"),
        ("c", "clear_logs", "Clear Logs"),
        ("s", "start_bot", "Start Bot"),
        ("x", "stop_bot", "Stop Bot"),
        ("h", "toggle_hf", "Toggle HF"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_row"):
            with Vertical(id="left_col"):
                yield MatchPanel(id="match_panel")
                yield ProbSparkline(id="spark_panel")
                yield TradeLogPanel(id="tradelog", highlight=True)
            with Vertical(id="right_col"):
                yield EnginePanel(id="engine_panel")
                yield BotLogPanel(id="botlog", highlight=True)
        yield Footer()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.title = "TennisBot77"
        self.sub_title = "scanning…"

        self._state_mtime:  float = 0.0
        self._events_mtime: float = 0.0
        self._log_offset:   int   = 0
        self._prob_history: deque[float] = deque(maxlen=80)
        self._tick: int = 0

        tradelog = self.query_one("#tradelog", TradeLogPanel)
        tradelog.border_title = "TRADE LOG"
        tradelog.write_line("[dim]Waiting for first trade evaluation…[/dim]")

        botlog = self.query_one("#botlog", BotLogPanel)
        botlog.border_title = "BOT LOG"
        botlog.write_line(f"[dim]Watching {BOT_LOG_PATH}[/dim]")

        # Seek to end of existing bot.log so we only show new lines
        self._log_offset = os.path.getsize(BOT_LOG_PATH) if os.path.exists(BOT_LOG_PATH) else 0

        self.set_interval(0.3,  self._poll_state)
        self.set_interval(0.5,  self._poll_events)
        self.set_interval(0.5,  self._poll_botlog)

    # ── polling ───────────────────────────────────────────────────────────────

    async def _poll_state(self) -> None:
        if not os.path.exists(LIVE_STATE_PATH):
            return
        try:
            mtime = os.path.getmtime(LIVE_STATE_PATH)
            if mtime <= self._state_mtime:
                return
            self._state_mtime = mtime

            data = await asyncio.to_thread(_read_json, LIVE_STATE_PATH)
            if not data:
                return

            match_panel  = self.query_one("#match_panel",  MatchPanel)
            engine_panel = self.query_one("#engine_panel", EnginePanel)
            spark        = self.query_one("#spark_panel",  ProbSparkline)

            match_panel.data  = dict(data)
            engine_panel.data = dict(data)
            match_panel.refresh()
            engine_panel.refresh()

            wa = float(data.get("win_prob_a") or 0.5)
            self._prob_history.append(wa)
            spark.history = list(self._prob_history)

            # Update subtitle with live match name
            pa = data.get("player_a") or ""
            pb = data.get("player_b") or ""
            fs = (data.get("feed_status") or "").lower()
            if pa and pb and fs not in ("scanning", ""):
                self.sub_title = f"{pa} vs {pb}  ●  {fs.upper()}"
            else:
                self.sub_title = "scanning…"

            self._tick += 1
        except Exception as exc:
            self.query_one("#botlog", BotLogPanel).write_line(
                f"[red][poll_state] {exc}[/red]"
            )

    async def _poll_events(self) -> None:
        if not os.path.exists(TUI_EVENTS_PATH):
            return
        try:
            mtime = os.path.getmtime(TUI_EVENTS_PATH)
            if mtime <= self._events_mtime:
                return
            self._events_mtime = mtime

            lines_raw = await asyncio.to_thread(self._read_events)
            tradelog  = self.query_one("#tradelog", TradeLogPanel)
            for ev in lines_raw:
                tradelog.write_line(self._format_event(ev))
        except Exception as exc:
            self.query_one("#botlog", BotLogPanel).write_line(
                f"[red][poll_events] {exc}[/red]"
            )

    @staticmethod
    def _read_events() -> list[dict]:
        """Read ALL lines from tui_events.jsonl and return parsed dicts."""
        out: list[dict] = []
        if not os.path.exists(TUI_EVENTS_PATH):
            return out
        with open(TUI_EVENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out[-50:]  # last 50 events

    @staticmethod
    def _format_event(ev: dict) -> str:
        ts   = datetime.fromtimestamp(float(ev.get("t") or time.time())).strftime("%H:%M:%S")
        pa   = (ev.get("pa") or "")[:12]
        pb   = (ev.get("pb") or "")[:12]
        sc   = ev.get("score") or "—"
        wa   = float(ev.get("prob_a") or 0.5)
        buys = int(ev.get("buys") or 0)
        pnl  = float(ev.get("pnl") or 0.0)
        mode = (ev.get("mode") or "").upper()
        nn   = ev.get("nn")
        xgb  = ev.get("xgb")
        nn_s  = f" NN={nn*100:.0f}%" if nn is not None else ""
        xgb_s = f" XGB={xgb*100:.0f}%" if xgb is not None else ""
        pnl_c = "green" if pnl >= 0 else "red"
        wa_c  = "green" if wa >= 0.55 else ("red" if wa <= 0.45 else "yellow")

        return (
            f"[dim]{ts}[/dim] [bold]{pa}[/bold] vs [bold]{pb}[/bold] "
            f"[{wa_c}]{wa*100:.1f}%[/] score=[yellow]{sc}[/]"
            f"{nn_s}{xgb_s} buys=[cyan]{buys}[/] "
            f"P&L=[{pnl_c}]{pnl:+.2f}[/] [{mode}]"
        )

    async def _poll_botlog(self) -> None:
        if not os.path.exists(BOT_LOG_PATH):
            return
        try:
            size = os.path.getsize(BOT_LOG_PATH)
            if size < self._log_offset:
                self._log_offset = 0
            if size <= self._log_offset:
                return

            new_size, new_lines = await asyncio.to_thread(
                self._tail_log, BOT_LOG_PATH, self._log_offset
            )
            self._log_offset = new_size

            botlog = self.query_one("#botlog", BotLogPanel)
            tradelog = self.query_one("#tradelog", TradeLogPanel)

            for raw in new_lines:
                line = raw.strip()
                if not line:
                    continue

                # Route trade-decision lines to trade log
                if any(k in line for k in ("Skipping:", "BUY YES", "BUY NO",
                                            "[CONVERGENCE]", "[FLOW] FADE",
                                            "[STATS-GATE]", "[MTO]", "[HF MODE]",
                                            "[MEAN-REV]", "DRY_RUN")):
                    tradelog.write_line(self._style_trade_line(line))
                    continue

                # Style and route bot log
                if "[ERROR]" in line:
                    botlog.write_line(f"[bold red]{line}[/bold red]")
                elif "[WARNING]" in line:
                    botlog.write_line(f"[yellow]{line}[/yellow]")
                elif "TENNIS DETECTED" in line:
                    botlog.write_line(f"[bold green]{line}[/bold green]")
                elif "[DISCOVER]" in line or "SCANNING" in line:
                    botlog.write_line(f"[cyan]{line}[/cyan]")
                elif "[LOCATION]" in line or "[SERVE" in line:
                    botlog.write_line(f"[dim]{line}[/dim]")
                else:
                    botlog.write_line(line)
        except Exception as exc:
            pass  # log read errors are transient

    @staticmethod
    def _tail_log(path: str, offset: int):
        with open(path, "r", encoding="utf-8") as f:
            f.seek(offset)
            lines = f.readlines()
            return f.tell(), lines

    @staticmethod
    def _style_trade_line(line: str) -> str:
        """Apply rich markup to a trade-decision log line."""
        if "BUY YES" in line or "BUY NO" in line:
            return f"[bold green]{line}[/bold green]"
        if "Skipping:" in line:
            return f"[dim]{line}[/dim]"
        if "[STATS-GATE]" in line or "[MTO]" in line:
            return f"[bold red]{line}[/bold red]"
        if "[CONVERGENCE]" in line and "BOOST" in line:
            return f"[bold cyan]{line}[/bold cyan]"
        if "[FLOW] FADE" in line:
            return f"[red]{line}[/red]"
        if "[MEAN-REV]" in line:
            return f"[bold yellow]{line}[/bold yellow]"
        return f"[yellow]{line}[/yellow]"

    # ── actions ───────────────────────────────────────────────────────────────

    def action_clear_logs(self) -> None:
        self.query_one("#tradelog", TradeLogPanel).clear()
        self.query_one("#botlog",   BotLogPanel).clear()

    async def action_start_bot(self) -> None:
        botlog = self.query_one("#botlog", BotLogPanel)
        try:
            def _start():
                log_path = os.path.join(BASE_DIR, "bot.log")
                subprocess.Popen(
                    [sys.executable, os.path.join(BASE_DIR, "main.py"), "--bot-only"],
                    stdout=open(log_path, "w"),
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    cwd=BASE_DIR,
                )
            await asyncio.to_thread(_start)
            botlog.write_line("[bold green][CMD] Bot started.[/bold green]")
        except Exception as e:
            botlog.write_line(f"[red][CMD] start_bot error: {e}[/red]")

    async def action_stop_bot(self) -> None:
        botlog = self.query_one("#botlog", BotLogPanel)
        try:
            def _stop():
                subprocess.run(["pkill", "-f", "main.py"], capture_output=True)
            await asyncio.to_thread(_stop)
            botlog.write_line("[bold red][CMD] Bot stopped.[/bold red]")
        except Exception as e:
            botlog.write_line(f"[red][CMD] stop_bot error: {e}[/red]")

    async def action_toggle_hf(self) -> None:
        botlog = self.query_one("#botlog", BotLogPanel)
        try:
            def _toggle():
                try:
                    with open(TRADING_MODE_PATH, "r") as f:
                        current = json.load(f).get("mode", "normal")
                except Exception:
                    current = "normal"
                new_mode = "normal" if current == "hf" else "hf"
                with open(TRADING_MODE_PATH, "w") as f:
                    json.dump({"mode": new_mode}, f)
                return new_mode
            new_mode = await asyncio.to_thread(_toggle)
            color = "cyan" if new_mode == "hf" else "green"
            botlog.write_line(f"[bold {color}][CMD] Trading mode → {new_mode.upper()}[/bold {color}]")
        except Exception as e:
            botlog.write_line(f"[red][CMD] toggle_hf error: {e}[/red]")


if __name__ == "__main__":
    app = TennisBotTUI()
    app.run()
