import os
import logging
import sys
from typing import Any, Dict
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

from app.config import Settings
from app.core.engine import SignalEngine
from app.core.telegram import send_telegram

# ---------- logging ----------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("signals")
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_sh)

logger.setLevel(getattr(logging, Settings().LOG_LEVEL.upper(), logging.INFO))
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler("logs/signals.log")
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# ---------- app setup ----------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "dev")
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", path='/socket.io')

# health
@app.route('/health')
def health():
    return {'status': 'ok'}

# ---------- socket helpers ----------
def emit_log(msg: str):
    try:
        socketio.emit("log", {"line": msg})
    except Exception:
        pass

def emit_status(asset: str, status: Dict[str, Any]):
    try:
        socketio.emit("status", {"asset": asset, "status": status})
    except Exception:
        pass

settings = Settings()
engine = SignalEngine(log_emit=emit_log, status_emit=emit_status)

# ---------- routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    data = {
        'running': bool(getattr(engine, 'is_running', lambda: False)()),
        'exchange': getattr(engine, 'exchange_name', None),
        'quote': getattr(engine, 'quote', None),
        'last_state': getattr(engine, 'last_state', {})
    }
    # optional: include strategies if available
    try:
        if hasattr(engine, 'list_strategies'):
            data['strategies'] = list(engine.list_strategies())
    except Exception:
        pass
    return jsonify(data)
@app.route('/start', methods=['POST'])
def start():
    try:
        if engine.is_running():
            return jsonify({"ok": True, "running": True, "note": "already running"})
        engine.start()
        return jsonify({"ok": True, "running": True})
    except Exception as e:
        logger.exception("Error in /start")
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/stop', methods=['POST'])
def stop():
    try:
        engine.stop()
        return jsonify({"ok": True, "running": False})
    except Exception as e:
        logger.exception("Error in /stop")
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/logs/clear', methods=['POST'])
def logs_clear():
    try:
        path = "logs/signals.log"
        open(path, "w").close()
        emit_log("== LOGS CLEARED ==")
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Error clearing logs")
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/telegram/test', methods=['POST'])
def telegram_test():
    try:
        r = send_telegram("✅ Test message from Crypto Signals bot")
        return jsonify(r), (200 if r.get("ok") else 400)
    except Exception as e:
        logger.exception("Error in telegram test")
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/signals/force', methods=['POST'])
def signals_force():
    try:
        sent = engine.force_send_all()
        return jsonify({"ok": True, "sent": sent})
    except Exception as e:
        logger.exception("Error in /signals/force")
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/signals/reset', methods=['POST'])
def signals_reset():
    try:
        engine.reset_states()
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Error in /signals/reset")
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/movers/scan', methods=['POST'])
def movers_scan():
    try:
        res = engine.scan_movers_once()
        return jsonify({"ok": True, "found": res})
    except Exception as e:
        logger.exception("Error in /movers/scan")
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == '__main__':
    socketio.run(app, host=settings.HOST, port=settings.PORT, use_reloader=False)


@app.route('/logs', methods=['GET'])
def logs_tail():
    try:
        path = "logs/signals.log"
        lines = int(request.args.get('lines', 300))
        if lines <= 0 or lines > 5000:
            lines = 300
        data = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.readlines()[-lines:]
        except FileNotFoundError:
            data = []
        # strip newline
        data = [ln.rstrip("\n") for ln in data]
        return jsonify({"ok": True, "lines": data})
    except Exception as e:
        logger.exception("Error in /logs")
        return jsonify({"ok": False, "error": str(e)}), 400
