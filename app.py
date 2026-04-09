"""
================================================================================
APP.PY — Flask Application Server & API Routes
================================================================================
Project: Complex Random Process Analysis for Communication Systems
         with Applications to Speech Enhancement
================================================================================
Authors: Navin Kumar PG (24BEC1055)
         A.P. Anirudh       (24BEC1158)
         Kailash N H        (24BEC1546)
Faculty: Dr. Kalaivan K
================================================================================
Description:
    This module is the entry point for the web application.  It creates and
    configures a Flask server that:

        • Serves the Single Page Application (SPA) front-end
        • Exposes RESTful API endpoints for:
            – Audio file upload
            – Synthetic test audio generation
            – DSP processing pipeline execution
            – Results retrieval
            – Audio file playback
            – Source code viewing (for the in-app code showcase)
            – Report download
        • Manages the SQLite database lifecycle via SQLAlchemy
        • Handles CORS, error responses, and logging

    All API routes return JSON responses.  Audio files are served via
    Flask's ``send_file`` for correct MIME types and range-request support.

Revision History:
    2026-04-08  Initial creation
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import uuid
import json
import time
import logging
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps

# ──────────────────────────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────────────────────────
from flask import (  # type: ignore
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    abort,
    Response,
    make_response,
)

# ──────────────────────────────────────────────────────────────────────────────
# Local Imports
# ──────────────────────────────────────────────────────────────────────────────
from database import (
    db,
    init_database,
    create_session,
    get_session,
    get_all_sessions,
    update_session_results,
    fail_session,
    add_processing_log,
    get_processing_logs,
    get_audio_samples,
    get_audio_sample_by_key,
    generate_text_report,
    ProcessingResult,
)
from speech_engine import (
    process_pipeline,
    process_uploaded_audio,
    process_synthetic_audio,
    generate_test_audio,
    load_audio_file,
    save_audio_to_wav,
    DEFAULT_FILTER_ORDER,
    DEFAULT_CUTOFF_HZ,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_DURATION,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Allowed Extensions
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "webm", "aac"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB limit

# ──────────────────────────────────────────────────────────────────────────────
# Source Files Available for Code Viewer
# ──────────────────────────────────────────────────────────────────────────────
SOURCE_FILES = {
    "app.py": BASE_DIR / "app.py",
    "speech_engine.py": BASE_DIR / "speech_engine.py",
    "database.py": BASE_DIR / "database.py",
    "index.html": TEMPLATES_DIR / "index.html",
    "main.js": STATIC_DIR / "js" / "main.js",
    "style.css": STATIC_DIR / "css" / "style.css",
}


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK APPLICATION FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns
    -------
    Flask
        The configured Flask application instance.
    """
    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        template_folder=str(TEMPLATES_DIR),
    )

    # ── Configuration ─────────────────────────────────────────────────
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dsp-vit-2026-secret-key")
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{BASE_DIR / 'instance' / 'dsp_app.db'}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
    app.config["PROCESSED_FOLDER"] = str(PROCESSED_DIR)

    # Ensure the instance directory exists
    (BASE_DIR / "instance").mkdir(exist_ok=True)

    # ── Initialise extensions ─────────────────────────────────────────
    db.init_app(app)
    init_database(app)

    # ── Register routes ───────────────────────────────────────────────
    _register_routes(app)

    # ── Register error handlers ───────────────────────────────────────
    _register_error_handlers(app)

    # ── CORS headers ──────────────────────────────────────────────────
    @app.after_request
    def add_cors_headers(response: Response) -> Response:
        """Add CORS headers to every response."""
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Max-Age"] = "3600"
        return response

    logger.info("Flask application created and configured.")
    logger.info("Upload directory  : %s", UPLOAD_DIR)
    logger.info("Processed directory: %s", PROCESSED_DIR)
    logger.info("Database URI      : %s", app.config["SQLALCHEMY_DATABASE_URI"])

    return app


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    """
    Check if a filename has an allowed extension.

    Parameters
    ----------
    filename : str
        The filename to check.

    Returns
    -------
    bool
        True if the extension is in ALLOWED_EXTENSIONS.
    """
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def get_file_extension(filename: str) -> str:
    """
    Extract the file extension from a filename.

    Parameters
    ----------
    filename : str
        The filename.

    Returns
    -------
    str
        The extension (lowercase, without dot).
    """
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def secure_filename_custom(filename: str) -> str:
    """
    Sanitise a filename to prevent directory traversal attacks.

    Parameters
    ----------
    filename : str
        The original filename.

    Returns
    -------
    str
        A safe version of the filename.
    """
    # Remove path separators and dangerous characters
    filename = filename.replace("/", "_").replace("\\", "_")
    filename = filename.replace("..", "_")
    # Keep only alphanumeric, dots, underscores, hyphens
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in "._-":
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars) or "unnamed"


def api_response(data: dict, status_code: int = 200, message: str = "OK") -> tuple:
    """
    Create a standardised JSON API response.

    Parameters
    ----------
    data : dict
        The response payload.
    status_code : int
        HTTP status code.
    message : str
        Status message.

    Returns
    -------
    tuple
        (Response, status_code)
    """
    response = {
        "status": "success" if status_code < 400 else "error",
        "message": message,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(response), status_code


def api_error(message: str, status_code: int = 400, details: dict = None) -> tuple:
    """
    Create a standardised JSON error response.

    Parameters
    ----------
    message : str
        Error description.
    status_code : int
        HTTP status code.
    details : dict or None
        Additional error details.

    Returns
    -------
    tuple
        (Response, status_code)
    """
    response = {
        "status": "error",
        "message": message,
        "details": details or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(response), status_code


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════

def _register_routes(app: Flask) -> None:
    """
    Register all route handlers on the Flask application.

    Parameters
    ----------
    app : Flask
        The Flask application instance.
    """

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: INDEX (Serve SPA)
    # ──────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        """
        Serve the main Single Page Application HTML.

        This is the only HTML page — all navigation happens via
        JavaScript in the front-end.
        """
        logger.info("Serving index.html")
        return render_template("index.html")

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: HEALTH CHECK
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/health")
    def health_check():
        """
        Simple health check endpoint.

        Returns basic server status and configuration info.
        """
        return api_response({
            "server": "running",
            "project": "Complex Random Process Analysis",
            "version": "1.0.0",
            "team": [
                {"name": "Navin Kumar PG", "id": "24BEC1055"},
                {"name": "A.P. Anirudh", "id": "24BEC1158"},
                {"name": "Kailash N H", "id": "24BEC1546"},
            ],
            "faculty": "Dr. Kalaivan K",
            "database": "connected",
            "upload_dir": str(UPLOAD_DIR),
            "processed_dir": str(PROCESSED_DIR),
        })

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: UPLOAD AUDIO
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/upload", methods=["POST"])
    def upload_audio():
        """
        Handle audio file upload.

        Accepts multipart/form-data with a file field named "audio".
        Saves the file to the uploads directory and creates a new
        database session.

        Returns
        -------
        JSON
            { session_id, filename, message }
        """
        logger.info("Received upload request")

        # Validate file presence
        if "audio" not in request.files:
            return api_error("No audio file provided. Use field name 'audio'.", 400)

        file = request.files["audio"]

        if file.filename == "" or file.filename is None:
            return api_error("No file selected.", 400)

        # Validate extension
        if not allowed_file(file.filename):
            return api_error(
                f"File type not allowed. Accepted formats: {', '.join(ALLOWED_EXTENSIONS)}",
                400,
            )

        # Generate unique filename
        session_id = str(uuid.uuid4())
        original_name = secure_filename_custom(file.filename)
        extension = get_file_extension(original_name)
        saved_name = f"input_{session_id}.{extension}"
        save_path = UPLOAD_DIR / saved_name

        # Save file
        try:
            file.save(str(save_path))
            logger.info("File saved: %s (%d bytes)", save_path, save_path.stat().st_size)
        except Exception as exc:
            logger.error("Failed to save upload: %s", exc)
            return api_error(f"Failed to save file: {exc}", 500)

        # ── Convert non-WAV files to WAV for reliable pipeline processing ─
        if extension != "wav":
            try:
                from pydub import AudioSegment  # type: ignore
                audio_seg = AudioSegment.from_file(str(save_path))
                wav_name = f"input_{session_id}.wav"
                wav_path = UPLOAD_DIR / wav_name
                audio_seg.export(str(wav_path), format="wav")
                logger.info(
                    "Converted %s → WAV: %s (%d bytes)",
                    extension, wav_path, wav_path.stat().st_size,
                )
                # Remove original non-WAV file
                try:
                    save_path.unlink()
                except OSError:
                    pass
                # Update references to use the WAV file
                save_path = wav_path
                saved_name = wav_name
            except Exception as conv_exc:
                logger.warning(
                    "Could not convert %s to WAV: %s (will try pydub at process time)",
                    extension, conv_exc,
                )

        # Create database session
        try:
            db_session = create_session(
                filename=original_name,
                sample_type="upload",
            )
            # Override the auto-generated session_id
            db_session.session_id = session_id
            db_session.input_audio_path = str(save_path)
            db.session.commit()

            logger.info(
                "Created upload session: %s for file '%s'",
                session_id[:8], original_name,
            )
        except Exception as exc:
            logger.error("Failed to create session: %s", exc)
            return api_error(f"Database error: {exc}", 500)

        return api_response(
            {
                "session_id": session_id,
                "filename": original_name,
                "file_size": save_path.stat().st_size,
                "file_path": str(save_path),
            },
            status_code=201,
            message="Audio file uploaded successfully.",
        )

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: GENERATE SYNTHETIC SAMPLE
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/generate-sample", methods=["POST"])
    def generate_sample():
        """
        Generate a synthetic test audio sample.

        Expects JSON body:
            { "sample_key": "cafe_noise" | "traffic_noise" | ... }

        Creates a synthetic speech signal with calibrated AWGN noise
        and saves it to the uploads directory.

        Returns
        -------
        JSON
            { session_id, sample_key, snr_db, duration }
        """
        logger.info("Received generate-sample request")

        data = request.get_json(silent=True) or {}
        sample_key = data.get("sample_key", "cafe_noise")

        # Validate sample key
        sample_info = get_audio_sample_by_key(sample_key)
        if sample_info is None:
            available = [s["sample_key"] for s in get_audio_samples()]
            return api_error(
                f"Unknown sample key: '{sample_key}'. Available: {available}",
                400,
            )

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Create database session
        try:
            db_session = create_session(
                filename=f"{sample_key}_synthetic.wav",
                sample_type=sample_key,
            )
            db_session.session_id = session_id
            db.session.commit()
        except Exception as exc:
            logger.error("Failed to create session: %s", exc)
            return api_error(f"Database error: {exc}", 500)

        # Generate the test audio
        try:
            input_path, clean_signal, noisy_signal, noise = generate_test_audio(
                sample_key=sample_key,
                duration=sample_info.get("duration_seconds", DEFAULT_DURATION),
                sample_rate=DEFAULT_SAMPLE_RATE,
                snr_db=sample_info.get("snr_db", 5.0),
                output_dir=str(UPLOAD_DIR),
                session_id=session_id,
            )

            # Update session with audio path
            db_session.input_audio_path = input_path
            db.session.commit()

            logger.info(
                "Generated synthetic sample: %s (SNR=%.1f dB) → %s",
                sample_key, sample_info["snr_db"], input_path,
            )

        except Exception as exc:
            logger.error("Failed to generate sample: %s", exc)
            fail_session(session_id, str(exc))
            return api_error(f"Failed to generate sample: {exc}", 500)

        return api_response(
            {
                "session_id": session_id,
                "sample_key": sample_key,
                "sample_name": sample_info["name"],
                "snr_db": sample_info["snr_db"],
                "duration": sample_info["duration_seconds"],
                "noise_type": sample_info["noise_type"],
            },
            status_code=201,
            message=f"Synthetic sample '{sample_info['name']}' generated.",
        )

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: PROCESS AUDIO
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/process/<session_id>", methods=["POST"])
    def process_audio(session_id: str):
        """
        Trigger the DSP processing pipeline for a session.

        This runs the full pipeline:
            1. Load noisy audio x(t)
            2. Apply 4th-order Butterworth LPF → y(t)
            3. Calculate SNR, MSE metrics
            4. Transcribe both x(t) and y(t)
            5. Generate AI summary
            6. Save all results to database

        Parameters
        ----------
        session_id : str
            The session UUID to process.

        Returns
        -------
        JSON
            Complete processing results.
        """
        logger.info("Processing request for session: %s", session_id[:8])

        # Retrieve session
        db_session = get_session(session_id)
        if db_session is None:
            return api_error(f"Session not found: {session_id}", 404)

        if db_session.status == "completed":
            logger.info("Session already completed, returning cached results.")
            return api_response(
                db_session.to_summary_dict(),
                message="Results already available (cached).",
            )

        if db_session.status == "processing":
            return api_response(
                {"session_id": session_id, "status": "processing"},
                message="Processing already in progress.",
            )

        # Mark as processing
        db_session.mark_processing()
        db.session.commit()

        # Determine processing mode
        sample_type = db_session.sample_type
        input_path = db_session.input_audio_path

        if not input_path or not os.path.isfile(input_path):
            # Try to find the file in uploads
            possible_files = list(UPLOAD_DIR.glob(f"input_{session_id}.*"))
            if possible_files:
                input_path = str(possible_files[0])
            else:
                fail_session(session_id, "Input audio file not found.")
                return api_error("Input audio file not found.", 404)

        # Run the processing pipeline
        try:
            if sample_type in ("cafe_noise", "traffic_noise", "office_hum",
                               "construction_noise", "clean_reference"):
                # Synthetic sample - regenerate with clean reference
                sample_info = get_audio_sample_by_key(sample_type)
                if sample_info:
                    results = process_synthetic_audio(
                        sample_key=sample_type,
                        output_dir=str(PROCESSED_DIR),
                        session_id=session_id,
                        duration=sample_info.get("duration_seconds", DEFAULT_DURATION),
                        sample_rate=DEFAULT_SAMPLE_RATE,
                        snr_db=sample_info.get("snr_db", 5.0),
                    )
                else:
                    results = process_uploaded_audio(
                        input_path=input_path,
                        output_dir=str(PROCESSED_DIR),
                        session_id=session_id,
                    )
            else:
                # User upload
                results = process_uploaded_audio(
                    input_path=input_path,
                    output_dir=str(PROCESSED_DIR),
                    session_id=session_id,
                )

            # Check for pipeline failure
            if results.get("status") == "failed":
                error_msg = results.get("error", "Unknown pipeline error")
                fail_session(session_id, error_msg)
                return api_error(f"Processing failed: {error_msg}", 500)

            # Save results to database
            update_session_results(session_id, results)

            # Also save processing logs
            for i, step in enumerate(results.get("steps", []), 1):
                add_processing_log(
                    session_id=session_id,
                    step_number=i,
                    step_name=step.get("step", "unknown"),
                    message=step.get("message", ""),
                    elapsed_seconds=step.get("elapsed", 0.0),
                )

            # Refresh session from DB
            db_session = get_session(session_id)
            response_data = db_session.to_summary_dict()
            response_data["steps"] = results.get("steps", [])

            logger.info(
                "Processing complete for session %s — SNR improved by %.2f dB",
                session_id[:8],
                results.get("snr_improvement", 0.0),
            )

            return api_response(
                response_data,
                message="Processing completed successfully.",
            )

        except Exception as exc:
            error_msg = f"Unexpected error: {exc}"
            logger.error(error_msg, exc_info=True)
            fail_session(session_id, error_msg)
            return api_error(error_msg, 500)

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: GET RESULTS
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/results/<session_id>", methods=["GET"])
    def get_results(session_id: str):
        """
        Retrieve processing results for a session.

        Parameters
        ----------
        session_id : str
            The session UUID.

        Returns
        -------
        JSON
            Full results including metrics, transcriptions, waveforms.
        """
        db_session = get_session(session_id)
        if db_session is None:
            return api_error(f"Session not found: {session_id}", 404)

        response_data = db_session.to_summary_dict()

        # Include processing logs
        response_data["logs"] = get_processing_logs(session_id)

        return api_response(response_data)

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: SERVE AUDIO FILES
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/audio/<audio_type>/<session_id>", methods=["GET"])
    def serve_audio(audio_type: str, session_id: str):
        """
        Serve audio files for playback.

        Parameters
        ----------
        audio_type : str
            Either "input" (noisy x(t)) or "output" (enhanced y(t)).
        session_id : str
            The session UUID.

        Returns
        -------
        File
            The WAV audio file.
        """
        if audio_type not in ("input", "output"):
            return api_error("audio_type must be 'input' or 'output'.", 400)

        db_session = get_session(session_id)
        if db_session is None:
            return api_error(f"Session not found: {session_id}", 404)

        if audio_type == "input":
            audio_path = db_session.input_audio_path
        else:
            audio_path = db_session.output_audio_path

        if not audio_path or not os.path.isfile(audio_path):
            # Try to find the file
            search_dir = UPLOAD_DIR if audio_type == "input" else PROCESSED_DIR
            pattern = f"{audio_type}_{session_id}.*"
            possible_files = list(search_dir.glob(pattern))
            if possible_files:
                audio_path = str(possible_files[0])
            else:
                return api_error(
                    f"Audio file not found for {audio_type}/{session_id}.", 404
                )

        logger.info("Serving audio: %s", audio_path)

        try:
            response = send_file(
                audio_path,
                mimetype="audio/wav",
                as_attachment=False,
                download_name=f"{audio_type}_{session_id[:8]}.wav",
            )
            # Enable range requests for audio seeking
            response.headers["Accept-Ranges"] = "bytes"
            response.headers["Cache-Control"] = "public, max-age=3600"
            return response

        except Exception as exc:
            logger.error("Failed to serve audio: %s", exc)
            return api_error(f"Failed to serve audio file: {exc}", 500)

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: SOURCE CODE VIEWER
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/source-code/<filename>", methods=["GET"])
    def get_source_code(filename: str):
        """
        Serve the source code of project files for the in-app code viewer.

        Parameters
        ----------
        filename : str
            One of: app.py, speech_engine.py, database.py, index.html,
            main.js, style.css

        Returns
        -------
        JSON
            { filename, language, content, line_count }
        """
        if filename not in SOURCE_FILES:
            available = list(SOURCE_FILES.keys())
            return api_error(
                f"Unknown file: '{filename}'. Available: {available}", 404
            )

        file_path = SOURCE_FILES[filename]

        if not file_path.is_file():
            return api_error(f"File not found on disk: {filename}", 404)

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")

            # Determine language for syntax highlighting
            ext_to_lang = {
                ".py": "python",
                ".html": "html",
                ".js": "javascript",
                ".css": "css",
            }
            extension = file_path.suffix.lower()
            language = ext_to_lang.get(extension, "text")

            return api_response({
                "filename": filename,
                "language": language,
                "content": content,
                "line_count": content.count("\n") + 1,
                "file_size": len(content.encode("utf-8")),
            })

        except Exception as exc:
            logger.error("Failed to read source file %s: %s", filename, exc)
            return api_error(f"Failed to read file: {exc}", 500)

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: DOWNLOAD REPORT
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/download-report/<session_id>", methods=["GET"])
    def download_report(session_id: str):
        """
        Generate and serve a downloadable plain-text results report.

        Parameters
        ----------
        session_id : str
            The session UUID.

        Returns
        -------
        File
            Plain text report file.
        """
        db_session = get_session(session_id)
        if db_session is None:
            return api_error(f"Session not found: {session_id}", 404)

        if db_session.status != "completed":
            return api_error(
                "Processing not yet completed. Cannot generate report.", 400
            )

        try:
            report_text = generate_text_report(session_id)

            response = make_response(report_text)
            response.headers["Content-Type"] = "text/plain; charset=utf-8"
            response.headers["Content-Disposition"] = (
                f"attachment; filename=dsp_results_{session_id[:8]}.txt"
            )
            response.headers["Cache-Control"] = "no-cache"

            logger.info("Generated download report for session %s", session_id[:8])
            return response

        except Exception as exc:
            logger.error("Failed to generate report: %s", exc)
            return api_error(f"Failed to generate report: {exc}", 500)

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: GET AVAILABLE SAMPLES
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/samples", methods=["GET"])
    def list_samples():
        """
        List all available quick-test audio samples.

        Returns
        -------
        JSON
            Array of sample objects.
        """
        samples = get_audio_samples(active_only=True)
        return api_response({
            "samples": samples,
            "count": len(samples),
        })

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: GET ALL SESSIONS
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/sessions", methods=["GET"])
    def list_sessions():
        """
        List recent processing sessions.

        Query Parameters
        ----------------
        limit : int
            Maximum number of sessions to return (default: 20).
        status : str
            Filter by status (optional).

        Returns
        -------
        JSON
            Array of session summaries.
        """
        limit = request.args.get("limit", 20, type=int)
        status = request.args.get("status", None)

        sessions = get_all_sessions(limit=limit, status=status)
        return api_response({
            "sessions": [s.to_dict() for s in sessions],
            "count": len(sessions),
        })

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: GET PROCESSING LOGS
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/logs/<session_id>", methods=["GET"])
    def get_logs(session_id: str):
        """
        Retrieve processing logs for the terminal animation.

        Parameters
        ----------
        session_id : str
            The session UUID.

        Returns
        -------
        JSON
            Array of log entries.
        """
        logs = get_processing_logs(session_id)
        return api_response({
            "logs": logs,
            "count": len(logs),
        })

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: PROJECT INFO
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/project-info", methods=["GET"])
    def project_info():
        """
        Return project metadata for the front-end.

        Returns
        -------
        JSON
            Project title, team, SDGs, and mathematical framework info.
        """
        return api_response({
            "title": "Complex Random Process Analysis for Communication Systems with Applications to Speech Enhancement",
            "team": [
                {
                    "name": "Navin Kumar PG",
                    "id": "24BEC1055",
                    "role": "Lead Developer & DSP Engineer",
                },
                {
                    "name": "A.P. Anirudh",
                    "id": "24BEC1158",
                    "role": "Backend Architect & AI Integration",
                },
                {
                    "name": "Kailash N H",
                    "id": "24BEC1546",
                    "role": "Frontend Engineer & Data Visualization",
                },
            ],
            "faculty": {
                "name": "Dr. Kalaivan K",
                "role": "Faculty Guide",
            },
            "institution": "Vellore Institute of Technology (VIT)",
            "sdgs": [
                {
                    "number": 3,
                    "title": "Good Health and Well-being",
                    "description": "Clear communication in telemedicine and healthcare systems.",
                },
                {
                    "number": 4,
                    "title": "Quality Education",
                    "description": "Enhanced audio quality for e-learning and virtual classrooms.",
                },
                {
                    "number": 9,
                    "title": "Industry, Innovation and Infrastructure",
                    "description": "Advanced DSP techniques for next-generation telecommunications.",
                },
                {
                    "number": 10,
                    "title": "Reduced Inequalities",
                    "description": "Accessible communication technology for underserved communities.",
                },
            ],
            "mathematical_framework": {
                "transmitter": "s(t) — Pure modulated signal with modulation index μ and frequency constant k_a",
                "channel": "n(t) — Additive White Gaussian Noise (AWGN) stochastic process",
                "receiver": "x(t) = s(t) + n(t) — Corrupted received signal",
                "processor": "y(t) — Enhanced output via 4th-order Butterworth digital LPF",
            },
            "summary": (
                "This project bridges the gap between theoretical Random Process "
                "mathematics and real-world telecommunications. By actively mitigating "
                "Additive White Gaussian Noise (AWGN) in digital channels—such as VoIP, "
                "video conferencing, and mobile networks—we apply rigorous mathematical "
                "frameworks to recover degraded audio arrays and extract semantic intelligence. "
                "Aligning with UN Sustainable Development Goals (SDGs 3, 4, 9, and 10), "
                "our system provides an interactive, full-stack digital signal processing "
                "environment designed to ensure accessible, reliable, and intelligent communication."
            ),
        })

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: SERVE REPORT PDF (if exists)
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/report-pdf", methods=["GET"])
    def serve_report_pdf():
        """
        Serve the full project report PDF if it exists.

        Looks for a PDF file matching the project title in the
        base directory.

        Returns
        -------
        File or JSON error
        """
        # Search in static/docs/ first, then project root
        search_dirs = [
            BASE_DIR / "static" / "docs",
            BASE_DIR,
        ]
        pdf_patterns = [
            "RP_Project_Report.pdf",
            "Complex Random Process Analysis*.pdf",
            "*.pdf",
        ]

        for search_dir in search_dirs:
            for pattern in pdf_patterns:
                found = list(search_dir.glob(pattern))
                if found:
                    pdf_path = found[0]
                    logger.info("Serving report PDF: %s", pdf_path)
                    return send_file(
                        str(pdf_path),
                        mimetype="application/pdf",
                        as_attachment=False,
                        download_name=pdf_path.name,
                    )

        return api_error(
            "Project report PDF not found. Please place the PDF in the project directory.",
            404,
        )

    # ──────────────────────────────────────────────────────────────────
    #  ROUTE: API DOCUMENTATION
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/docs", methods=["GET"])
    def api_docs():
        """
        Return API endpoint documentation.

        Returns
        -------
        JSON
            List of available endpoints with descriptions.
        """
        endpoints = [
            {
                "method": "GET",
                "path": "/",
                "description": "Serve the Single Page Application",
            },
            {
                "method": "GET",
                "path": "/api/health",
                "description": "Health check endpoint",
            },
            {
                "method": "POST",
                "path": "/api/upload",
                "description": "Upload an audio file for processing",
                "body": "multipart/form-data with 'audio' field",
            },
            {
                "method": "POST",
                "path": "/api/generate-sample",
                "description": "Generate a synthetic test audio sample",
                "body": '{"sample_key": "cafe_noise"}',
            },
            {
                "method": "POST",
                "path": "/api/process/<session_id>",
                "description": "Trigger DSP processing pipeline",
            },
            {
                "method": "GET",
                "path": "/api/results/<session_id>",
                "description": "Retrieve processing results",
            },
            {
                "method": "GET",
                "path": "/api/audio/<type>/<session_id>",
                "description": "Serve input/output audio files",
            },
            {
                "method": "GET",
                "path": "/api/source-code/<filename>",
                "description": "View project source code",
            },
            {
                "method": "GET",
                "path": "/api/download-report/<session_id>",
                "description": "Download text results report",
            },
            {
                "method": "GET",
                "path": "/api/samples",
                "description": "List available test samples",
            },
            {
                "method": "GET",
                "path": "/api/sessions",
                "description": "List recent processing sessions",
            },
            {
                "method": "GET",
                "path": "/api/logs/<session_id>",
                "description": "Get processing logs for terminal animation",
            },
            {
                "method": "GET",
                "path": "/api/project-info",
                "description": "Get project metadata and team info",
            },
            {
                "method": "GET",
                "path": "/api/report-pdf",
                "description": "Download project report PDF",
            },
        ]

        return api_response({
            "endpoints": endpoints,
            "total": len(endpoints),
            "base_url": request.host_url.rstrip("/"),
        })


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def _register_error_handlers(app: Flask) -> None:
    """
    Register custom error handlers for common HTTP errors.

    Parameters
    ----------
    app : Flask
        The Flask application instance.
    """

    @app.errorhandler(400)
    def bad_request(error):
        return api_error("Bad request.", 400)

    @app.errorhandler(404)
    def not_found(error):
        return api_error("Resource not found.", 404)

    @app.errorhandler(405)
    def method_not_allowed(error):
        return api_error("Method not allowed.", 405)

    @app.errorhandler(413)
    def request_too_large(error):
        return api_error(
            f"File too large. Maximum size: {MAX_CONTENT_LENGTH // (1024*1024)} MB.",
            413,
        )

    @app.errorhandler(500)
    def internal_error(error):
        logger.error("Internal server error: %s", error)
        return api_error("Internal server error.", 500)


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

# Create the application instance
app = create_app()

if __name__ == "__main__":
    # ── Print startup banner ──────────────────────────────────────────
    banner = (
        "\n"
        "    ==================================================================\n"
        "    |                                                                |\n"
        "    |   Complex Random Process Analysis for Communication Systems   |\n"
        "    |        with Applications to Speech Enhancement                |\n"
        "    |                                                                |\n"
        "    |   Team:  Navin Kumar PG  (24BEC1055)                          |\n"
        "    |          A.P. Anirudh    (24BEC1158)                          |\n"
        "    |          Kailash N H     (24BEC1546)                          |\n"
        "    |                                                                |\n"
        "    |   Faculty: Dr. Kalaivan K                                     |\n"
        "    |   Institution: Vellore Institute of Technology (VIT)          |\n"
        "    |                                                                |\n"
        "    |   Server starting on http://127.0.0.1:5000                    |\n"
        "    |                                                                |\n"
        "    ==================================================================\n"
    )
    print(banner)

    # ── Run the development server ────────────────────────────────────
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=True,
        threaded=True,
    )
