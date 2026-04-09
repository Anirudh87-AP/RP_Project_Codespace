"""
================================================================================
DATABASE.PY — Database Models & Management Layer
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
    This module defines the SQLAlchemy ORM models and helper utilities for
    persisting Digital Signal Processing (DSP) results in a local SQLite
    database.  Every processing session is stored as a single row in the
    `processing_results` table, capturing:

        • Session metadata (UUID, timestamps)
        • Signal-quality metrics (input SNR, output SNR, MSE)
        • Transcription results (noisy input text, clean output text)
        • AI-generated summary of the enhanced speech
        • Down-sampled waveform arrays serialised as JSON blobs for the
          Chart.js front-end visualisation

    The module exposes a thin repository / data-access-object (DAO) pattern
    so that `app.py` never touches raw SQL or sessions directly.

Revision History:
    2026-04-08  Initial creation
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────────────────────────
import json
import uuid
import logging
from datetime import datetime, timezone

# ──────────────────────────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────────────────────────
from flask_sqlalchemy import SQLAlchemy  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Module-Level Logger
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ──────────────────────────────────────────────────────────────────────────────
# SQLAlchemy Instance  (shared across the application via init_app)
# ──────────────────────────────────────────────────────────────────────────────
db = SQLAlchemy()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — ORM MODEL : ProcessingResult
# ══════════════════════════════════════════════════════════════════════════════

class ProcessingResult(db.Model):
    """
    Represents a single audio-processing session.

    Each time a user uploads (or generates) an audio sample and runs the DSP
    pipeline, the system creates one row that stores **all** outputs of that
    run — from raw numeric metrics to AI-synthesised text.

    Columns
    -------
    id : int
        Auto-incrementing primary key.
    session_id : str
        A UUID-4 string that the front-end uses to poll / fetch results.
    created_at : datetime
        UTC timestamp recorded when the row is first inserted.
    updated_at : datetime
        UTC timestamp updated whenever the row is modified.
    status : str
        Workflow state — one of: ``pending``, ``processing``, ``completed``,
        ``failed``.
    filename : str
        Original filename supplied by the user (or generated label).
    sample_type : str
        Descriptor for the kind of sample: ``upload``, ``cafe_noise``,
        ``traffic_noise``, ``clean_speech``, ``custom``.
    duration_seconds : float
        Length of the input audio in seconds.
    sample_rate : int
        Sample rate of the processed audio in Hz.
    input_snr_db : float
        Signal-to-Noise Ratio of the *noisy* input  x(t)  in decibels.
    output_snr_db : float
        Signal-to-Noise Ratio of the *enhanced* output  y(t)  in decibels.
    mse : float
        Mean Square Error between the clean reference and the enhanced signal.
    snr_improvement_db : float
        The gain in SNR achieved by the Butterworth filter (output − input).
    filter_order : int
        Order of the Butterworth filter used (default 4).
    filter_cutoff_hz : float
        Cutoff frequency of the low-pass filter in Hz.
    input_text : str
        Raw transcription of the noisy audio  x(t)  (often garbled).
    output_text : str
        Transcription of the enhanced audio  y(t).
    ai_summary : str
        AI-generated concise summary of the output transcription.
    input_waveform_json : str
        JSON-encoded list of 100 float values — down-sampled  x(t)  for the
        Chart.js visualisation on Page 5.
    output_waveform_json : str
        JSON-encoded list of 100 float values — down-sampled  y(t)  for the
        Chart.js visualisation on Page 5.
    input_audio_path : str
        Server-side file path to the noisy input audio file.
    output_audio_path : str
        Server-side file path to the enhanced output audio file.
    processing_time_seconds : float
        Wall-clock time taken to run the full DSP + ASR pipeline.
    error_message : str | None
        If the session failed, the human-readable error description.
    """

    # ── Table name ────────────────────────────────────────────────────────
    __tablename__ = "processing_results"

    # ── Primary key & identifiers ─────────────────────────────────────────
    id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True,
        doc="Auto-incrementing primary key.",
    )
    session_id = db.Column(
        db.String(36),
        unique=True,
        nullable=False,
        index=True,
        default=lambda: str(uuid.uuid4()),
        doc="UUID-4 session identifier used by the front-end.",
    )

    # ── Timestamps ────────────────────────────────────────────────────────
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        doc="UTC creation timestamp.",
    )
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        doc="UTC last-update timestamp.",
    )

    # ── Workflow state ────────────────────────────────────────────────────
    status = db.Column(
        db.String(20),
        nullable=False,
        default="pending",
        doc="One of: pending | processing | completed | failed.",
    )

    # ── Audio metadata ────────────────────────────────────────────────────
    filename = db.Column(
        db.String(256),
        nullable=True,
        default="unknown.wav",
        doc="Original filename of the uploaded audio.",
    )
    sample_type = db.Column(
        db.String(50),
        nullable=True,
        default="upload",
        doc="Source category: upload | cafe_noise | traffic_noise | clean_speech | custom.",
    )
    duration_seconds = db.Column(
        db.Float,
        nullable=True,
        default=0.0,
        doc="Duration of the input audio in seconds.",
    )
    sample_rate = db.Column(
        db.Integer,
        nullable=True,
        default=16000,
        doc="Sample rate used during processing (Hz).",
    )

    # ── DSP Metrics ──────────────────────────────────────────────────────
    input_snr_db = db.Column(
        db.Float,
        nullable=True,
        doc="SNR of the noisy input x(t) in dB.",
    )
    output_snr_db = db.Column(
        db.Float,
        nullable=True,
        doc="SNR of the enhanced output y(t) in dB.",
    )
    mse = db.Column(
        db.Float,
        nullable=True,
        doc="Mean Square Error between clean and enhanced signals.",
    )
    snr_improvement_db = db.Column(
        db.Float,
        nullable=True,
        doc="SNR improvement (output_snr − input_snr).",
    )

    # ── Filter parameters ────────────────────────────────────────────────
    filter_order = db.Column(
        db.Integer,
        nullable=True,
        default=4,
        doc="Order of the Butterworth low-pass filter.",
    )
    filter_cutoff_hz = db.Column(
        db.Float,
        nullable=True,
        doc="Cutoff frequency of the Butterworth LPF (Hz).",
    )

    # ── Transcription & Summary ──────────────────────────────────────────
    input_text = db.Column(
        db.Text,
        nullable=True,
        default="",
        doc="ASR transcription of the noisy input x(t).",
    )
    output_text = db.Column(
        db.Text,
        nullable=True,
        default="",
        doc="ASR transcription of the enhanced output y(t).",
    )
    ai_summary = db.Column(
        db.Text,
        nullable=True,
        default="",
        doc="AI-generated summary of the output transcription.",
    )

    # ── Waveform data (JSON) ─────────────────────────────────────────────
    input_waveform_json = db.Column(
        db.Text,
        nullable=True,
        doc="JSON list of 100 floats — down-sampled x(t).",
    )
    output_waveform_json = db.Column(
        db.Text,
        nullable=True,
        doc="JSON list of 100 floats — down-sampled y(t).",
    )

    # ── File paths ───────────────────────────────────────────────────────
    input_audio_path = db.Column(
        db.String(512),
        nullable=True,
        doc="Server path to the noisy input WAV file.",
    )
    output_audio_path = db.Column(
        db.String(512),
        nullable=True,
        doc="Server path to the enhanced output WAV file.",
    )

    # ── Processing metadata ──────────────────────────────────────────────
    processing_time_seconds = db.Column(
        db.Float,
        nullable=True,
        default=0.0,
        doc="Wall-clock time of the processing pipeline (seconds).",
    )
    error_message = db.Column(
        db.Text,
        nullable=True,
        doc="Error description if status == 'failed'.",
    )

    # ──────────────────────────────────────────────────────────────────────
    #  Magic / dunder methods
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Concise developer-friendly representation."""
        return (
            f"<ProcessingResult "
            f"id={self.id} "
            f"session={self.session_id[:8]}… "
            f"status={self.status}>"
        )

    def __str__(self) -> str:
        """Human-readable summary string."""
        return (
            f"Session {self.session_id} | "
            f"Status: {self.status} | "
            f"Input SNR: {self.input_snr_db} dB | "
            f"Output SNR: {self.output_snr_db} dB | "
            f"MSE: {self.mse}"
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Serialisation helpers
    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialise the model instance to a plain Python dictionary suitable
        for JSON responses.

        Returns
        -------
        dict
            All column values with waveform JSON decoded to native lists.
        """
        return {
            "id": self.id,
            "session_id": self.session_id,
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
            "updated_at": (
                self.updated_at.isoformat() if self.updated_at else None
            ),
            "status": self.status,
            "filename": self.filename,
            "sample_type": self.sample_type,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "input_snr_db": self.input_snr_db,
            "output_snr_db": self.output_snr_db,
            "mse": self.mse,
            "snr_improvement_db": self.snr_improvement_db,
            "filter_order": self.filter_order,
            "filter_cutoff_hz": self.filter_cutoff_hz,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "ai_summary": self.ai_summary,
            "input_waveform": self._decode_waveform(self.input_waveform_json),
            "output_waveform": self._decode_waveform(self.output_waveform_json),
            "input_audio_path": self.input_audio_path,
            "output_audio_path": self.output_audio_path,
            "processing_time_seconds": self.processing_time_seconds,
            "error_message": self.error_message,
        }

    def to_summary_dict(self) -> dict:
        """
        Lightweight serialisation containing only the fields needed by
        the Page 5 dashboard.

        Returns
        -------
        dict
            Subset of fields for the analytics view.
        """
        return {
            "session_id": self.session_id,
            "status": self.status,
            "filename": self.filename,
            "sample_type": self.sample_type,
            "duration_seconds": self.duration_seconds,
            "input_snr_db": self.input_snr_db,
            "output_snr_db": self.output_snr_db,
            "mse": self.mse,
            "snr_improvement_db": self.snr_improvement_db,
            "filter_order": self.filter_order,
            "filter_cutoff_hz": self.filter_cutoff_hz,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "ai_summary": self.ai_summary,
            "input_waveform": self._decode_waveform(self.input_waveform_json),
            "output_waveform": self._decode_waveform(self.output_waveform_json),
            "processing_time_seconds": self.processing_time_seconds,
        }

    def to_metrics_dict(self) -> dict:
        """
        Serialise only the numeric DSP metrics — useful for the
        'Download Report' feature.

        Returns
        -------
        dict
            Numeric metrics only.
        """
        return {
            "input_snr_db": self.input_snr_db,
            "output_snr_db": self.output_snr_db,
            "mse": self.mse,
            "snr_improvement_db": self.snr_improvement_db,
            "filter_order": self.filter_order,
            "filter_cutoff_hz": self.filter_cutoff_hz,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "processing_time_seconds": self.processing_time_seconds,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_waveform(json_str: str | None) -> list:
        """
        Safely decode a JSON string into a Python list.

        Parameters
        ----------
        json_str : str or None
            JSON-encoded list of floats.

        Returns
        -------
        list
            Decoded list, or empty list on failure.
        """
        if not json_str:
            return []
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            logger.warning("Waveform JSON decoded to non-list type: %s", type(data))
            return []
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("Failed to decode waveform JSON: %s", exc)
            return []

    # ──────────────────────────────────────────────────────────────────────
    #  Update helpers (used by app.py routes)
    # ──────────────────────────────────────────────────────────────────────

    def mark_processing(self) -> None:
        """Transition the session to the 'processing' state."""
        self.status = "processing"
        self.updated_at = datetime.now(timezone.utc)
        logger.info("Session %s → processing", self.session_id[:8])

    def mark_completed(
        self,
        input_snr: float,
        output_snr: float,
        mse: float,
        input_text: str,
        output_text: str,
        ai_summary: str,
        input_waveform: list,
        output_waveform: list,
        input_audio_path: str,
        output_audio_path: str,
        processing_time: float,
        duration_seconds: float = 0.0,
        sample_rate: int = 16000,
        filter_order: int = 4,
        filter_cutoff_hz: float = 4000.0,
    ) -> None:
        """
        Populate all result fields and transition to 'completed'.

        Parameters
        ----------
        input_snr : float
            Measured SNR of the noisy input.
        output_snr : float
            Measured SNR of the enhanced output.
        mse : float
            Mean Square Error metric.
        input_text : str
            ASR transcription of the noisy audio.
        output_text : str
            ASR transcription of the enhanced audio.
        ai_summary : str
            AI summary generated from the clean transcription.
        input_waveform : list[float]
            100-point down-sampled x(t).
        output_waveform : list[float]
            100-point down-sampled y(t).
        input_audio_path : str
            Path to the noisy audio file on disk.
        output_audio_path : str
            Path to the enhanced audio file on disk.
        processing_time : float
            Total pipeline duration in seconds.
        duration_seconds : float
            Length of the audio in seconds.
        sample_rate : int
            Audio sample rate in Hz.
        filter_order : int
            Butterworth filter order used.
        filter_cutoff_hz : float
            Butterworth cutoff frequency in Hz.
        """
        self.status = "completed"
        self.input_snr_db = round(input_snr, 4)
        self.output_snr_db = round(output_snr, 4)
        self.mse = round(mse, 8)
        self.snr_improvement_db = round(output_snr - input_snr, 4)
        self.input_text = input_text
        self.output_text = output_text
        self.ai_summary = ai_summary
        self.input_waveform_json = json.dumps(
            [round(float(v), 6) for v in input_waveform]
        )
        self.output_waveform_json = json.dumps(
            [round(float(v), 6) for v in output_waveform]
        )
        self.input_audio_path = input_audio_path
        self.output_audio_path = output_audio_path
        self.processing_time_seconds = round(processing_time, 4)
        self.duration_seconds = round(duration_seconds, 4)
        self.sample_rate = sample_rate
        self.filter_order = filter_order
        self.filter_cutoff_hz = filter_cutoff_hz
        self.updated_at = datetime.now(timezone.utc)
        logger.info(
            "Session %s → completed  |  SNR Improvement: %.2f dB",
            self.session_id[:8],
            self.snr_improvement_db,
        )

    def mark_failed(self, error_message: str) -> None:
        """
        Transition the session to the 'failed' state with an error message.

        Parameters
        ----------
        error_message : str
            Human-readable description of what went wrong.
        """
        self.status = "failed"
        self.error_message = error_message
        self.updated_at = datetime.now(timezone.utc)
        logger.error(
            "Session %s → failed: %s",
            self.session_id[:8],
            error_message,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — ORM MODEL : ProcessingLog
# ══════════════════════════════════════════════════════════════════════════════

class ProcessingLog(db.Model):
    """
    Stores individual log entries emitted during a processing session.

    This provides a detailed audit trail that can be replayed on the
    front-end's "terminal" loading animation (Page 4).  Each row captures
    a single step or status message from the DSP pipeline.

    Columns
    -------
    id : int
        Auto-incrementing primary key.
    session_id : str
        Foreign key linking to ``ProcessingResult.session_id``.
    timestamp : datetime
        UTC time the log entry was created.
    step_number : int
        Sequential step number within the session (1-based).
    step_name : str
        Short label for the pipeline step (e.g., "butterworth_filter").
    message : str
        Detailed human-readable message.
    level : str
        Log level — one of: ``info``, ``warning``, ``error``, ``debug``.
    elapsed_seconds : float
        Seconds elapsed since the start of the pipeline.
    """

    __tablename__ = "processing_logs"

    id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True,
    )
    session_id = db.Column(
        db.String(36),
        db.ForeignKey("processing_results.session_id"),
        nullable=False,
        index=True,
    )
    timestamp = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    step_number = db.Column(
        db.Integer,
        nullable=False,
        default=0,
    )
    step_name = db.Column(
        db.String(100),
        nullable=False,
        default="unknown",
    )
    message = db.Column(
        db.Text,
        nullable=False,
        default="",
    )
    level = db.Column(
        db.String(10),
        nullable=False,
        default="info",
    )
    elapsed_seconds = db.Column(
        db.Float,
        nullable=True,
        default=0.0,
    )

    def __repr__(self) -> str:
        return (
            f"<ProcessingLog "
            f"session={self.session_id[:8]}… "
            f"step={self.step_number} "
            f"{self.step_name}>"
        )

    def to_dict(self) -> dict:
        """Serialise to a dictionary for JSON responses."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": (
                self.timestamp.isoformat() if self.timestamp else None
            ),
            "step_number": self.step_number,
            "step_name": self.step_name,
            "message": self.message,
            "level": self.level,
            "elapsed_seconds": self.elapsed_seconds,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — ORM MODEL : AudioSample
# ══════════════════════════════════════════════════════════════════════════════

class AudioSample(db.Model):
    """
    Catalog of pre-built / synthetic audio samples available for quick-test.

    The front-end's "Quick-Test Sample" buttons on Page 3 pull from this
    table to display available presets.

    Columns
    -------
    id : int
        Primary key.
    name : str
        Display name shown on the UI button (e.g., "Cafe Noise (AWGN)").
    description : str
        Tooltip / subtitle text.
    sample_key : str
        Machine-readable key used in API calls (e.g., "cafe_noise").
    noise_type : str
        Type of noise profile: ``awgn``, ``heavy_awgn``, ``pink``, ``none``.
    snr_db : float
        Target SNR when generating this sample.
    duration_seconds : float
        Length of the generated sample.
    frequency_hz : float
        Carrier / base frequency (for synthetic signals).
    is_active : bool
        Whether this sample is available for selection.
    icon : str
        Emoji or icon identifier for the UI card.
    sort_order : int
        Display order on the UI.
    """

    __tablename__ = "audio_samples"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True, default="")
    sample_key = db.Column(
        db.String(50), unique=True, nullable=False, index=True
    )
    noise_type = db.Column(db.String(30), nullable=False, default="awgn")
    snr_db = db.Column(db.Float, nullable=False, default=5.0)
    duration_seconds = db.Column(db.Float, nullable=False, default=3.0)
    frequency_hz = db.Column(db.Float, nullable=True, default=440.0)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    icon = db.Column(db.String(10), nullable=True, default="🔊")
    sort_order = db.Column(db.Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return f"<AudioSample {self.sample_key} snr={self.snr_db}dB>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "sample_key": self.sample_key,
            "noise_type": self.noise_type,
            "snr_db": self.snr_db,
            "duration_seconds": self.duration_seconds,
            "frequency_hz": self.frequency_hz,
            "is_active": self.is_active,
            "icon": self.icon,
            "sort_order": self.sort_order,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — DATABASE INITIALIZATION & SEED DATA
# ══════════════════════════════════════════════════════════════════════════════

# Default quick-test audio samples to seed the database with.
DEFAULT_AUDIO_SAMPLES = [
    {
        "name": "Cafe Noise (AWGN)",
        "description": (
            "Simulates a VoIP call in a busy café environment. "
            "Additive White Gaussian Noise at moderate SNR (5 dB) "
            "overlays a synthetic speech-like signal. Ideal for "
            "demonstrating the Butterworth filter's ability to "
            "recover the underlying message signal s(t)."
        ),
        "sample_key": "cafe_noise",
        "noise_type": "awgn",
        "snr_db": 5.0,
        "duration_seconds": 4.0,
        "frequency_hz": 440.0,
        "is_active": True,
        "icon": "☕",
        "sort_order": 1,
    },
    {
        "name": "Traffic Noise (Heavy AWGN)",
        "description": (
            "Simulates a mobile phone call near heavy traffic. "
            "The stochastic noise n(t) is severely degrading with "
            "SNR as low as 0 dB, creating a challenging environment "
            "for the 4th-order Butterworth low-pass filter and "
            "downstream ASR engine."
        ),
        "sample_key": "traffic_noise",
        "noise_type": "heavy_awgn",
        "snr_db": 0.0,
        "duration_seconds": 4.0,
        "frequency_hz": 440.0,
        "is_active": True,
        "icon": "🚗",
        "sort_order": 2,
    },
    {
        "name": "Office Hum (Light AWGN)",
        "description": (
            "A gentle background hum typical of open-plan offices. "
            "SNR is relatively high (15 dB), making this a gentle "
            "test case where the filter should achieve near-perfect "
            "envelope recovery."
        ),
        "sample_key": "office_hum",
        "noise_type": "awgn",
        "snr_db": 15.0,
        "duration_seconds": 4.0,
        "frequency_hz": 440.0,
        "is_active": True,
        "icon": "🏢",
        "sort_order": 3,
    },
    {
        "name": "Construction Site (Extreme AWGN)",
        "description": (
            "Extreme noise environment with SNR at −5 dB. "
            "The noise power actually exceeds the signal power. "
            "This stress-tests the DSP pipeline and demonstrates "
            "the limits of linear filtering for speech recovery."
        ),
        "sample_key": "construction_noise",
        "noise_type": "heavy_awgn",
        "snr_db": -5.0,
        "duration_seconds": 4.0,
        "frequency_hz": 440.0,
        "is_active": True,
        "icon": "🏗️",
        "sort_order": 4,
    },
    {
        "name": "Clean Reference (No Noise)",
        "description": (
            "A clean synthetic signal with zero noise added. "
            "Use this as a control / baseline to verify that the "
            "filter passes the signal through with minimal "
            "distortion when noise is absent."
        ),
        "sample_key": "clean_reference",
        "noise_type": "none",
        "snr_db": 100.0,
        "duration_seconds": 4.0,
        "frequency_hz": 440.0,
        "is_active": True,
        "icon": "✨",
        "sort_order": 5,
    },
]


def init_database(app) -> None:
    """
    Initialise the database: create all tables and seed default data.

    This function should be called once during application startup,
    typically inside the Flask application factory or after
    ``db.init_app(app)`` has been invoked.

    Parameters
    ----------
    app : flask.Flask
        The Flask application instance with database configuration.
    """
    with app.app_context():
        # Create all tables that don't exist yet
        db.create_all()
        logger.info("Database tables created / verified.")

        # Seed audio samples if the table is empty
        _seed_audio_samples()

        logger.info("Database initialization complete.")


def _seed_audio_samples() -> None:
    """
    Insert default audio sample presets if they don't already exist.

    Uses an upsert-like pattern: for each default sample, check if a
    row with the same ``sample_key`` exists.  If not, insert it.
    """
    existing_keys = {
        row.sample_key
        for row in AudioSample.query.with_entities(AudioSample.sample_key).all()
    }

    inserted_count = 0
    for sample_data in DEFAULT_AUDIO_SAMPLES:
        if sample_data["sample_key"] not in existing_keys:
            sample = AudioSample(**sample_data)
            db.session.add(sample)
            inserted_count += 1
            logger.debug("Seeding audio sample: %s", sample_data["sample_key"])

    if inserted_count > 0:
        db.session.commit()
        logger.info("Seeded %d audio sample(s) into the database.", inserted_count)
    else:
        logger.debug("Audio samples already seeded — skipping.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — DATA ACCESS OBJECT (DAO) FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_session(
    filename: str = "unknown.wav",
    sample_type: str = "upload",
) -> ProcessingResult:
    """
    Create a new processing session in the database.

    Parameters
    ----------
    filename : str
        Original filename of the uploaded audio.
    sample_type : str
        Source category identifier.

    Returns
    -------
    ProcessingResult
        The newly created (and committed) database row.
    """
    session = ProcessingResult(
        session_id=str(uuid.uuid4()),
        filename=filename,
        sample_type=sample_type,
        status="pending",
    )
    db.session.add(session)
    db.session.commit()
    logger.info(
        "Created new session %s for '%s' (%s)",
        session.session_id[:8],
        filename,
        sample_type,
    )
    return session


def get_session(session_id: str) -> ProcessingResult | None:
    """
    Retrieve a processing session by its UUID.

    Parameters
    ----------
    session_id : str
        The UUID-4 session identifier.

    Returns
    -------
    ProcessingResult or None
        The matching row, or ``None`` if not found.
    """
    result = ProcessingResult.query.filter_by(session_id=session_id).first()
    if result is None:
        logger.warning("Session not found: %s", session_id)
    return result


def get_all_sessions(
    limit: int = 50,
    status: str | None = None,
) -> list[ProcessingResult]:
    """
    Retrieve recent processing sessions, optionally filtered by status.

    Parameters
    ----------
    limit : int
        Maximum number of rows to return (default 50).
    status : str or None
        If provided, only return sessions with this status.

    Returns
    -------
    list[ProcessingResult]
        List of matching sessions ordered by creation time (newest first).
    """
    query = ProcessingResult.query
    if status:
        query = query.filter_by(status=status)
    return query.order_by(ProcessingResult.created_at.desc()).limit(limit).all()


def update_session_results(session_id: str, results: dict) -> bool:
    """
    Update a session with processing results.

    Parameters
    ----------
    session_id : str
        Session to update.
    results : dict
        Dictionary containing all result fields. Expected keys:
        ``input_snr``, ``output_snr``, ``mse``, ``input_text``,
        ``output_text``, ``ai_summary``, ``input_waveform``,
        ``output_waveform``, ``input_audio_path``, ``output_audio_path``,
        ``processing_time``, ``duration_seconds``, ``sample_rate``,
        ``filter_order``, ``filter_cutoff_hz``.

    Returns
    -------
    bool
        ``True`` if the update succeeded, ``False`` if the session was
        not found.
    """
    session = get_session(session_id)
    if session is None:
        return False

    session.mark_completed(
        input_snr=results.get("input_snr", 0.0),
        output_snr=results.get("output_snr", 0.0),
        mse=results.get("mse", 0.0),
        input_text=results.get("input_text", ""),
        output_text=results.get("output_text", ""),
        ai_summary=results.get("ai_summary", ""),
        input_waveform=results.get("input_waveform", []),
        output_waveform=results.get("output_waveform", []),
        input_audio_path=results.get("input_audio_path", ""),
        output_audio_path=results.get("output_audio_path", ""),
        processing_time=results.get("processing_time", 0.0),
        duration_seconds=results.get("duration_seconds", 0.0),
        sample_rate=results.get("sample_rate", 16000),
        filter_order=results.get("filter_order", 4),
        filter_cutoff_hz=results.get("filter_cutoff_hz", 4000.0),
    )
    db.session.commit()
    return True


def fail_session(session_id: str, error_message: str) -> bool:
    """
    Mark a session as failed.

    Parameters
    ----------
    session_id : str
        Session to mark.
    error_message : str
        Description of the failure.

    Returns
    -------
    bool
        ``True`` if update succeeded, ``False`` if session not found.
    """
    session = get_session(session_id)
    if session is None:
        return False
    session.mark_failed(error_message)
    db.session.commit()
    return True


def add_processing_log(
    session_id: str,
    step_number: int,
    step_name: str,
    message: str,
    level: str = "info",
    elapsed_seconds: float = 0.0,
) -> ProcessingLog:
    """
    Append a log entry to a processing session.

    Parameters
    ----------
    session_id : str
        The session this log belongs to.
    step_number : int
        Sequential step index (1-based).
    step_name : str
        Machine-friendly step label.
    message : str
        Human-readable log message.
    level : str
        Log level (info, warning, error, debug).
    elapsed_seconds : float
        Time elapsed since pipeline start.

    Returns
    -------
    ProcessingLog
        The newly created log entry.
    """
    log_entry = ProcessingLog(
        session_id=session_id,
        step_number=step_number,
        step_name=step_name,
        message=message,
        level=level,
        elapsed_seconds=elapsed_seconds,
    )
    db.session.add(log_entry)
    db.session.commit()
    return log_entry


def get_processing_logs(session_id: str) -> list[dict]:
    """
    Retrieve all log entries for a session, ordered by step number.

    Parameters
    ----------
    session_id : str
        The session to query.

    Returns
    -------
    list[dict]
        Serialised log entries.
    """
    logs = (
        ProcessingLog.query
        .filter_by(session_id=session_id)
        .order_by(ProcessingLog.step_number.asc())
        .all()
    )
    return [log.to_dict() for log in logs]


def get_audio_samples(active_only: bool = True) -> list[dict]:
    """
    Retrieve available quick-test audio samples.

    Parameters
    ----------
    active_only : bool
        If ``True``, only return samples where ``is_active`` is ``True``.

    Returns
    -------
    list[dict]
        Serialised sample records ordered by ``sort_order``.
    """
    query = AudioSample.query
    if active_only:
        query = query.filter_by(is_active=True)
    samples = query.order_by(AudioSample.sort_order.asc()).all()
    return [s.to_dict() for s in samples]


def get_audio_sample_by_key(sample_key: str) -> dict | None:
    """
    Retrieve a single audio sample by its machine-readable key.

    Parameters
    ----------
    sample_key : str
        e.g., ``"cafe_noise"``, ``"traffic_noise"``.

    Returns
    -------
    dict or None
        The sample data, or ``None`` if not found.
    """
    sample = AudioSample.query.filter_by(sample_key=sample_key).first()
    return sample.to_dict() if sample else None


def delete_session(session_id: str) -> bool:
    """
    Delete a processing session and its associated logs.

    Parameters
    ----------
    session_id : str
        Session to delete.

    Returns
    -------
    bool
        ``True`` if the session was found and deleted.
    """
    session = get_session(session_id)
    if session is None:
        return False

    # Delete associated logs first
    ProcessingLog.query.filter_by(session_id=session_id).delete()
    db.session.delete(session)
    db.session.commit()
    logger.info("Deleted session %s and associated logs.", session_id[:8])
    return True


def get_session_count(status: str | None = None) -> int:
    """
    Count total sessions, optionally filtered by status.

    Parameters
    ----------
    status : str or None
        Filter by this status if provided.

    Returns
    -------
    int
        Number of matching sessions.
    """
    query = ProcessingResult.query
    if status:
        query = query.filter_by(status=status)
    return query.count()


def get_average_snr_improvement() -> float:
    """
    Calculate the average SNR improvement across all completed sessions.

    Returns
    -------
    float
        Mean SNR improvement in dB, or 0.0 if no completed sessions exist.
    """
    results = ProcessingResult.query.filter_by(status="completed").all()
    if not results:
        return 0.0
    improvements = [
        r.snr_improvement_db for r in results if r.snr_improvement_db is not None
    ]
    return sum(improvements) / len(improvements) if improvements else 0.0


def generate_text_report(session_id: str) -> str:
    """
    Generate a formatted plain-text report for downloading.

    Parameters
    ----------
    session_id : str
        The session to generate a report for.

    Returns
    -------
    str
        Formatted report text, or an error message if the session
        is not found.
    """
    session = get_session(session_id)
    if session is None:
        return "ERROR: Session not found."

    divider = "=" * 72
    thin_divider = "-" * 72

    report_lines = [
        divider,
        "  COMPLEX RANDOM PROCESS ANALYSIS — RESULTS REPORT",
        "  Speech Enhancement via 4th-Order Butterworth LPF",
        divider,
        "",
        f"  Session ID    : {session.session_id}",
        f"  Filename      : {session.filename}",
        f"  Sample Type   : {session.sample_type}",
        f"  Processed At  : {session.updated_at}",
        f"  Duration      : {session.duration_seconds:.2f} seconds",
        f"  Sample Rate   : {session.sample_rate} Hz",
        "",
        divider,
        "  SIGNAL QUALITY METRICS",
        divider,
        "",
        f"  Input SNR  (x(t))  : {session.input_snr_db:.4f} dB",
        f"  Output SNR (y(t))  : {session.output_snr_db:.4f} dB",
        f"  SNR Improvement    : {session.snr_improvement_db:.4f} dB",
        f"  Mean Square Error  : {session.mse:.8f}",
        "",
        f"  Filter Order       : {session.filter_order}",
        f"  Filter Cutoff      : {session.filter_cutoff_hz:.1f} Hz",
        f"  Processing Time    : {session.processing_time_seconds:.4f} s",
        "",
        divider,
        "  SPEECH TRANSCRIPTION — INPUT x(t) (Noisy Audio)",
        divider,
        "",
        f"  {session.input_text or '[No transcription available]'}",
        "",
        divider,
        "  SPEECH TRANSCRIPTION — OUTPUT y(t) (Enhanced Audio)",
        divider,
        "",
        f"  {session.output_text or '[No transcription available]'}",
        "",
        divider,
        "  AI-GENERATED SUMMARY",
        divider,
        "",
        f"  {session.ai_summary or '[No summary available]'}",
        "",
        divider,
        "  PROJECT INFORMATION",
        divider,
        "",
        "  Project : Complex Random Process Analysis for Communication",
        "            Systems with Applications to Speech Enhancement",
        "",
        "  Team    : Navin Kumar PG  (24BEC1055)",
        "            A.P. Anirudh    (24BEC1158)",
        "            Kailash N H     (24BEC1546)",
        "",
        "  Faculty : Dr. Kalaivan K",
        "",
        "  Institution : Vellore Institute of Technology (VIT)",
        "",
        thin_divider,
        "  Mathematical Framework:",
        "    • Transmitter output : s(t) — pure modulated signal",
        "    • Channel noise      : n(t) — Additive White Gaussian Noise",
        "    • Receiver input     : x(t) = s(t) + n(t)",
        "    • Enhanced output    : y(t) — recovered via Butterworth LPF",
        thin_divider,
        "",
        "  SDG Alignment: Goals 3, 4, 9, 10",
        "",
        divider,
        f"  Report generated at {datetime.now(timezone.utc).isoformat()}",
        divider,
    ]

    return "\n".join(report_lines)


# ══════════════════════════════════════════════════════════════════════════════
#  END OF DATABASE.PY
# ══════════════════════════════════════════════════════════════════════════════
