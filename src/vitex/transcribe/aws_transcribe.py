"""Amazon Transcribe backend (optional, in-stack with Bedrock).

Uploads the audio to a scratch S3 bucket, runs an async transcription job,
polls for completion, then groups the per-word results into sentence-like
:class:`Segment` objects. Requires a writable S3 bucket.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import uuid
from pathlib import Path

from vitex.errors import DependencyError, TranscriptionError
from vitex.models import Segment

log = logging.getLogger("vitex")

# Start a new segment when the gap since the last word exceeds this, or after
# sentence-ending punctuation. Keeps segments short enough for tight alignment.
_MAX_GAP_SECONDS = 1.2
_MAX_SEGMENT_SECONDS = 12.0
_SENTENCE_END = {".", "?", "!"}


class AWSTranscriber:
    def __init__(self, bucket: str, region: str, poll_seconds: float = 10.0):
        self.bucket = bucket
        self.region = region
        self.poll_seconds = poll_seconds

    def _client(self, service: str):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - import guard
            raise DependencyError(
                "boto3 is required for the AWS transcriber. Install with:\n"
                "  uv pip install -e '.[aws]'"
            ) from exc
        return boto3.client(service, region_name=self.region)

    def transcribe(self, audio_path: Path) -> list[Segment]:
        s3 = self._client("s3")
        transcribe = self._client("transcribe")

        job = f"vitex-{uuid.uuid4().hex[:12]}"
        key = f"vitex/{job}/{audio_path.name}"
        log.info("Uploading audio to s3://%s/%s", self.bucket, key)
        s3.upload_file(str(audio_path), self.bucket, key)

        media_uri = f"s3://{self.bucket}/{key}"
        transcribe.start_transcription_job(
            TranscriptionJobName=job,
            Media={"MediaFileUri": media_uri},
            MediaFormat="wav",
            IdentifyLanguage=True,
        )
        log.info("Started Transcribe job %s; polling...", job)

        while True:
            desc = transcribe.get_transcription_job(TranscriptionJobName=job)
            status = desc["TranscriptionJob"]["TranscriptionJobStatus"]
            if status == "COMPLETED":
                uri = desc["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                break
            if status == "FAILED":
                reason = desc["TranscriptionJob"].get("FailureReason", "unknown")
                raise TranscriptionError(f"Transcribe job failed: {reason}")
            time.sleep(self.poll_seconds)

        with urllib.request.urlopen(uri) as resp:  # nosec - AWS-signed URL
            payload = json.loads(resp.read().decode("utf-8"))

        segments = self._group_items(payload)
        if not segments:
            raise TranscriptionError("Transcribe returned no usable items.")
        log.info("Transcribed %d segments via Amazon Transcribe", len(segments))
        return segments

    @staticmethod
    def _group_items(payload: dict) -> list[Segment]:
        """Merge per-word items into segments on pauses/punctuation."""
        items = payload.get("results", {}).get("items", [])
        segments: list[Segment] = []
        words: list[str] = []
        seg_start: float | None = None
        last_end: float | None = None

        def flush(end: float) -> None:
            nonlocal words, seg_start
            if words and seg_start is not None:
                text = "".join(
                    (" " + w if not (w in _SENTENCE_END or w == ",") else w) for w in words
                ).strip()
                segments.append(Segment(start=seg_start, end=end, text=text))
            words = []
            seg_start = None

        for item in items:
            content = item["alternatives"][0]["content"]
            if item["type"] == "punctuation":
                words.append(content)
                if content in _SENTENCE_END and last_end is not None:
                    flush(last_end)
                continue
            start = float(item["start_time"])
            end = float(item["end_time"])
            if seg_start is None:
                seg_start = start
            elif (last_end is not None and start - last_end > _MAX_GAP_SECONDS) or (
                end - seg_start > _MAX_SEGMENT_SECONDS
            ):
                flush(last_end if last_end is not None else start)
                seg_start = start
            words.append(content)
            last_end = end

        if last_end is not None:
            flush(last_end)
        return segments
