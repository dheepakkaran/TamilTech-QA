"""YouTube comment scraper using the YouTube Data API v3.

For each configured channel:

1. Resolve the channel's ``uploads`` playlist via ``channels.list``.
2. Page through ``playlistItems`` to collect video IDs.
3. For each video, page through ``commentThreads`` to harvest top-level
   comments plus replies.

Output: one JSONL file per channel under ``data/raw/``. The script keeps a
``data/raw/.scraped_videos.json`` ledger of already-processed video IDs so
runs can resume after interruption.

Example
-------
.. code-block:: bash

    python -m src.data_collection.youtube_scraper \
        --config config/data_config.yaml --limit-channels 1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

from src.utils import ensure_dir, load_config, project_root
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:  # pragma: no cover - module imports must still work for help/-tests
    build = None  # type: ignore[assignment]
    HttpError = Exception  # type: ignore[misc,assignment]


class YouTubeScraper:
    """Thin, resumable wrapper around the YouTube Data API v3 client.

    Args:
        api_key: A YouTube Data API v3 key.
        max_videos_per_channel: Cap on videos fetched per channel.
        max_comments_per_video: Cap on top-level comments fetched per video.
        retry_max_attempts: Number of retries for transient errors.
        retry_base_delay: Base delay in seconds for exponential backoff.
        retry_max_delay: Cap on backoff delay.

    Example:
        >>> scraper = YouTubeScraper(api_key="...")           # doctest: +SKIP
        >>> for record in scraper.scrape_channel("UCxxxx"):   # doctest: +SKIP
        ...     ...
    """

    LEDGER_NAME = ".scraped_videos.json"

    def __init__(
        self,
        api_key: str,
        max_videos_per_channel: int = 200,
        max_comments_per_video: int = 500,
        retry_max_attempts: int = 5,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "YOUTUBE_API_KEY is empty. Set it in .env or in the data_config.yaml."
            )
        if build is None:
            raise ImportError(
                "google-api-python-client is not installed. "
                "Run: pip install google-api-python-client"
            )
        self.api_key = api_key
        self.max_videos_per_channel = max_videos_per_channel
        self.max_comments_per_video = max_comments_per_video
        self.retry_max_attempts = retry_max_attempts
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.client = build("youtube", "v3", developerKey=api_key, cache_discovery=False)

    # ------------------------------------------------------------------ #
    # Backoff helper
    # ------------------------------------------------------------------ #
    def _retry(self, fn, *args, **kwargs):
        """Execute ``fn(*args, **kwargs)`` with exponential backoff on errors.

        Args:
            fn: Callable returning a YouTube API response.

        Returns:
            Whatever ``fn`` returns.

        Raises:
            HttpError: Re-raised after all retries are exhausted.
        """
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs).execute()
            except HttpError as e:
                status = getattr(getattr(e, "resp", None), "status", None)
                attempt += 1
                if status in (403,) and "quotaExceeded" in str(e):
                    log.error("YouTube quota exhausted: {}", e)
                    raise
                if attempt >= self.retry_max_attempts:
                    log.error("Giving up after {} attempts: {}", attempt, e)
                    raise
                delay = min(
                    self.retry_max_delay,
                    self.retry_base_delay * (2 ** (attempt - 1)),
                ) + random.uniform(0, 0.5)
                log.warning(
                    "HTTP {} from YouTube (attempt {}/{}). Sleeping {:.1f}s...",
                    status, attempt, self.retry_max_attempts, delay,
                )
                time.sleep(delay)

    # ------------------------------------------------------------------ #
    # Channel → uploads playlist
    # ------------------------------------------------------------------ #
    def get_uploads_playlist_id(self, channel_id: str) -> Optional[str]:
        """Return the ``uploads`` playlist ID for a channel.

        Args:
            channel_id: YouTube channel ID (UC...).

        Returns:
            The uploads playlist ID, or None if the channel is not found.
        """
        resp = self._retry(
            self.client.channels().list,
            part="contentDetails,snippet",
            id=channel_id,
        )
        items = resp.get("items", [])
        if not items:
            log.warning("No channel found for ID {}", channel_id)
            return None
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    def get_channel_title(self, channel_id: str) -> str:
        """Return the channel's display name, falling back to the ID.

        Args:
            channel_id: YouTube channel ID.

        Returns:
            Channel title as a string.
        """
        resp = self._retry(
            self.client.channels().list,
            part="snippet",
            id=channel_id,
        )
        items = resp.get("items", [])
        return items[0]["snippet"]["title"] if items else channel_id

    # ------------------------------------------------------------------ #
    # Playlist → video IDs
    # ------------------------------------------------------------------ #
    def iter_video_ids(self, playlist_id: str) -> Iterator[Dict[str, Any]]:
        """Yield ``{video_id, title, published_at, description}`` per video.

        Args:
            playlist_id: YouTube playlist ID (use the uploads playlist).

        Yields:
            Per-video metadata dictionaries.
        """
        page_token: Optional[str] = None
        emitted = 0
        while True:
            resp = self._retry(
                self.client.playlistItems().list,
                part="snippet,contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=page_token,
            )
            for item in resp.get("items", []):
                vid = item["contentDetails"]["videoId"]
                snippet = item.get("snippet", {})
                yield {
                    "video_id": vid,
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "published_at": snippet.get("publishedAt", ""),
                }
                emitted += 1
                if emitted >= self.max_videos_per_channel:
                    return
            page_token = resp.get("nextPageToken")
            if not page_token:
                return

    # ------------------------------------------------------------------ #
    # Video → comments + replies
    # ------------------------------------------------------------------ #
    def iter_comments(self, video_id: str) -> Iterator[Dict[str, Any]]:
        """Yield comment records (top-level + replies) for ``video_id``.

        Args:
            video_id: YouTube video ID.

        Yields:
            Dictionaries with ``comment_text``, ``reply_text`` (list),
            ``like_count``, and ``published_at``.
        """
        page_token: Optional[str] = None
        seen = 0
        while seen < self.max_comments_per_video:
            try:
                resp = self._retry(
                    self.client.commentThreads().list,
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=page_token,
                    textFormat="plainText",
                )
            except HttpError as e:
                msg = str(e)
                if "commentsDisabled" in msg or "videoNotFound" in msg:
                    log.info("Comments disabled / unavailable for {}", video_id)
                    return
                raise
            for item in resp.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                replies = item.get("replies", {}).get("comments", []) or []
                yield {
                    "comment_text": top.get("textDisplay", ""),
                    "reply_text": [r["snippet"].get("textDisplay", "") for r in replies],
                    "like_count": top.get("likeCount", 0),
                    "published_at": top.get("publishedAt", ""),
                    "author": top.get("authorDisplayName", ""),
                }
                seen += 1
                if seen >= self.max_comments_per_video:
                    return
            page_token = resp.get("nextPageToken")
            if not page_token:
                return

    # ------------------------------------------------------------------ #
    # Top-level: scrape a channel
    # ------------------------------------------------------------------ #
    def scrape_channel(
        self,
        channel_id: str,
        already_seen: Optional[Set[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate per-video records (with comments) for one channel.

        Args:
            channel_id: YouTube channel ID.
            already_seen: Optional set of video IDs to skip (resumability).

        Yields:
            Per-video records of the shape::

                {
                  "video_id": str, "title": str, "description": str,
                  "published_at": str, "channel_id": str,
                  "comments": [ {comment_text, reply_text, ...}, ... ]
                }
        """
        already_seen = already_seen or set()
        uploads = self.get_uploads_playlist_id(channel_id)
        if not uploads:
            return
        for vid_meta in self.iter_video_ids(uploads):
            if vid_meta["video_id"] in already_seen:
                log.debug("Skip already-scraped video {}", vid_meta["video_id"])
                continue
            comments = list(self.iter_comments(vid_meta["video_id"]))
            yield {**vid_meta, "channel_id": channel_id, "comments": comments}


# ---------------------------------------------------------------------- #
# Ledger helpers
# ---------------------------------------------------------------------- #
def _load_ledger(path: Path) -> Set[str]:
    """Load the resume ledger of already-scraped video IDs.

    Args:
        path: Ledger JSON file path.

    Returns:
        Set of video IDs.
    """
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        log.warning("Corrupt ledger at {}; starting fresh.", path)
        return set()


def _save_ledger(path: Path, ids: Iterable[str]) -> None:
    """Persist the resume ledger.

    Args:
        path: Ledger JSON file path.
        ids: Iterable of video IDs to persist.

    Returns:
        None.
    """
    path.write_text(json.dumps(sorted(set(ids))), encoding="utf-8")


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def run(config_path: str, limit_channels: Optional[int] = None) -> None:
    """Run the full multi-channel scrape from a config file.

    Args:
        config_path: Path to ``config/data_config.yaml``.
        limit_channels: If set, only scrape the first N channels.

    Returns:
        None. Writes one JSONL per channel under ``paths.raw_dir``.
    """
    cfg = load_config(config_path)
    yt_cfg = cfg["youtube"]
    raw_dir = ensure_dir(project_root() / cfg["paths"]["raw_dir"])
    ledger_path = raw_dir / YouTubeScraper.LEDGER_NAME
    seen = _load_ledger(ledger_path)

    scraper = YouTubeScraper(
        api_key=yt_cfg["api_key"],
        max_videos_per_channel=yt_cfg.get("max_videos_per_channel", 200),
        max_comments_per_video=yt_cfg.get("max_comments_per_video", 500),
        retry_max_attempts=yt_cfg.get("retry", {}).get("max_attempts", 5),
        retry_base_delay=yt_cfg.get("retry", {}).get("base_delay_seconds", 2.0),
        retry_max_delay=yt_cfg.get("retry", {}).get("max_delay_seconds", 60.0),
    )

    channels: List[str] = list(yt_cfg.get("target_channels", []))
    if limit_channels:
        channels = channels[:limit_channels]

    grand_total = 0
    for ch_id in channels:
        title = scraper.get_channel_title(ch_id).replace("/", "_").replace(" ", "_")
        out_file = raw_dir / f"youtube_comments_{title}.jsonl"
        log.info("Scraping channel '{}' ({}) → {}", title, ch_id, out_file)

        channel_total = 0
        with out_file.open("a", encoding="utf-8") as f:
            for record in scraper.scrape_channel(ch_id, already_seen=seen):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(record["video_id"])
                channel_total += len(record["comments"])
                if len(seen) % 25 == 0:
                    _save_ledger(ledger_path, seen)
        _save_ledger(ledger_path, seen)
        log.info("Channel '{}': {} comments scraped.", title, channel_total)
        grand_total += channel_total

    log.info("All channels done. Total comments scraped: {}", grand_total)


def main() -> None:
    """CLI entrypoint for ``python -m src.data_collection.youtube_scraper``."""
    setup_logging()
    p = argparse.ArgumentParser(description="Scrape YouTube comments for TamilTech-QA.")
    p.add_argument(
        "--config",
        default="config/data_config.yaml",
        help="Path to data config YAML.",
    )
    p.add_argument(
        "--limit-channels",
        type=int,
        default=None,
        help="If set, only scrape the first N channels (useful for smoke tests).",
    )
    args = p.parse_args()
    run(args.config, limit_channels=args.limit_channels)


if __name__ == "__main__":
    main()
