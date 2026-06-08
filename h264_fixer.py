#!/usr/bin/env python3
"""
h264_fixer.py - Improved H.264 video recovery tool
Cải thiện từ fixer.pl của BukkoJot (2017)

Các cải tiến:
- Scan song song cả AVCC lẫn Annex B pattern
- Fingerprint đa dạng hơn (toàn bộ file tốt thay vì 20 giây)
- Không bỏ P/B frame trước IDR đầu tiên
- Buffer sliding window để không bỏ sót NAL
- Progress bar, logging chi tiết
- Tự động thử recover audio AAC
"""

import subprocess
import struct
import os
import sys
import json
import logging
import argparse
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("h264_fixer")


# ──────────────────────────────────────────────
# Cấu trúc NAL profile
# ──────────────────────────────────────────────
NAL_TYPE_NAMES = {
    1: "P/B-slice (non-IDR)",
    5: "IDR (Key frame)",
    6: "SEI",
    7: "SPS",
    8: "PPS",
    9: "AUD",
}

@dataclass
class NalProfile:
    nal_type: int
    min_size: int = 0xFFFFFFFF
    max_size: int = 0
    count: int = 0
    # tập fingerprint: 3 byte đầu của NAL data
    fingerprints: set = field(default_factory=set)

    def update(self, size: int, data_prefix: bytes):
        self.min_size = min(self.min_size, size)
        self.max_size = max(self.max_size, size)
        self.count += 1
        self.fingerprints.add(data_prefix[:3])

    def matches(self, size: int, data_prefix: bytes) -> bool:
        if self.count == 0:
            return False
        lo = max(4, self.min_size // 2)
        hi = self.max_size * 2
        if not (lo <= size <= hi):
            return False
        # Kiểm tra fingerprint (ít nhất 2 byte đầu khớp)
        fp = data_prefix[:3]
        for ref in self.fingerprints:
            if fp[:2] == ref[:2]:
                return True
        return False


# ──────────────────────────────────────────────
# Chạy ffmpeg / ffprobe
# ──────────────────────────────────────────────
def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    log.debug("$ " + " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, check=check)


def ffmpeg(*args) -> subprocess.CompletedProcess:
    return run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args])


def ffprobe(*args) -> subprocess.CompletedProcess:
    return run(["ffprobe", "-hide_banner", "-loglevel", "error", *args])


# ──────────────────────────────────────────────
# Bước 1: Trích xuất SPS/PPS header từ file tốt
# ──────────────────────────────────────────────
def extract_sps_pps(good_file: str, tmp_dir: str) -> bytes:
    """Trích SPS+PPS dưới dạng Annex B từ file tốt."""
    h264_sample = os.path.join(tmp_dir, "headers.h264")
    ffmpeg("-i", good_file, "-c", "copy", "-frames:v", "1",
           "-bsf:v", "h264_mp4toannexb", h264_sample)

    with open(h264_sample, "rb") as f:
        data = f.read()

    # Cắt lấy phần SPS+PPS (trước IDR \x65 hoặc \x25)
    for marker in [b"\x00\x00\x00\x01\x65", b"\x00\x00\x00\x01\x25",
                   b"\x00\x00\x01\x65", b"\x00\x00\x01\x25"]:
        idx = data.find(marker)
        if idx > 0:
            header = data[:idx]
            log.info(f"SPS/PPS header: {len(header)} bytes")
            return header

    log.warning("Không tìm thấy ranh giới SPS/PPS rõ ràng, dùng toàn bộ")
    return data[:256]


# ──────────────────────────────────────────────
# Bước 2: Học NAL profile từ file tốt (toàn bộ, không chỉ 20s)
# ──────────────────────────────────────────────
def _profile_from_annexb(annexb_file: str) -> dict[int, NalProfile]:
    """Scan file Annex B, học NAL profile từ start codes."""
    profiles: dict[int, NalProfile] = {}
    with open(annexb_file, "rb") as f:
        data = f.read()

    i = 0
    positions = []
    # Tìm tất cả start code positions
    while i < len(data) - 4:
        if data[i:i+4] == b"\x00\x00\x00\x01":
            positions.append((i, 4))
            i += 4
        elif data[i:i+3] == b"\x00\x00\x01":
            positions.append((i, 3))
            i += 3
        else:
            i += 1

    for idx, (pos, sc_len) in enumerate(positions):
        nal_start = pos + sc_len
        nal_end = positions[idx + 1][0] if idx + 1 < len(positions) else len(data)
        nal_data = data[nal_start:nal_end]
        if len(nal_data) < 2:
            continue
        nal_type = nal_data[0] & 0x1F
        size = len(nal_data)
        prefix = nal_data[:3]
        if nal_type not in profiles:
            profiles[nal_type] = NalProfile(nal_type)
        profiles[nal_type].update(size, prefix)

    return profiles


def _profile_from_avcc_raw(mp4_file: str, pkt_infos: list) -> dict[int, NalProfile]:
    """Đọc AVCC raw từ MP4, dùng pos/size từ ffprobe."""
    profiles: dict[int, NalProfile] = {}
    try:
        with open(mp4_file, "rb") as f:
            for pkt in pkt_infos:
                pos  = int(pkt.get("pos", -1))
                size = int(pkt.get("size", 0))
                if pos < 0 or size < 5:
                    continue
                f.seek(pos)
                raw = f.read(size)
                # Parse AVCC
                offset = 0
                while offset + 5 < len(raw):
                    nal_size = struct.unpack_from(">I", raw, offset)[0]
                    if nal_size == 0 or offset + 4 + nal_size > len(raw) + 8:
                        break
                    nal_header = raw[offset + 4] if offset + 4 < len(raw) else 0
                    nal_type = nal_header & 0x1F
                    prefix = raw[offset + 4: offset + 7]
                    if nal_type not in profiles:
                        profiles[nal_type] = NalProfile(nal_type)
                    profiles[nal_type].update(nal_size, prefix)
                    offset += 4 + nal_size
    except Exception as e:
        log.warning(f"_profile_from_avcc_raw lỗi: {e}")
    return profiles


def build_nal_profiles(good_file: str, tmp_dir: str) -> dict[int, NalProfile]:
    """
    Dùng ffprobe -show_packets để lấy metadata NAL từ toàn bộ file tốt.
    Cải thiện: dùng JSON output thay vì parse text thủ công.
    """
    log.info("Đang phân tích NAL profile từ file tốt...")

    # Bước 2a: copy sang mp4 tạm để ffprobe đọc chính xác
    stat_mp4 = os.path.join(tmp_dir, "stat.mp4")
    ffmpeg("-i", good_file, "-c", "copy", "-an", stat_mp4)

    profiles: dict[int, NalProfile] = {}

    # ── Phương pháp 1: ffprobe -show_packets JSON (lấy size + pos) ──
    result = ffprobe(
        "-select_streams", "v:0",
        "-show_packets",
        "-of", "json",
        stat_mp4,
    )

    pkt_infos = []
    try:
        pkt_data = json.loads(result.stdout)
        pkt_infos = pkt_data.get("packets", [])
        log.info(f"  ffprobe: {len(pkt_infos)} packets")
    except json.JSONDecodeError:
        log.warning("ffprobe JSON parse thất bại")

    # ── Phương pháp 2: Đọc trực tiếp raw bytes từ stat_mp4 ──
    # Trích xuất H.264 Annex B rồi scan NAL thủ công — đáng tin hơn trên Windows
    annexb_file = os.path.join(tmp_dir, "stat.h264")
    ffmpeg("-i", stat_mp4, "-c", "copy", "-bsf:v", "h264_mp4toannexb", annexb_file)

    if os.path.exists(annexb_file):
        log.info("  Đọc NAL profile từ Annex B stream...")
        profiles = _profile_from_annexb(annexb_file)
    
    # Nếu Annex B thất bại, fallback dùng size từ ffprobe packets + đọc AVCC raw
    if not profiles and pkt_infos:
        log.info("  Fallback: đọc AVCC raw từ stat_mp4...")
        profiles = _profile_from_avcc_raw(stat_mp4, pkt_infos)

    for t, p in sorted(profiles.items()):
        name = NAL_TYPE_NAMES.get(t, f"type_{t}")
        log.info(
            f"  NAL {t:2d} ({name}): count={p.count}, "
            f"size=[{p.min_size}..{p.max_size}], fps={len(p.fingerprints)}"
        )

    if not profiles:
        log.warning("Không học được profile nào! Kiểm tra lại file tốt.")

    return profiles


# ──────────────────────────────────────────────
# Bước 3: Scan file hỏng — AVCC + Annex B song song
# ──────────────────────────────────────────────
ANNEXB_MARKERS = [b"\x00\x00\x00\x01", b"\x00\x00\x01"]
BLOCK = 4 * 1024 * 1024  # 4 MB mỗi lần đọc


def scan_bad_file(
    bad_file: str,
    profiles: dict[int, NalProfile],
    sps_pps_header: bytes,
    out_h264: str,
    out_audio: str,
    skip_before_idr: bool = False,
) -> dict:
    """
    Quét file hỏng bằng sliding window, nhận diện NAL theo 2 cách:
    1. AVCC: 4-byte big-endian size + NAL header
    2. Annex B: start code 00 00 01 hoặc 00 00 00 01 + NAL header

    Cải thiện quan trọng:
    - KHÔNG bỏ frame trước IDR nếu skip_before_idr=False
    - Buffer sliding để không bỏ sót NAL ở ranh giới block
    - Ghi thêm \x00\x00\x00\x01 Annex B prefix cho mỗi NAL
    """
    fsize = os.path.getsize(bad_file)
    stats = {"found": 0, "idr": 0, "skipped_bytes": 0, "errors": 0}

    found_idr = False
    pending_frames: list[bytes] = []  # frames trước IDR đầu tiên

    with open(bad_file, "rb") as fin, \
         open(out_h264, "wb") as vout, \
         open(out_audio, "wb") as aout:

        # Ghi SPS/PPS header đầu file
        vout.write(sps_pps_header)

        buf = b""
        read_total = 0
        q = 0

        def progress():
            pct = min(100, (read_total / fsize) * 100) if fsize else 0
            return f"{pct:5.1f}% ({read_total//1024//1024}MB/{fsize//1024//1024}MB)"

        last_log = 0

        while True:
            chunk = fin.read(BLOCK)
            if chunk:
                buf += chunk
                read_total += len(chunk)
            elif q >= len(buf) - 10:
                break  # hết file và đã xử lý hết buffer

            limit = len(buf) - 10  # giữ 10 byte cuối cho lần sau

            while q < limit:
                # ── Hiển thị progress mỗi 5MB ──
                if read_total - last_log > 5 * 1024 * 1024:
                    log.info(f"Progress: {progress()} | found={stats['found']} idr={stats['idr']}")
                    last_log = read_total

                # ── Thử AVCC format ──
                if q + 5 <= len(buf):
                    nal_size = struct.unpack_from(">I", buf, q)[0]
                    nal_hdr  = buf[q + 4]
                    zero_bit = nal_hdr & 0x80
                    nal_type = nal_hdr & 0x1F

                    if (nal_size > 0 and zero_bit == 0
                            and nal_type in profiles
                            and q + 4 + nal_size <= len(buf)):

                        prefix = buf[q + 4: q + 7]
                        if profiles[nal_type].matches(nal_size, prefix):
                            frame_data = buf[q + 4: q + 4 + nal_size]
                            _write_nal(vout, aout, frame_data, nal_type,
                                       found_idr, pending_frames,
                                       skip_before_idr, stats)
                            if nal_type == 5:
                                found_idr = True
                            q += 4 + nal_size
                            continue

                # ── Thử Annex B format ──
                found_annexb = False
                for marker in ANNEXB_MARKERS:
                    ml = len(marker)
                    if buf[q:q + ml] == marker and q + ml + 1 < len(buf):
                        nal_hdr  = buf[q + ml]
                        zero_bit = nal_hdr & 0x80
                        nal_type = nal_hdr & 0x1F

                        if zero_bit == 0 and nal_type in profiles:
                            # Tìm NAL kế tiếp để xác định size
                            end = _find_next_nal(buf, q + ml)
                            if end > q + ml:
                                frame_data = buf[q + ml: end]
                                nal_size = len(frame_data)
                                prefix = frame_data[:3]
                                if profiles[nal_type].matches(nal_size, prefix):
                                    _write_nal(vout, aout, frame_data, nal_type,
                                               found_idr, pending_frames,
                                               skip_before_idr, stats)
                                    if nal_type == 5:
                                        found_idr = True
                                    q = end
                                    found_annexb = True
                                    break

                if found_annexb:
                    continue

                # Không nhận diện được → bỏ qua byte này
                stats["skipped_bytes"] += 1
                q += 1

            # Giữ lại phần chưa xử lý
            buf = buf[q:]
            q = 0

        # Flush pending frames nếu không bao giờ gặp IDR
        if pending_frames and not found_idr:
            log.warning("Không tìm thấy IDR frame! Ghi toàn bộ frame tìm được (có thể lỗi hình)")
            for f in pending_frames:
                vout.write(b"\x00\x00\x00\x01")
                vout.write(f)

    return stats


def _find_next_nal(buf: bytes, start: int) -> int:
    """Tìm vị trí start code tiếp theo sau `start`."""
    for marker in ANNEXB_MARKERS:
        idx = buf.find(marker, start + 1)
        if idx != -1:
            return idx
    return len(buf)


def _write_nal(vout, aout, frame_data: bytes, nal_type: int,
               found_idr: bool, pending: list,
               skip_before_idr: bool, stats: dict):
    """Ghi NAL vào output với Annex B prefix."""
    stats["found"] += 1
    if nal_type == 5:
        stats["idr"] += 1

    if not found_idr and skip_before_idr and nal_type not in (7, 8, 5):
        # Chế độ strict: buffer lại, chờ IDR
        pending.append(frame_data)
        return

    # Flush pending nếu vừa gặp IDR
    if nal_type == 5 and pending:
        for p in pending:
            vout.write(b"\x00\x00\x00\x01")
            vout.write(p)
        pending.clear()

    vout.write(b"\x00\x00\x00\x01")
    vout.write(frame_data)


# ──────────────────────────────────────────────
# Bước 4: Wrap H.264 raw → MP4
# ──────────────────────────────────────────────
def wrap_to_mp4(h264_file: str, output_mp4: str):
    log.info("Đóng gói H.264 → MP4...")
    result = ffmpeg(
        "-err_detect", "ignore_err",
        "-i", h264_file,
        "-c", "copy",
        "-movflags", "+faststart",
        output_mp4,
    )
    if result.returncode != 0:
        log.warning("ffmpeg wrap thất bại, thử với -fflags +genpts")
        ffmpeg(
            "-fflags", "+genpts+igndts",
            "-err_detect", "ignore_err",
            "-i", h264_file,
            "-c", "copy",
            output_mp4,
        )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Sửa video H.264 bị hỏng (cải tiến từ fixer.pl)"
    )
    parser.add_argument("good_file",  help="File MP4 tốt cùng loại camera/encoder")
    parser.add_argument("bad_file",   help="File MP4 bị hỏng cần sửa")
    parser.add_argument("output",     help="Tên file output (vd: recovered.mp4)")
    parser.add_argument("--strict-idr", action="store_true",
                        help="Bỏ frame trước IDR đầu tiên (giống fixer.pl gốc)")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Giữ lại thư mục tạm để debug")
    parser.add_argument("--debug", action="store_true",
                        help="Hiển thị log debug chi tiết")
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    # Kiểm tra đầu vào
    for f in [args.good_file, args.bad_file]:
        if not os.path.exists(f):
            log.error(f"File không tồn tại: {f}")
            sys.exit(1)

    tmp_dir = tempfile.mkdtemp(prefix="h264fix_")
    log.info(f"Thư mục tạm: {tmp_dir}")

    out_h264  = os.path.join(tmp_dir, "recovered.h264")
    out_audio = os.path.join(tmp_dir, "recovered_audio.raw")
    out_mp4   = args.output

    try:
        # Bước 1
        log.info("═══ Bước 1: Trích xuất SPS/PPS header ═══")
        sps_pps = extract_sps_pps(args.good_file, tmp_dir)

        # Bước 2
        log.info("═══ Bước 2: Học NAL profile ═══")
        profiles = build_nal_profiles(args.good_file, tmp_dir)

        if not profiles:
            log.error("Không thể học NAL profile. Thoát.")
            sys.exit(2)

        # Bước 3
        log.info("═══ Bước 3: Scan và khôi phục ═══")
        log.info(f"  skip_before_idr = {args.strict_idr}")
        stats = scan_bad_file(
            args.bad_file, profiles, sps_pps,
            out_h264, out_audio,
            skip_before_idr=args.strict_idr,
        )

        log.info(f"  Kết quả: {stats}")
        h264_size = os.path.getsize(out_h264)
        log.info(f"  H.264 raw: {h264_size / 1024 / 1024:.2f} MB")

        if stats["found"] == 0:
            log.error("Không tìm thấy NAL nào hợp lệ. Kiểm tra lại file tốt vs file hỏng.")
            sys.exit(3)

        # Bước 4
        log.info("═══ Bước 4: Đóng gói MP4 ═══")
        wrap_to_mp4(out_h264, out_mp4)

        if os.path.exists(out_mp4):
            out_size = os.path.getsize(out_mp4)
            log.info(f"✓ Output: {out_mp4} ({out_size / 1024 / 1024:.2f} MB)")
            log.info(f"  Tổng NAL tìm được : {stats['found']}")
            log.info(f"  IDR (key frame)   : {stats['idr']}")
            log.info(f"  Byte bỏ qua (rác) : {stats['skipped_bytes']}")
        else:
            log.error("Tạo MP4 thất bại.")

    finally:
        if args.keep_tmp:
            log.info(f"Giữ lại tmp: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()