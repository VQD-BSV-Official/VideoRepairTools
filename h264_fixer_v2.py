#!/usr/bin/env python3
"""
h264_fixer.py  v3  -  H.264 / Sony XAVC video recovery tool
Sửa các lỗi thực tế từ log ffplay:
  - non-existing SPS 32 referenced in buffering period
  - pps_id out of range
  - slice type too large
  - Invalid NAL unit 0/1, decode_slice_header error

Chiến lược lọc NAL nâng cấp:
  SPS  : parse RBSP -> kiem tra profile_idc, sps_id (0-31), width/height hop ly
  PPS  : parse RBSP -> kiem tra pps_id (0-255), sps_id phai khop voi SPS file tot
  SEI  : strip buffering_period khi tham chieu SPS id ngoai whitelist
  Slice: kiem tra first_mb + slice_type (0-9) bang Golomb UE
  AUD  : kiem tra primary_pic_type (0-7)
  Fingerprint: so sanh 4 byte (tang tu 2 byte)
  _find_next_nal: gioi han 2MB (tranh nhan NAL gia khong lo)
"""

import subprocess, struct, os, sys, json, logging, argparse, tempfile, shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("h264_fixer")

NAL_TYPE_NAMES = {
    1:"Non-IDR slice (P/B)", 2:"Slice partition A", 3:"Slice partition B",
    4:"Slice partition C",   5:"IDR slice (Key)",   6:"SEI",
    7:"SPS",                 8:"PPS",               9:"AUD",
   10:"End of sequence",    11:"End of stream",    12:"Filler",
   13:"SPS extension",      14:"Prefix NAL (SVC)", 15:"Subset SPS",
   19:"Auxiliary picture",  20:"SVC Extension",    21:"MVC depth",
}
XAVC_NAL_WHITELIST = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,20,21}
SLICE_TYPES = {1,2,3,4,5,20,21}
ANNEXB_MARKERS = [b"\x00\x00\x00\x01", b"\x00\x00\x01"]
BLOCK = 4 * 1024 * 1024

AUD_NAL    = bytes([0x09, 0xF0])
AUD_ANNEXB = b"\x00\x00\x00\x01" + AUD_NAL


# ============================================================
# RBSP bit reader
# ============================================================
def _remove_emulation_prevention(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        if i + 2 < n and data[i] == 0 and data[i+1] == 0 and data[i+2] == 3:
            out.append(0); out.append(0)
            i += 3
        else:
            out.append(data[i])
            i += 1
    return bytes(out)


class BitReader:
    def __init__(self, data: bytes):
        self._data = _remove_emulation_prevention(data)
        self._bit  = 0

    def _bit_len(self):
        return len(self._data) * 8

    def read_u(self, n: int) -> int:
        val = 0
        for _ in range(n):
            if self._bit >= self._bit_len():
                raise EOFError("bit stream exhausted")
            bi = self._bit >> 3
            bo = 7 - (self._bit & 7)
            val = (val << 1) | ((self._data[bi] >> bo) & 1)
            self._bit += 1
        return val

    def read_ue(self) -> int:
        leading = 0
        while self._bit < self._bit_len():
            b = self.read_u(1)
            if b:
                break
            leading += 1
        if leading > 20:
            raise ValueError(f"UE golomb too large: leading={leading}")
        suffix = self.read_u(leading) if leading else 0
        return (1 << leading) - 1 + suffix

    def read_se(self) -> int:
        ue = self.read_ue()
        return (ue + 1) // 2 * (1 if ue & 1 else -1)

    def read_flag(self) -> int:
        return self.read_u(1)


# ============================================================
# SPS parse + validate
# ============================================================
@dataclass
class SPSInfo:
    sps_id:      int = 0
    profile_idc: int = 0
    level_idc:   int = 0
    width:       int = 0
    height:      int = 0

    def plausible(self) -> bool:
        VALID_PROFILES = {
            66,77,88,100,110,122,244,
            44,83,86,118,128,138,139,134,135
        }
        if self.sps_id    not in range(32):      return False
        if self.profile_idc not in VALID_PROFILES: return False
        if not (10 <= self.level_idc <= 62):     return False
        if not (16 <= self.width  <= 8192):      return False
        if not (16 <= self.height <= 8192):      return False
        return True


def parse_sps(nal_data: bytes) -> Optional[SPSInfo]:
    try:
        br = BitReader(nal_data[1:])
        profile_idc = br.read_u(8)
        _constraint = br.read_u(8)
        level_idc   = br.read_u(8)
        sps_id      = br.read_ue()

        if profile_idc in (100,110,122,244,44,83,86,118,128,138,139,134,135):
            chroma_fmt = br.read_ue()
            if chroma_fmt == 3:
                br.read_flag()
            br.read_ue(); br.read_ue(); br.read_flag()
            if br.read_flag():
                n = 12 if chroma_fmt == 3 else 8
                for si in range(n):
                    if br.read_flag():
                        sz   = 16 if si < 6 else 64
                        last = 8; nxt = 8
                        for _ in range(sz):
                            if nxt:
                                delta = br.read_se()
                                nxt   = (last + delta + 256) % 256
                            last = nxt if nxt else last

        br.read_ue()
        poc_type = br.read_ue()
        if poc_type == 0:
            br.read_ue()
        elif poc_type == 1:
            br.read_flag(); br.read_se(); br.read_se()
            for _ in range(br.read_ue()):
                br.read_se()

        br.read_ue(); br.read_flag()
        w = (br.read_ue() + 1) * 16
        h_map = br.read_ue() + 1
        fmo   = br.read_flag()
        h     = h_map * 16 * (2 - fmo)

        info = SPSInfo(sps_id, profile_idc, level_idc, w, h)
        return info if info.plausible() else None
    except Exception:
        return None


def parse_pps_ids(nal_data: bytes) -> Optional[tuple]:
    try:
        br = BitReader(nal_data[1:])
        pps_id = br.read_ue()
        sps_id = br.read_ue()
        if pps_id > 255 or sps_id > 31:
            return None
        return (pps_id, sps_id)
    except Exception:
        return None


def validate_aud(nal_data: bytes) -> bool:
    return len(nal_data) >= 2


def validate_slice_header(nal_data: bytes) -> bool:
    try:
        br = BitReader(nal_data[1:])
        first_mb   = br.read_ue()
        slice_type = br.read_ue()
        if first_mb > 36864:  return False   # > 4K MB count
        if slice_type > 9:    return False   # only 0-9 valid
        return True
    except Exception:
        return False


def _encode_ff(n: int) -> bytes:
    return bytes([0xFF] * (n // 255)) + bytes([n % 255])


def filter_sei(nal_data: bytes, valid_sps_ids: set) -> Optional[bytes]:
    """
    Duyet tung SEI payload; loai buffering_period tham chieu SPS id sai.
    Tra ve None neu khong con payload nao hop le.
    """
    if len(nal_data) < 2:
        return None
    nal_hdr = nal_data[0:1]
    data    = nal_data[1:]
    i       = 0
    out_payloads = bytearray()

    while i < len(data):
        ptype = 0
        while i < len(data) and data[i] == 0xFF:
            ptype += 255; i += 1
        if i >= len(data): break
        ptype += data[i]; i += 1

        psize = 0
        while i < len(data) and data[i] == 0xFF:
            psize += 255; i += 1
        if i >= len(data): break
        psize += data[i]; i += 1

        payload = data[i:i+psize]
        i += psize
        if psize == 0 or len(payload) < psize:
            break

        keep = True
        if ptype == 0:  # buffering_period
            try:
                br2 = BitReader(payload)
                bp_id = br2.read_ue()
                if bp_id not in valid_sps_ids:
                    keep = False
            except Exception:
                keep = False

        if keep:
            out_payloads += _encode_ff(ptype) + _encode_ff(psize) + payload

    if not out_payloads:
        return None
    return nal_hdr + bytes(out_payloads) + b"\x80"


# ============================================================
# NAL Profile
# ============================================================
@dataclass
class NalProfile:
    nal_type: int
    min_size: int = 0xFFFFFFFF
    max_size: int = 0
    count:    int = 0
    fingerprints: set = field(default_factory=set)

    def update(self, size: int, prefix: bytes):
        self.min_size = min(self.min_size, size)
        self.max_size = max(self.max_size, size)
        self.count += 1
        if len(prefix) >= 4:
            self.fingerprints.add(bytes(prefix[:4]))

    def matches(self, size: int, prefix: bytes) -> bool:
        if self.count == 0:
            return False
        if self.nal_type in (6,7,8,9,14):
            lo, hi = 1, max(self.max_size * 4, 8192)
        else:
            lo = max(4, self.min_size // 2)
            hi = self.max_size * 2 + 4096
        if not (lo <= size <= hi):
            return False
        if not self.fingerprints:
            return True
        fp = bytes(prefix[:4])
        for ref in self.fingerprints:
            if fp[:4] == ref[:4]:
                return True
        # Fallback 2-byte chi cho non-slice
        if self.nal_type not in SLICE_TYPES:
            for ref in self.fingerprints:
                if fp[:2] == ref[:2]:
                    return True
        return False


# ============================================================
# ffmpeg helpers
# ============================================================
def run(cmd, check=True):
    log.debug("$ " + " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, check=check)

def ffmpeg(*args):  return run(["ffmpeg",  "-y","-hide_banner","-loglevel","error",*args])
def ffprobe(*args): return run(["ffprobe",    "-hide_banner","-loglevel","error",*args])


def detect_container(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".mxf": return "mxf"
    if ext in (".mp4",".m4v",".mov"): return "mp4"
    with open(path,"rb") as f: hdr = f.read(16)
    if hdr[:4] == b"\x06\x0e\x2b\x34": return "mxf"
    if hdr[4:8] == b"ftyp" or b"XAVC" in hdr: return "mp4"
    return "raw"


def is_xavc(path: str) -> bool:
    if detect_container(path) == "mxf": return True
    try:
        with open(path,"rb") as f: d = f.read(64)
        return b"XAVC" in d or b"xavc" in d
    except Exception:
        return False


def extract_annexb(src: str, dst: str, xavc: bool = False) -> bool:
    extra = ["-err_detect","ignore_err","-fflags","+discardcorrupt"] if xavc else []
    r = ffmpeg(*extra, "-i", src,
               "-map","0:v:0","-c:v","copy",
               "-bsf:v","h264_mp4toannexb", dst)
    return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0


# ============================================================
# Buoc 1: Trich SPS/PPS header
# ============================================================
def _find_all_startcodes(data: bytes) -> list:
    pos = []; i = 0; n = len(data)
    while i < n - 3:
        if data[i] == 0 and data[i+1] == 0:
            if i+3 < n and data[i+2] == 0 and data[i+3] == 1:
                pos.append((i,4)); i += 4; continue
            if data[i+2] == 1:
                pos.append((i,3)); i += 3; continue
        i += 1
    return pos


def extract_sps_pps(good_file: str, tmp_dir: str, xavc: bool) -> tuple:
    """Returns (header_bytes, valid_sps_ids, valid_pps_ids)."""
    h264_sample = os.path.join(tmp_dir, "headers.h264")
    if not extract_annexb(good_file, h264_sample, xavc):
        log.error("Khong trich duoc Annex B tu file tot!"); sys.exit(1)

    with open(h264_sample, "rb") as f:
        data = f.read()

    idr_markers = [
        b"\x00\x00\x00\x01\x65", b"\x00\x00\x00\x01\x45",
        b"\x00\x00\x00\x01\x25", b"\x00\x00\x01\x65", b"\x00\x00\x01\x25",
    ]
    header = b""
    for m in idr_markers:
        idx = data.find(m)
        if idx > 4:
            header = data[:idx]; break
    if not header:
        header = data[:512]

    valid_sps_ids: set = set()
    valid_pps_ids: set = set()
    positions = _find_all_startcodes(header)
    for i, (pos, sc) in enumerate(positions):
        ns = pos + sc
        ne = positions[i+1][0] if i+1 < len(positions) else len(header)
        nd = header[ns:ne]
        if len(nd) < 2: continue
        nt = nd[0] & 0x1F
        if nt == 7:
            info = parse_sps(nd)
            if info: valid_sps_ids.add(info.sps_id)
        elif nt == 8:
            ids = parse_pps_ids(nd)
            if ids: valid_pps_ids.add(ids[0])

    log.info(f"SPS ids hop le: {sorted(valid_sps_ids)}")
    log.info(f"PPS ids hop le: {sorted(valid_pps_ids)}")

    if not header.startswith(b"\x00\x00\x00\x01\x09") and \
       not header.startswith(b"\x00\x00\x01\x09"):
        header = AUD_ANNEXB + header

    return header, valid_sps_ids, valid_pps_ids


# ============================================================
# Buoc 2: NAL Profile
# ============================================================
def _profile_from_annexb(annexb_file: str) -> dict:
    profiles: dict[int, NalProfile] = {}
    with open(annexb_file,"rb") as f: data = f.read()
    positions = _find_all_startcodes(data)
    for idx,(pos,sc) in enumerate(positions):
        ns = pos + sc
        ne = positions[idx+1][0] if idx+1 < len(positions) else len(data)
        nd = data[ns:ne]
        if len(nd) < 1: continue
        if nd[0] & 0x80: continue
        nt = nd[0] & 0x1F
        if nt not in XAVC_NAL_WHITELIST: continue
        if nt not in profiles: profiles[nt] = NalProfile(nt)
        profiles[nt].update(len(nd), nd[:4])
    return profiles


def _profile_from_avcc_raw(mp4_file: str, pkt_infos: list) -> dict:
    profiles: dict[int, NalProfile] = {}
    try:
        with open(mp4_file,"rb") as f:
            for pkt in pkt_infos:
                pos  = int(pkt.get("pos",-1))
                size = int(pkt.get("size",0))
                if pos < 0 or size < 5: continue
                f.seek(pos); raw = f.read(size)
                off = 0
                while off + 5 <= len(raw):
                    ns = struct.unpack_from(">I",raw,off)[0]
                    if ns == 0 or off+4+ns > len(raw)+8: break
                    if off+4 >= len(raw): break
                    nh = raw[off+4]
                    if nh & 0x80: off += 1; continue
                    nt = nh & 0x1F
                    if nt in XAVC_NAL_WHITELIST:
                        pf = raw[off+4:off+8]
                        if nt not in profiles: profiles[nt] = NalProfile(nt)
                        profiles[nt].update(ns, pf)
                    off += 4+ns
    except Exception as e:
        log.warning(f"avcc_raw error: {e}")
    return profiles


def build_nal_profiles(good_file: str, tmp_dir: str, xavc: bool) -> dict:
    log.info("Dang phan tich NAL profile tu file tot...")
    stat_mp4 = os.path.join(tmp_dir, "stat.mp4")
    ffmpeg("-i", good_file, "-c","copy","-an", stat_mp4)

    profiles: dict[int, NalProfile] = {}
    annexb = os.path.join(tmp_dir, "stat.h264")
    if extract_annexb(stat_mp4, annexb, xavc):
        log.info("  Doc NAL profile tu Annex B stream...")
        profiles = _profile_from_annexb(annexb)

    if not profiles:
        try:
            r = ffprobe("-select_streams","v:0","-show_packets","-of","json", stat_mp4)
            pkt_infos = json.loads(r.stdout).get("packets",[])
        except Exception:
            pkt_infos = []
        if pkt_infos:
            log.info("  Fallback: AVCC raw...")
            profiles = _profile_from_avcc_raw(stat_mp4, pkt_infos)

    for t, sz in [(9,2),(6,128),(7,64),(8,16)]:
        if t not in profiles:
            p = NalProfile(t); p.min_size=1; p.max_size=sz*8; p.count=1
            profiles[t] = p

    for t, p in sorted(profiles.items()):
        log.info(f"  NAL {t:2d} ({NAL_TYPE_NAMES.get(t,f'type_{t}')}): "
                 f"count={p.count}, size=[{p.min_size}..{p.max_size}]")
    return profiles


# ============================================================
# Access Unit Buffer
# ============================================================
class AccessUnitBuffer:
    _WRITE_ORDER = [9,7,8,6,13,14,15,19,1,2,3,4,5,20,21,10,11,12]

    def __init__(self, fout, inject_aud: bool = True):
        self._fout = fout
        self._inject_aud = inject_aud
        self._buf:  dict[int,list] = defaultdict(list)
        self._has_slice = False
        self._au_count  = 0

    def add(self, nt: int, nd: bytes):
        if nt == 9 and self._has_slice:
            self.flush()
        self._buf[nt].append(nd)
        if nt in SLICE_TYPES:
            self._has_slice = True

    def flush(self):
        if not self._buf: return
        if self._inject_aud and 9 not in self._buf:
            self._buf[9] = [AUD_NAL]
        written = set(self._WRITE_ORDER)
        for nt in self._WRITE_ORDER:
            for nd in self._buf.get(nt,[]):
                self._fout.write(b"\x00\x00\x00\x01"); self._fout.write(nd)
        for nt in sorted(self._buf):
            if nt not in written:
                for nd in self._buf[nt]:
                    self._fout.write(b"\x00\x00\x00\x01"); self._fout.write(nd)
        self._buf.clear(); self._has_slice = False; self._au_count += 1

    def close(self): self.flush()

    @property
    def au_count(self): return self._au_count


# ============================================================
# Helpers scan
# ============================================================
def _find_next_nal(buf: bytes, start: int, max_search: int = 2*1024*1024) -> int:
    """
    Tim start code ke tiep sau start, gioi han max_search byte.
    Tra ve -1 neu khong tim thay (tranh nhan garbage NAL qua lon).
    """
    limit = min(len(buf), start + max_search)
    best  = -1
    for m in ANNEXB_MARKERS:
        idx = buf.find(m, start + 1, limit)
        if idx != -1 and (best == -1 or idx < best):
            best = idx
    return best


def _check_escape_bytes(data: bytes) -> bool:
    i = 0
    while i < len(data) - 2:
        if data[i] == 0 and data[i+1] == 0 and data[i+2] == 3:
            if i+3 < len(data) and data[i+3] > 3:
                return False
        i += 1
    return True


# ============================================================
# Buoc 3: Scan file hong
# ============================================================
def scan_bad_file(
    bad_file:        str,
    profiles:        dict,
    sps_pps_header:  bytes,
    valid_sps_ids:   set,
    valid_pps_ids:   set,
    out_h264:        str,
    out_audio:       str,
    skip_before_idr: bool = False,
    xavc_mode:       bool = False,
    inject_aud:      bool = True,
) -> dict:
    fsize = os.path.getsize(bad_file)
    stats = dict(
        found=0, idr=0, sei=0, aud=0, sps=0, pps=0,
        skipped_bytes=0, bad_sps=0, bad_pps=0, bad_sei=0,
        bad_slice=0, bad_escape=0, access_units=0,
    )

    found_idr = False
    pending:   list[tuple] = []

    with open(bad_file,"rb") as fin, \
         open(out_h264, "wb") as vout, \
         open(out_audio,"wb") as aout:

        au_buf = AccessUnitBuffer(vout, inject_aud)
        vout.write(sps_pps_header)

        buf = b""; read_total = 0; q = 0; last_log = 0

        def progress():
            pct = min(100.0, (read_total / fsize)*100) if fsize else 0
            return f"{pct:5.1f}% ({read_total//1048576}MB/{fsize//1048576}MB)"

        def accept_nal(nt: int, nd: bytes):
            nonlocal found_idr

            # Escape byte check
            if not _check_escape_bytes(nd):
                stats["bad_escape"] += 1; return

            # Per-type validation
            if nt == 7:   # SPS
                info = parse_sps(nd)
                if info is None:
                    stats["bad_sps"] += 1; return
                if valid_sps_ids and info.sps_id not in valid_sps_ids:
                    stats["bad_sps"] += 1
                    log.debug(f"SPS id={info.sps_id} khong co trong {valid_sps_ids}")
                    return
                stats["sps"] += 1

            elif nt == 8:   # PPS
                ids = parse_pps_ids(nd)
                if ids is None:
                    stats["bad_pps"] += 1; return
                _, sps_id = ids
                if valid_sps_ids and sps_id not in valid_sps_ids:
                    stats["bad_pps"] += 1
                    log.debug(f"PPS tham chieu sps_id={sps_id} khong ton tai")
                    return
                stats["pps"] += 1

            elif nt == 6:   # SEI
                nonlocal_nd = filter_sei(nd, valid_sps_ids if valid_sps_ids else set(range(32)))
                if nonlocal_nd is None:
                    stats["bad_sei"] += 1; return
                nd = nonlocal_nd
                stats["sei"] += 1

            elif nt == 9:   # AUD
                if not validate_aud(nd):
                    stats["bad_escape"] += 1; return
                stats["aud"] += 1

            elif nt in SLICE_TYPES:
                if not validate_slice_header(nd):
                    stats["bad_slice"] += 1; return

            if nt == 5:
                stats["idr"] += 1; found_idr = True
            stats["found"] += 1

            if not found_idr and skip_before_idr and nt not in (7,8,5,6,9):
                pending.append((nt, nd)); return

            if nt == 5 and pending:
                for pt, pd in pending: au_buf.add(pt, pd)
                pending.clear()

            au_buf.add(nt, nd)

            if not xavc_mode and nt in (1, 5):
                au_buf.flush()

        # Sliding window
        while True:
            chunk = fin.read(BLOCK)
            if chunk:
                buf += chunk; read_total += len(chunk)
            elif q >= len(buf) - 10:
                break

            limit = len(buf) - 10

            while q < limit:
                if read_total - last_log > 5*1024*1024:
                    log.info(f"Progress: {progress()} | "
                             f"found={stats['found']} idr={stats['idr']} "
                             f"bad_sps={stats['bad_sps']} bad_pps={stats['bad_pps']} "
                             f"bad_slice={stats['bad_slice']}")
                    last_log = read_total

                # AVCC
                if q + 5 <= len(buf):
                    nal_size = struct.unpack_from(">I", buf, q)[0]
                    if 0 < nal_size <= 50*1024*1024 and q+4+nal_size <= len(buf):
                        nh = buf[q+4]
                        if not (nh & 0x80):
                            nt = nh & 0x1F
                            if nt in profiles and nt in XAVC_NAL_WHITELIST:
                                pf = buf[q+4:q+8]
                                if profiles[nt].matches(nal_size, pf):
                                    accept_nal(nt, bytes(buf[q+4:q+4+nal_size]))
                                    q += 4+nal_size; continue

                # Annex B
                found_ab = False
                for marker in ANNEXB_MARKERS:
                    ml = len(marker)
                    if q+ml+1 >= len(buf): continue
                    if buf[q:q+ml] != marker: continue
                    nh = buf[q+ml]
                    if nh & 0x80: continue
                    nt = nh & 0x1F
                    if nt not in XAVC_NAL_WHITELIST: continue

                    end = _find_next_nal(buf, q+ml)
                    if end == -1:
                        remaining = len(buf) - (q+ml)
                        if remaining > 1*1024*1024:
                            break   # qua lon, co the la garbage
                        end = len(buf)

                    nd_c = bytes(buf[q+ml:end])
                    ns   = len(nd_c)
                    pf   = nd_c[:4]
                    in_p = nt in profiles
                    if not in_p or profiles[nt].matches(ns, pf):
                        accept_nal(nt, nd_c)
                        q = end; found_ab = True; break

                if found_ab: continue
                stats["skipped_bytes"] += 1; q += 1

            buf = buf[q:]; q = 0

        # Flush con lai
        if pending and not found_idr:
            log.warning("Khong tim thay IDR! Ghi toan bo frame")
            for pt, pd in pending: au_buf.add(pt, pd)
        au_buf.close()
        stats["access_units"] = au_buf.au_count

    return stats


# ============================================================
# Buoc 4: Wrap -> MP4
# ============================================================
def wrap_to_mp4(h264_file: str, output_mp4: str, xavc: bool, fps: Optional[str]=None):
    log.info("Dong goi H.264 -> MP4...")
    extra_fps = ["-r", fps] if fps else []
    brand = ["-brand","XAVC","-movflags","+faststart+write_colr"] if xavc \
            else ["-movflags","+faststart"]
    r = ffmpeg(
        "-err_detect","ignore_err","-fflags","+discardcorrupt",
        "-i", h264_file, "-c:v","copy",
        *extra_fps, *brand, output_mp4,
    )
    if r.returncode != 0:
        log.warning("Lan 1 that bai, thu voi genpts+igndts...")
        ffmpeg("-fflags","+genpts+igndts+discardcorrupt",
               "-err_detect","ignore_err",
               "-i", h264_file, "-c:v","copy", output_mp4)


def probe_fps(good_file: str) -> Optional[str]:
    try:
        r = ffprobe("-select_streams","v:0",
                    "-show_entries","stream=r_frame_rate",
                    "-of","json", good_file)
        fps = json.loads(r.stdout).get("streams",[{}])[0].get("r_frame_rate","")
        if fps and fps != "0/0":
            log.info(f"  FPS: {fps}"); return fps
    except Exception:
        pass
    return None


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Sua video H.264 / Sony XAVC bi hong  (v3)")
    parser.add_argument("good_file")
    parser.add_argument("bad_file")
    parser.add_argument("output")
    parser.add_argument("--strict-idr", action="store_true",
                        help="Bo frame truoc IDR dau tien")
    parser.add_argument("--xavc",   action="store_true",
                        help="Buoc che do Sony XAVC")
    parser.add_argument("--no-aud", action="store_true",
                        help="Khong chen AUD tu dong")
    parser.add_argument("--fps",    default=None,
                        help="Ghi de framerate (vd: 25, 29.97, 50)")
    parser.add_argument("--keep-tmp", action="store_true")
    parser.add_argument("--debug",    action="store_true")
    args = parser.parse_args()

    if args.debug: log.setLevel(logging.DEBUG)

    for f in [args.good_file, args.bad_file]:
        if not os.path.exists(f):
            log.error(f"File khong ton tai: {f}"); sys.exit(1)

    xavc = args.xavc or is_xavc(args.good_file) or is_xavc(args.bad_file)
    if xavc: log.info("XAVC mode ON")

    tmp = tempfile.mkdtemp(prefix="h264fix_")
    log.info(f"Tmp dir: {tmp}")

    out_h264  = os.path.join(tmp, "recovered.h264")
    out_audio = os.path.join(tmp, "recovered_audio.raw")

    try:
        log.info("=== Buoc 1: Trich SPS/PPS/AUD header ===")
        header, valid_sps, valid_pps = extract_sps_pps(args.good_file, tmp, xavc)

        log.info("=== Buoc 2: Hoc NAL profile ===")
        profiles = build_nal_profiles(args.good_file, tmp, xavc)
        if not profiles:
            log.error("Khong hoc duoc NAL profile."); sys.exit(2)

        fps = args.fps or probe_fps(args.good_file)

        log.info("=== Buoc 3: Scan & khoi phuc ===")
        log.info(f"  xavc={xavc}  inject_aud={not args.no_aud}  strict_idr={args.strict_idr}")
        stats = scan_bad_file(
            args.bad_file, profiles, header,
            valid_sps, valid_pps,
            out_h264, out_audio,
            skip_before_idr=args.strict_idr,
            xavc_mode=xavc,
            inject_aud=not args.no_aud,
        )

        log.info(f"  Ket qua: {stats}")
        if stats["found"] == 0:
            log.error("Khong tim thay NAL hop le nao."); sys.exit(3)

        log.info("=== Buoc 4: Dong goi MP4 ===")
        wrap_to_mp4(out_h264, args.output, xavc, fps)

        if os.path.exists(args.output):
            sz = os.path.getsize(args.output)
            log.info(f"OK  {args.output}  ({sz/1048576:.2f} MB)")
            log.info(f"   Access Units : {stats['access_units']}")
            log.info(f"   IDR frames   : {stats['idr']}")
            log.info(f"   AUD / SEI    : {stats['aud']} / {stats['sei']}")
            log.info(f"   SPS / PPS    : {stats['sps']} / {stats['pps']}")
            log.info(f"   Bo SPS rac   : {stats['bad_sps']}")
            log.info(f"   Bo PPS rac   : {stats['bad_pps']}")
            log.info(f"   Bo SEI rac   : {stats['bad_sei']}")
            log.info(f"   Bo slice rac : {stats['bad_slice']}")
            log.info(f"   Byte rac     : {stats['skipped_bytes']}")
        else:
            log.error("Tao MP4 that bai.")

    finally:
        if args.keep_tmp: log.info(f"Tmp: {tmp}")
        else: shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()