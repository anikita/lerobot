"""
Coordinate Metadata Subscriber  [v1]
======================================
Subscribes to the coordinate ZMQ PUB socket (port 5558) and logs
received metadata to a CSV sidecar file alongside a recording session.

Usage:
    python coord_subscriber.py                     # default: port 5558
    python coord_subscriber.py --port 5559         # custom port
    python coord_subscriber.py --output my_log.csv # custom output file

The CSV columns:
    timestamp, target_x_mm, target_y_mm, bbox_x1, bbox_y1, bbox_x2, bbox_y2, status

Setup:
    conda activate lerobot
    pip install pyzmq
"""

import csv
import json
import time
import signal
import sys
import argparse
from pathlib import Path
from datetime import datetime

# ═════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════

DEFAULT_PORT   = 5558
DEFAULT_OUTPUT = None   # auto-generated from timestamp if None
CSV_HEADER     = [
    "timestamp",
    "target_x_mm", "target_y_mm",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "status",
]


def make_filename():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"coord_log_{ts}.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Coordinate Metadata Subscriber v1"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"ZMQ SUB port (default: {DEFAULT_PORT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output CSV file (default: auto-generated)")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else Path(make_filename())

    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")   # receive all messages
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://localhost:{args.port}")

    print(f"Subscribed to tcp://localhost:{args.port}")
    print(f"Logging to: {out_path.resolve()}")
    print("Press Ctrl+C to stop.\n")

    running = [True]

    def handle_sigint(sig, frame):
        running[0] = False
        print("\nStopping...")

    signal.signal(signal.SIGINT, handle_sigint)

    row_count = 0

    with open(str(out_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        while running[0]:
            try:
                # poll with timeout so we can check the running flag
                if sock.poll(500):  # 500ms timeout
                    msg = sock.recv_string()
                    data = json.loads(msg)

                    coord = data.get("target_coord_mm")
                    bbox  = data.get("bbox_px")
                    status = data.get("status", "")

                    row = [
                        data.get("timestamp", time.monotonic()),
                        coord[0] if coord else "",
                        coord[1] if coord else "",
                        bbox[0] if bbox else "",
                        bbox[1] if bbox else "",
                        bbox[2] if bbox else "",
                        bbox[3] if bbox else "",
                        status,
                    ]
                    writer.writerow(row)
                    row_count += 1

                    # print status line
                    if coord:
                        print(f"  [{row_count:5d}] {status:<8s}  "
                              f"plate=({coord[0]:7.1f}, {coord[1]:7.1f}) mm  "
                              f"bbox={tuple(bbox) if bbox else 'None'}")
                    else:
                        print(f"  [{row_count:5d}] {status:<8s}  (no target)")
            except zmq.Again:
                continue
            except json.JSONDecodeError:
                print("  [WARN] malformed JSON message, skipping")
                continue

    sock.close()
    ctx.term()
    print(f"\nLogged {row_count} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
