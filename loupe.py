"""
Loupe: single-image forgery detection (MD5 + OpenCV).

Two files go together: loupe.py (this) and index.html (the website).

  python loupe.py              -> serves index.html (same folder) at http://localhost:8000
  python loupe.py photo.jpg    -> analyze one image in the terminal

The website needs only Python. Terminal mode also needs:
  pip install opencv-python numpy

What it does for one image:
  1. MD5 fingerprint of the file (keep it to prove a later copy is unchanged)
  2. Error Level Analysis: re-saves the image as JPEG and maps where the
     compression error differs; pasted or retouched areas tend to stand out
  3. Copy-move check: finds regions duplicated inside the same image (ORB)
"""
import hashlib
from pathlib import Path
import http.server
import socketserver
import sys
import threading
import webbrowser

try:
    import cv2
    import numpy as np
except ImportError:  # website mode still works without these
    cv2 = np = None

PORT = 8000

HERE = Path(__file__).resolve().parent
PAGE = HERE / "index.html"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if not PAGE.exists():
            self.send_error(404, "index.html must be in the same folder as loupe.py")
            return
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def serve():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as srv:
        url = f"http://localhost:{PORT}"
        print(f"Loupe is running at {url}  (press Ctrl+C to stop)")
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def ela(img, quality=90):
    """Error Level Analysis. Returns (heatmap, signal). Signal above ~1.8 is notable."""
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    resaved = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    d = cv2.absdiff(img, resaved).max(axis=2)
    blocks = cv2.resize(d.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA).flatten()
    blocks.sort()
    signal = float(blocks[243] / (blocks[128] + 0.5))
    top = max(6.0, float(np.percentile(d, 99.5)))
    heat = cv2.applyColorMap(np.clip(d * 255.0 / top, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heat, signal


def copy_move(img, min_dist=40, max_hamming=30, min_pairs=8):
    """Find regions duplicated inside the same image using ORB keypoints."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, des = cv2.ORB_create(nfeatures=5000).detectAndCompute(gray, None)
    if des is None or len(kps) < 3:
        return img, 0
    out, pairs = img.copy(), 0
    for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des, des, k=3):
        if len(m) < 3 or m[1].distance > max_hamming:
            continue
        p1 = np.array(kps[m[1].queryIdx].pt)
        p2 = np.array(kps[m[1].trainIdx].pt)
        if np.linalg.norm(p1 - p2) < min_dist:
            continue
        pairs += 1
        cv2.line(out, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 200, 0), 1)
        cv2.circle(out, tuple(p1.astype(int)), 3, (0, 0, 255), -1)
        cv2.circle(out, tuple(p2.astype(int)), 3, (0, 0, 255), -1)
    return out, (pairs if pairs >= min_pairs else 0)


def analyze(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Cannot read image: {path}")
    print(f"MD5 fingerprint : {md5_file(path)}")
    print(f"Size            : {img.shape[1]} x {img.shape[0]} px")

    heat, signal = ela(img)
    verdict = ("looks consistent" if signal < 1.8
               else "some areas stand out" if signal < 3.2 else "strong signs of editing")
    print(f"ELA signal      : {signal:.2f}  ->  {verdict}")
    cv2.imwrite("ela_heatmap.png", heat)
    print("Saved ela_heatmap.png (brighter = more suspicious)")

    out, pairs = copy_move(img)
    if pairs:
        cv2.imwrite("copy_move_result.png", out)
        print(f"Copy-move       : SUSPICIOUS, {pairs} duplicated keypoint pairs (saved copy_move_result.png)")
    else:
        print("Copy-move       : nothing found")
    print("\nNote: these are hints, not proof. Best on JPEG photos.")


def main():
    if len(sys.argv) == 1:
        return serve()
    if cv2 is None:
        sys.exit("Terminal mode needs OpenCV:  pip install opencv-python numpy")
    analyze(sys.argv[1])


if __name__ == "__main__":
    main()
