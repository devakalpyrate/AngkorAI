import os
import re
import json
import time
import secrets
import string
import logging
from collections import defaultdict
from threading import Lock
from urllib.parse import urlparse, parse_qs, quote_plus
import requests
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from dotenv import load_dotenv

from redis_cache import (
    get_redis_client,
    get_cached_transcript,
    get_cached_analysis,
    set_cached_analysis,
    set_sync_payload,
    get_sync_payload
)
from transcripts import get_transcript_with_fallback
from llm import generate_video_intelligence, chat_with_transcript
from pdf_generator import generate_pdf_report

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, 'static'),
    template_folder=os.path.join(BASE_DIR, 'templates')
)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/static/<path:filename>')
def serve_static(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(BASE_DIR, 'static'), filename)

@app.route('/favicon.ico')
def favicon():
    from flask import send_from_directory
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

app.secret_key = os.environ.get('FLASK_SECRET_KEY') or os.urandom(32).hex()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'Request payload too large. Maximum size is 16 MB.'}), 413

@app.after_request
def set_security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), camera=(self), microphone=()'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
        "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net; "
        "img-src 'self' data: blob: https://i.ytimg.com https://*.ytimg.com https://images.unsplash.com; "
        "connect-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "frame-ancestors 'none';"
    )
    return response

class SimpleRateLimiter:
    def __init__(self, max_requests=10, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        with self.lock:
            cutoff = now - self.window_seconds
            timestamps = [t for t in self.requests[client_ip] if t > cutoff]
            if len(timestamps) >= self.max_requests:
                self.requests[client_ip] = timestamps
                return False
            timestamps.append(now)
            self.requests[client_ip] = timestamps
            return True

rate_limiter = SimpleRateLimiter(max_requests=10, window_seconds=60)

def generate_lecture_id():
    alphabet = string.ascii_letters + string.digits
    res = ''.join(secrets.choice(alphabet) for _ in range(20))
    if not any(c.isdigit() for c in res):
        res = res[:10] + secrets.choice(string.digits) + res[11:]
    if not any(c.isalpha() for c in res):
        res = res[:5] + secrets.choice(string.ascii_letters) + res[6:]
    return res

def extract_video_id(url):
    if not url or not isinstance(url, str):
        return None
    url = url.strip()

    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url

    try:
        parsed = urlparse(url if '://' in url else 'https://' + url)
        netloc = (parsed.netloc or '').lower()

        if 'youtube.com' in netloc:
            if parsed.path == '/watch':
                qs = parse_qs(parsed.query)
                if 'v' in qs and qs['v']:
                    vid = qs['v'][0]
                    if re.match(r'^[a-zA-Z0-9_-]{11}$', vid):
                        return vid
            parts = [p for p in parsed.path.split('/') if p]
            if parts and parts[0] in ('shorts', 'embed', 'v', 'live') and len(parts) >= 2:
                vid = parts[1]
                if re.match(r'^[a-zA-Z0-9_-]{11}$', vid):
                    return vid

        elif 'youtu.be' in netloc:
            parts = [p for p in parsed.path.split('/') if p]
            if parts:
                vid = parts[0].split('?')[0]
                if re.match(r'^[a-zA-Z0-9_-]{11}$', vid):
                    return vid
    except Exception:
        pass

    patterns = [
        r'(?:https?:\/\/)?(?:www\.|m\.)?youtube\.com\/watch\?.*?\bv=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/(?:embed|v|shorts|live)\/([a-zA-Z0-9_-]{11})',
        r'(?:v=|\/v\/|\/embed\/|\/shorts\/|\/live\/|youtu\.be\/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_metadata(url):
    metadata = {
        'title': 'YouTube Video',
        'author': 'Unknown Creator',
        'thumbnail': None,
        'duration': None
    }

    video_id = extract_video_id(url)
    if not video_id:
        return metadata
    safe_canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    is_serverless = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
    timeout_sec = 4 if is_serverless else 10

    try:
        encoded_url = quote_plus(safe_canonical_url)
        response = requests.get(f"https://www.youtube.com/oembed?url={encoded_url}&format=json", timeout=timeout_sec)
        if response.status_code == 200:
            data = response.json()
            metadata['title'] = data.get('title', 'YouTube Video')
            metadata['author'] = data.get('author_name', 'Unknown Creator')
            metadata['thumbnail'] = data.get('thumbnail_url')
    except Exception as e:
        logging.warning("Error fetching oEmbed metadata: %s", e)

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        html_response = requests.get(safe_canonical_url, headers=headers, timeout=timeout_sec)
        if html_response.status_code == 200:
            duration_match = re.search(r'<meta itemprop="duration" content="([^"]+)">', html_response.text)
            if duration_match:
                duration_str = duration_match.group(1)
                parsed = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
                if parsed:
                    hours = int(parsed.group(1) or 0)
                    minutes = int(parsed.group(2) or 0)
                    seconds = int(parsed.group(3) or 0)
                    metadata['duration'] = hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        logging.warning("Error fetching exact duration: %s", e)

    supadata_key = (os.environ.get('SUPADATA_API', '') or os.environ.get('SUPADATA_API_KEY', '')).strip()
    if supadata_key and (metadata['duration'] is None or metadata['title'] == 'YouTube Video'):
        try:
            logging.info("Attempting Supadata universal metadata fallback for %s...", video_id)
            meta_resp = requests.get(
                f"https://api.supadata.ai/v1/metadata?url={quote_plus(safe_canonical_url)}",
                headers={'x-api-key': supadata_key},
                timeout=8 if is_serverless else 12
            )
            if meta_resp.status_code == 200:
                s_data = meta_resp.json()
                if isinstance(s_data, dict):
                    if s_data.get('title') and metadata['title'] == 'YouTube Video':
                        metadata['title'] = s_data['title']
                    if s_data.get('author') and metadata['author'] == 'Unknown Creator':
                        metadata['author'] = s_data['author']
                    if s_data.get('duration') and metadata['duration'] is None:
                        try:
                            metadata['duration'] = float(s_data['duration'])
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            logging.warning("Supadata metadata fallback error: %s", e)

    return metadata

@app.route('/')
@app.route('/home')
@app.route('/home/<lecture_id>')
def home(lecture_id=None):
    return render_template('index.html')

@app.route('/about')
def about():
    return redirect(url_for('home'))

@app.route('/api/generate-pdf', methods=['POST'])
@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.json or {}
    markdown_text = data.get('markdown') or ''
    metadata = data.get('metadata') or {}
    engine_name = data.get('engine') or 'Angkor Executive AI'

    if not markdown_text.strip():
        return jsonify({'error': 'Markdown content is required to generate PDF.'}), 400

    try:
        import io
        pdf_bytes = generate_pdf_report(metadata, markdown_text, engine_name)
        title = metadata.get('title') or 'report'
        clean_title = re.sub(r'[^a-zA-Z0-9_-]', '_', title)[:32].lower()
        filename = f"angkor_report_{clean_title}.pdf"

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logging.error("Server-side PDF generation error: %s", e, exc_info=True)
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def chat():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if client_ip:
        client_ip = client_ip.split(',')[0].strip()
    if not rate_limiter.is_allowed(client_ip or 'unknown'):
        return jsonify({'error': 'Rate limit exceeded. Please wait a moment.'}), 429

    data = request.json or {}
    video_id = data.get('video_id', '').strip()
    message = data.get('message', '').strip()
    history = data.get('history', [])

    if not video_id:
        return jsonify({'error': 'video_id is required.'}), 400
    if not message:
        return jsonify({'error': 'Message content cannot be empty.'}), 400

    cached_transcript = get_cached_transcript(video_id)
    transcript_text = cached_transcript.get('text', '') if cached_transcript else ''

    if not transcript_text:
        transcript_text = data.get('transcript_text', '').strip()

    if not transcript_text:
        return jsonify({'error': 'Transcript context not found for this video. Please analyze the video first.'}), 404

    keys_dict = {
        'nvidia': os.environ.get('NVIDIA_API_KEY', '').strip(),
        'gemini': os.environ.get('GEMINI_API_KEY', '').strip(),
        'groq': os.environ.get('GROQ_API_KEY', '').strip()
    }

    try:
        reply = chat_with_transcript(transcript_text, message, history, keys_dict)
        return jsonify({'reply': reply})
    except Exception as e:
        logging.error("Chat route error: %s", e, exc_info=True)
        return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
@app.route('/analyze', methods=['POST'])
def analyze():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if client_ip:
        client_ip = client_ip.split(',')[0].strip()
    if not rate_limiter.is_allowed(client_ip or 'unknown'):
        return jsonify({'error': 'Rate limit exceeded. Please wait a moment before analyzing another lecture.'}), 429

    data = request.json or {}
    url = data.get('url')
    if not url or not isinstance(url, str):
        return jsonify({'error': 'A valid YouTube URL is required.'}), 400

    url = url.strip()
    if len(url) > 500:
        return jsonify({'error': 'URL length exceeds maximum permitted limit.'}), 400

    video_id = extract_video_id(url)
    if not video_id or not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return jsonify({'error': 'Invalid YouTube URL or Video ID. Please provide a legitimate link.'}), 400

    safe_canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    cached_payload = get_cached_analysis(video_id)
    if cached_payload and cached_payload.get('analysis'):
        logging.info("Full Analysis Cache HIT for %s: serving instantly from Redis/Memory.", video_id)
        cached_payload['cached'] = True
        return jsonify(cached_payload)

    user_sarvam_key = os.environ.get('SARVAM_API_KEY', '').strip()
    user_groq_key = os.environ.get('GROQ_API_KEY', '').strip()
    user_nvidia_key = os.environ.get('NVIDIA_API_KEY', '').strip()
    user_gemini_key = os.environ.get('GEMINI_API_KEY', '').strip()
    user_supadata_key = (os.environ.get('SUPADATA_API', '') or os.environ.get('SUPADATA_API_KEY', '')).strip()
    sarvam_model = os.environ.get('SARVAM_MODEL', 'sarvam-105b-conversations')

    try:
        metadata = get_youtube_metadata(safe_canonical_url)
        metadata['video_id'] = video_id
        metadata['url'] = safe_canonical_url

        video_duration = metadata.get('duration')
        if video_duration is not None and float(video_duration) > 1220:
            return jsonify({
                'error': 'Video is too long! We currently cannot generate notes for videos exceeding 20 minutes and 20 seconds.'
            }), 400

        try:
            transcript_data = get_transcript_with_fallback(
                video_id=video_id,
                canonical_url=safe_canonical_url,
                supadata_key=user_supadata_key,
                groq_key=user_groq_key
            )
        except Exception as transcript_err:
            logging.error("Transcript retrieval failed: %s", transcript_err)
            return jsonify({
                'error': f'Failed to retrieve video transcript: {str(transcript_err)}'
            }), 400

        if not transcript_data or not transcript_data.get('text'):
            return jsonify({
                'error': 'Could not extract transcript for this video. Please check if the video has speech or captions.'
            }), 400

        if (metadata.get('duration') is None or metadata.get('duration') == 0) and transcript_data.get('segments'):
            try:
                last_segment = transcript_data['segments'][-1]
                metadata['duration'] = float(last_segment.get('start', 0)) + float(last_segment.get('duration', 0))
            except Exception:
                pass

        video_duration = metadata.get('duration')
        if video_duration is None:
            return jsonify({
                'error': 'Could not safely determine video duration. Processing aborted to protect server limits.'
            }), 400

        if float(video_duration) > 1220:
            return jsonify({
                'error': 'Video is too long! We currently cannot generate notes for videos exceeding 20 minutes and 20 seconds.'
            }), 400

        keys_dict = {
            'groq': user_groq_key,
            'gemini': user_gemini_key,
            'nvidia': user_nvidia_key,
            'sarvam': user_sarvam_key
        }

        analysis_result, analysis_engine = generate_video_intelligence(
            transcript_text=transcript_data['text'],
            keys=keys_dict,
            sarvam_model=sarvam_model
        )

        lecture_id = generate_lecture_id()
        response_payload = {
            'lecture_id': lecture_id,
            'metadata': metadata,
            'transcript': transcript_data,
            'analysis': analysis_result,
            'engine': analysis_engine
        }

        try:
            set_cached_analysis(video_id, response_payload)
        except Exception as cache_err:
            logging.warning("Failed to store analysis in Redis for %s: %s", video_id, cache_err)

        return jsonify(response_payload)

    except Exception as e:
        logging.error("Analysis route error: %s", e, exc_info=True)
        return jsonify({'error': f'An unexpected error occurred while analyzing the lecture: {str(e)}'}), 500

@app.route('/api/vault/recent', methods=['GET'])
@app.route('/vault/recent', methods=['GET'])
def get_recent_vault():
    client = get_redis_client()
    if not client:
        return jsonify({'success': False, 'lectures': []})

    try:
        keys = client.keys('analysis:*')
        lectures = []
        for k in keys:
            raw = client.get(k)
            if raw:
                data = json.loads(raw) if isinstance(raw, str) else raw
                if data and isinstance(data, dict) and data.get('metadata') and data.get('analysis'):
                    vid = data['metadata'].get('video_id') or k.replace('analysis:', '')
                    stored_id = data.get('lecture_id') or generate_lecture_id()
                    lectures.append({
                        'lecture_id': stored_id,
                        'metadata': data['metadata'],
                        'transcript': data.get('transcript', {}),
                        'analysis': data['analysis'],
                        'engine': data.get('engine', 'Angkor AI')
                    })
        return jsonify({'success': True, 'lectures': lectures})
    except Exception as e:
        logging.warning("Error fetching recent vault from Redis: %s", e)
        return jsonify({'success': False, 'error': str(e), 'lectures': []})

@app.route('/api/sync/create', methods=['POST'])
@app.route('/sync/create', methods=['POST'])
def create_sync_code():
    data = request.get_json(silent=True) or {}
    payload = data.get('payload')
    if not payload:
        return jsonify({'error': 'No payload provided for cross-browser sync.'}), 400

    import secrets
    code_raw = str(secrets.randbelow(900000) + 100000)
    formatted_code = f"{code_raw[:3]}-{code_raw[3:]}"

    vault_token = f"ak_v1_{secrets.token_hex(16)}"

    ttl_seconds = 180
    set_sync_payload(code_raw, payload, ttl_seconds=ttl_seconds)
    set_sync_payload(formatted_code, payload, ttl_seconds=ttl_seconds)
    set_sync_payload(vault_token, payload, ttl_seconds=ttl_seconds)

    return jsonify({
        'success': True,
        'code': formatted_code,
        'raw_code': code_raw,
        'vault_token': vault_token,
        'expires_in_seconds': ttl_seconds
    })

@app.route('/api/sync/fetch', methods=['GET', 'POST'])
@app.route('/sync/fetch', methods=['GET', 'POST'])
def fetch_sync_code():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        code = data.get('vault') or data.get('code') or ''
    else:
        code = request.args.get('vault') or request.args.get('code') or ''

    code_str = str(code).strip()
    if not code_str:
        return jsonify({'error': 'Please provide a valid sync code or encrypted vault token.'}), 400

    payload = get_sync_payload(code_str)

    if not payload:
        clean_code = re.sub(r'[^a-zA-Z0-9_]', '', code_str)
        payload = get_sync_payload(clean_code)

    if not payload:
        raw_lookup = re.sub(r'[^a-zA-Z0-9]', '', code_str).upper()
        if raw_lookup.startswith('AKQR'):
            raw_lookup = raw_lookup[4:]
        try:
            bin_str = bin(int(raw_lookup, 16))[2:]
            dec_num = str(int(bin_str, 2)).zfill(6)
            if dec_num:
                payload = get_sync_payload(dec_num)
        except Exception:
            pass

    if not payload:
        return jsonify({'error': 'This sync link or code has expired (3-minute limit) or is invalid.'}), 404

    return jsonify({
        'success': True,
        'payload': payload
    })

@app.route('/api/index', methods=['GET', 'POST'])
@app.route('/api/index.py', methods=['GET', 'POST'])
@app.route('/api', methods=['GET', 'POST'])
def vercel_entrypoint():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        if 'url' in data:
            return analyze()
        elif 'message' in data and 'video_id' in data:
            return chat()
        elif 'markdown' in data:
            return generate_pdf()
        elif 'payload' in data or data.get('sync_action') == 'create':
            return create_sync_code()
        elif 'code' in data or 'vault' in data or data.get('sync_action') == 'fetch':
            return fetch_sync_code()
        return jsonify({'error': 'Unrecognized request payload'}), 400

    if 'recent' in request.path or request.args.get('vault_action') == 'recent':
        return get_recent_vault()
    if request.args.get('sync_action') == 'fetch' or (request.args.get('code') and request.path.startswith('/api/sync')):
        return fetch_sync_code()

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1', 'yes')
    app.run(debug=debug_mode, host=host, port=port)
