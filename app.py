import os, json, uuid, shutil, subprocess, base64, mimetypes
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from openai import OpenAI

app = Flask(__name__)
BASE = Path('/tmp/yvens_goat')
BASE.mkdir(parents=True, exist_ok=True)
MAX_MB = int(os.getenv('MAX_VIDEO_MB','250'))
app.config['MAX_CONTENT_LENGTH'] = MAX_MB * 1024 * 1024
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

@app.get('/')
def home():
    return render_template('index.html', max_mb=MAX_MB)

@app.get('/health')
def health():
    return {'ok': True, 'ffmpeg': bool(shutil.which('ffmpeg'))}

def duration(path):
    p=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)],capture_output=True,text=True,check=True)
    return float(p.stdout.strip())

def extract_audio(video, out):
    subprocess.run(['ffmpeg','-y','-i',str(video),'-vn','-ac','1','-ar','16000','-b:a','48k',str(out)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def split_audio(audio, folder):
    folder.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg','-y','-i',str(audio),'-f','segment','-segment_time','600','-c','copy',str(folder/'audio_%03d.mp3')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return sorted(folder.glob('audio_*.mp3'))

def transcribe(parts):
    texts=[]
    for i,p in enumerate(parts):
        with p.open('rb') as f:
            r=client.audio.transcriptions.create(model='gpt-4o-mini-transcribe',file=f,response_format='text')
        texts.append(f'[AUDIO PART {i+1}]\n{r}')
    return '\n'.join(texts)

def extract_frames(video, folder, dur):
    folder.mkdir(exist_ok=True)
    # Dense enough for MVP: 1 frame every 3 sec, cap 120 frames (6 min at 3s cadence).
    interval=max(3.0, dur/120.0)
    subprocess.run(['ffmpeg','-y','-i',str(video),'-vf',f'fps=1/{interval:.3f},scale=640:-2','-q:v','4',str(folder/'frame_%04d.jpg')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    files=sorted(folder.glob('frame_*.jpg'))[:120]
    return files, interval

def data_url(p):
    return 'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()

def analyze_batches(frames, interval):
    reports=[]
    batch=12
    for start in range(0,len(frames),batch):
        group=frames[start:start+batch]
        content=[{'type':'input_text','text':f'''Analyse ces images successives d'une vidéo. Elles sont espacées d'environ {interval:.1f} secondes. Décris uniquement ce qui est visuellement observable, chronologiquement, en français. Pour chaque changement important: timestamp approximatif, personnes, actions, objets, lieu, réactions. N'invente aucun dialogue. Sois détaillé pour permettre un résumé narratif fidèle.'''}]
        for p in group:
            content.append({'type':'input_image','image_url':data_url(p),'detail':'low'})
        r=client.responses.create(model=os.getenv('VISION_MODEL','gpt-5.6-luna'),input=[{'role':'user','content':content}])
        reports.append(r.output_text)
    return '\n\n'.join(reports)

def finalize(transcript, visuals, dur):
    prompt=f'''Tu es Yvens GOAT, analyste vidéo. Durée vidéo: {dur:.1f}s.\n\nTRANSCRIPTION AUDIO COMPLÈTE (langue source potentiellement non française):\n{transcript}\n\nOBSERVATIONS VISUELLES CHRONOLOGIQUES:\n{visuals}\n\nCrée un résultat en français qui couvre toute la vidéo. Traduis fidèlement les paroles en français et fusionne-les avec les actions visibles au bon moment. Ne prétends jamais voir une action absente des observations. Si un lien précis dialogue/action est incertain, indique-le prudemment.\n\nRetourne exactement ces sections:\nTITRE\nRÉSUMÉ\nSCRIPT VOIX-OFF COMPLET\nTIMELINE DÉTAILLÉE\nDIALOGUES TRADUITS\nNOTES DE CONFIANCE\n\nLe SCRIPT VOIX-OFF COMPLET doit être naturel, narratif, prêt à coller dans ElevenLabs, et suffisamment détaillé pour raconter l'histoire entière plutôt qu'un résumé superficiel.'''
    r=client.responses.create(model=os.getenv('FINAL_MODEL','gpt-5.6-luna'),input=prompt)
    return r.output_text

@app.post('/api/analyze')
def analyze():
    if not os.environ.get('OPENAI_API_KEY'):
        return jsonify(error='OPENAI_API_KEY manke sou serveur la.'),500
    f=request.files.get('video')
    if not f or not f.filename:
        return jsonify(error='Chwazi yon videyo.'),400
    job=BASE/str(uuid.uuid4()); job.mkdir()
    try:
        ext=Path(f.filename).suffix or '.mp4'; video=job/('video'+ext); f.save(video)
        dur=duration(video)
        if dur>15*60: return jsonify(error='Premye vèsyon an limite a 15 minit pou nou valide kalite a.'),400
        audio=job/'audio.mp3'; extract_audio(video,audio)
        parts=split_audio(audio,job/'audio_parts'); transcript=transcribe(parts)
        frames,interval=extract_frames(video,job/'frames',dur); visuals=analyze_batches(frames,interval)
        result=finalize(transcript,visuals,dur)
        return jsonify(ok=True,duration=dur,frames=len(frames),audio_parts=len(parts),result=result)
    except subprocess.CalledProcessError:
        return jsonify(error='FFmpeg pa rive trete videyo sa a.'),500
    except Exception as e:
        return jsonify(error=str(e)),500
    finally:
        shutil.rmtree(job,ignore_errors=True)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.getenv('PORT','10000')))
