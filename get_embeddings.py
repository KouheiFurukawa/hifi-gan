from resnet import Encoder
from meldataset import *
from tqdm import tqdm

model = Encoder().to('cuda')
encoder_state = torch.load('/data/unagi0/furukawa/cp_hifigan_static/m_00300000')['model']
model.load_state_dict(encoder_state)
model.eval()
os.makedirs('/data/unagi0/furukawa/musicnet_static_emb_10sec', exist_ok=True)
for i in range(21):
    os.makedirs('/data/unagi0/furukawa/musicnet_static_emb_10sec/{}'.format(str(i)), exist_ok=True)

for filename in tqdm(glob('/data/unagi0/furukawa/musicnet_wav_10sec/*/*.wav')):
    audio, sampling_rate = load_wav(filename)
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    if sampling_rate != 16000:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, 16000))
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    max_audio_start = audio.size(1) - 81920
    audio_start = random.randint(0, max_audio_start)
    audio = audio[:, audio_start:audio_start+81920]
    audio = mel_spectrogram(audio, 1024, 80, 16000, 256, 1024, 0, 8000, center=False).to('cuda')
    with torch.no_grad():
        emb = model(audio.unsqueeze(0))
        np.save(filename.replace('_wav_', '_static_emb_'), emb.cpu().numpy())
