import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio


class AudioEmbeddingGenerator:
    """
    Audio embedding generator using HuggingFace Wav2Vec2 model.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def generate_audio_embedding(self, audio_path, target_sample_rate=16000):
        """
        Loads an audio file, resamples if necessary, computes the audio embeddings
        using Wav2Vec2, and returns a mean-pooled embedding over the time dimension.
        """
        # audio_path is a .wav file
        waveform, sample_rate = torchaudio.load(audio_path)

        # if multi-channel, convert to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # resample if needed.
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        # remove channel dimension (Wav2Vec2 expects shape (batch_size, num_samples)).
        waveform = waveform.squeeze(0)

        # process waveform with the Wav2Vec2 processor.
        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # compute the embeddings.
        with torch.no_grad():
            outputs = self.model(**inputs)
        # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.last_hidden_state
        # mean pooling over the time (sequence_length) dimension.
        audio_embedding = hidden_states.mean(dim=1).squeeze(0)
        return audio_embedding
