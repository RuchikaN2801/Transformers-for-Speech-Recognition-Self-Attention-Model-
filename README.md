
# Transformers for Speech Recognition (Self-Attention Models)
An end-to-end Automatic Speech Recognition (ASR) system developed as part of a Predictive Analytics project. This system leverages the **Wav2Vec 2.0** architecture to transcribe raw audio waveforms into accurate text using self-supervised learning and Transformer-based contextualized representations.


##  Overview
Traditional ASR systems (HMMs/GMMs) struggle with long-range dependencies. Our project utilizes the **Transformer** architecture (Vaswani et al., 2017) to capture temporal context more effectively. By employing a pre-trained **Wav2Vec 2.0** model, we demonstrate high-accuracy transcription even with minimal labeled data.

### Key Technical Specs:
* **Architecture:** CNN-based Feature Encoder + Multi-layer Transformer.
* **Loss Function:** Connectionist Temporal Classification (CTC) Loss.
* **Dataset:** [LibriSpeech ASR Corpus](https://www.openslr.org/12/) (1000h read speech).
* **Optimization:** SpecAugment, Gradient Clipping, and Cosine Annealing.


##  Performance & Results
The model was evaluated using **Word Error Rate (WER)** and **Character Error Rate (CER)**:

| Dataset Segment | WER (%) | CER (%) |
| :--- | :--- | :--- |
| **LibriSpeech (Clean)** | **5.2%** | 2.4% |
| **LibriSpeech (Noisy)** | **12.1%** | 6.3% |


##  Installation

```bash
# Install required libraries
pip install torch torchaudio transformers datasets librosa
```

##  Quick Start (Inference)

You can use the following snippet to transcribe a local `.wav` file:
```python
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Load and resample audio to 16kHz
audio_path = "path_to_your_audio.wav"
speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

# Process and Predict
input_values = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values
with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print(f"Transcription: {transcription[0]}")
```


##  Preprocessing Pipeline
1. **Resampling:** All audio converted to 16kHz.
2. **Normalization:** Loudness levels adjusted for consistency.
3. **Silence Trimming:** Removed leading/trailing silences to optimize processing.
4. **Tokenization:** Text transcriptions normalized (lowercase/no punctuation) and tokenized for CTC.

##  Future Scope
* **Multilingual Support:** Extending the model to handle languages beyond English.
* **Edge Deployment:** Optimizing for real-time inference on mobile devices.
* **Noise Robustness:** Further fine-tuning on heavily augmented noisy data.

##  References
* [Wav2Vec 2.0 Paper (Baevski et al.)](https://arxiv.org/abs/2006.11477)
* [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
* [HuggingFace Wav2Vec2 Documentation](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
```
