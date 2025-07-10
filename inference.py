import torch
import librosa
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperModel, WhisperFeatureExtractor
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
import soundfile as sf
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

class Projector(nn.Module):
    def __init__(self, speech_encoder_hidden_size, llm_hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(speech_encoder_hidden_size, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, llm_hidden_size),
        )

    def forward(self, x):
        return self.proj(x)


class SLAM_ASR(torch.nn.Module):
    def __init__(self, 
                pretrained_model_name="openai/whisper-large-v3", 
                llm_model_name="google/gemma-2b-it" ,
                lora_weights_path=None):  # 👈 Add this argument
        super().__init__()

        # Load Whisper
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name)
        self.speech_encoder = WhisperModel.from_pretrained(pretrained_model_name).encoder

        # Load LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=True)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)

        # Apply LoRA config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.llm_model = get_peft_model(self.llm_model, lora_config)

        # ✅ If lora adapter weights are provided, load them here
        if lora_weights_path is not None:
            from peft import PeftModel
            self.llm_model = PeftModel.from_pretrained(self.llm_model, lora_weights_path)

        # Downstream architecture
        self.downsample_factor = 5
        self.projector = Projector(
            self.speech_encoder.config.hidden_size * self.downsample_factor,
            self.llm_model.config.hidden_size,
        )

        # Freeze Whisper (not LLM during training!)
        for param in self.speech_encoder.parameters():
            param.requires_grad = False

        self.init_prompts()

    def init_prompts(self):
        # More explicit prompt for Hindi
        user_prompt_text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTranscribe the following speech into Hindi:"
        
        # You might even refine the assistant prompt for clarity, though it's less critical than the user prompt
        assistant_prompt_text = ". Transcribe speech to text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        self.user_prompt_embeds, self.user_mask = self.get_text_embedding(user_prompt_text, return_attention_mask=True)
        self.assistant_prompt_embeds, self.assistant_mask = self.get_text_embedding(assistant_prompt_text, return_attention_mask=True)


    def forward(self, input_features):
        with torch.no_grad():
            encoded_features = self.speech_encoder(input_features)
            downsampled = self.downsample(encoded_features.last_hidden_state, k=self.downsample_factor)
            projected = self.projector(downsampled)

            inputs_embeds = torch.cat([
                self.user_prompt_embeds.to(self.llm_model.device, dtype=torch.bfloat16),
                projected,
                self.assistant_prompt_embeds.to(self.llm_model.device, dtype=torch.bfloat16)
            ], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=200,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                num_beams=5,
                early_stopping=True,
            )

        return outputs

    @staticmethod
    def downsample(features, k):
        B, T, H = features.shape
        if T % k != 0:
            raise ValueError("Sequence length must be divisible by downsample factor")
        return features.view(B, T // k, H * k)

    def get_text_embedding(self, text, return_attention_mask=False):
        tokens = self.llm_tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=1024)
        token_ids = tokens.input_ids
        attention_mask = tokens.attention_mask if return_attention_mask else None
        embeddings = self.llm_model.get_input_embeddings()(token_ids)
        return embeddings.to(self.llm_model.device), attention_mask.to(self.llm_model.device) if attention_mask is not None else None


def load_model(checkpoint_path):
    """ Load the model from a checkpoint. """
    model = SLAM_ASR()
    pretrained_dict = torch.load(checkpoint_path)
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model


def load_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=None)
    audio = librosa.to_mono(audio)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


def segment_audio(audio, segment_length=30, sr=16000):
    samples_per_segment = segment_length * sr
    return [audio[i:i + samples_per_segment] for i in range(0, len(audio), samples_per_segment)]


def transcribe_audio(model, audio_path, device='cuda'):
    audio, sr = load_audio(audio_path)
    segments = segment_audio(audio, segment_length=20, sr=sr)
    full_transcription = []

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    for segment in segments:
        audio_features = feature_extractor(segment, return_tensors="pt", sampling_rate=sr)
        input_features = audio_features.input_features.to(device, dtype=torch.bfloat16)
        outputs = model.forward(input_features)
        transcription = model.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_transcription.append(transcription)

    return ' '.join(full_transcription)


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = '/speech/soumya/slam-asr/lightning_logs/version_45/checkpoints/epoch=2-step=29955.ckpt'
    audio_path = '/speech/soumya/slam-asr/test1.wav'

    # Load model and tokenizer
    model = load_model(checkpoint_path)
    model.to(device, dtype=torch.bfloat16)
    model.eval()

    ds = load_dataset(
        "google/fleurs",
        "hi_in",
        split='test',
        cache_dir='/speech/advait/.cache/huggingface/datasets'
    )

    output_file = "output.txt"
    with open(output_file, "w") as f:
        for i in tqdm(range(len(ds))):
            sf.write('test1.wav', data=ds[i]["audio"]["array"], samplerate=16000)
            transcription = transcribe_audio(model, 'test1.wav', device)
            f.write(str(transcription) + "\n")
