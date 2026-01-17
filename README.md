# NeuTTS for Unity (Open-Phonemizer Edition)

High-performance, on-device Text-to-Speech implementation for Unity. This version replaces `espeak` with an **ONNX-based phonemizer** pipeline and uses **NeuCodec** for high-fidelity audio reconstruction.

## 🚀 Key Features
* **Engine:** English-only synthesis.
* **No Dependencies:** No system-level installation of eSpeak required.
* **Inference:** Fully local using GGUF for the backbone and ONNX for the codec and phonemizer.

---

## 📦 1. Required Files
Place the following files into your project's `Assets/StreamingAssets` folder.

### A. Synthesis & Codec (Neuphonic)
* **Backbone:** A Neuphonic GGUF model (e.g., `neutts-air-q4.gguf` or any variant from Neuphonic).
* **Codec Decoder:** `decoder_model.onnx` (This is the **NeuCodec ONNX** model).

### B. Phonemizer ([lookbe/open-phonemizer-onnx](https://huggingface.co/lookbe/open-phonemizer-onnx))
* `model.onnx`
* `tokenizer.json`
* `phoneme_dict.json`

### C. Voice Reference
* `sample.json` (Reference voice codes converted from `.pt`).
* `sample_transcript.txt` (The matching text for the reference voice).

---

## 🛠 2. Preparation: Reference Conversion
Unity cannot read PyTorch `.pt` files directly. Use this Python script to convert your reference voice codes to plain JSON before moving them to `StreamingAssets`:

```python
import torch
import json

def convert_pt_to_json(input_file, output_file):
    # Load the torch tensor
    data = torch.load(input_file, map_location='cpu')
    
    # Extract codes and convert to list for JSON compatibility
    if isinstance(data, torch.Tensor):
        data = data.detach().numpy().tolist()
    
    with open(output_file, 'w') as f:
        json.dump({"codes": data}, f)
    print(f"Success: {output_file} created.")
```

## 🖥️ How to Run

1.  **Prepare StreamingAssets:** Ensure all required files (GGUF, NeuCodec ONNX, Phonemizer ONNX, tokenizer, dict, and reference JSON) are placed inside the `Assets/StreamingAssets` folder.
2.  **Open Scene:** Open the `BasicNeuTTS` scene in the Unity Editor.
3.  **Find NeuTTS Object:** Select the **NeuTTS** GameObject in the Hierarchy window.
4.  **Update Paths:** In the Inspector, locate the file path fields. Enter the paths **relative to StreamingAssets**.
    * **Example 1 (Root):** If the file is directly in StreamingAssets, just enter the name: `neutts-air-q4.gguf`
    * **Example 2 (Folder):** If the file is in a folder, include the folder name: `Models/decoder_model.onnx`
5.  **Press Play:** Click the **Play** button in Unity. The system will initialize the models and process the text using the local English-only pipeline.

---

5.  **Press Play:** Click the **Play** button in Unity. The system will initialize the models and process the text using the local English-only pipeline.

# convert_pt_to_json('reference_voice.pt', 'sample.json')
