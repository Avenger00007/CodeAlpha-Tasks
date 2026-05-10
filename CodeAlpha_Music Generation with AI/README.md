<div align="center">

# 🎵 Music Generation with AI

### Deep Learning · LSTM Neural Network · MIDI Output · CPU Friendly

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-LSTM-red?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![music21](https://img.shields.io/badge/music21-MIDI-green?style=for-the-badge)](https://web.mit.edu/music21/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

<br/>

> **An AI that listens to music → learns its patterns → composes brand-new pieces 🎼**

<br/>

```
📁 MIDI Files  →  🔧 Extract Notes  →  🔢 Encode  →  🧠 Train LSTM  →  🎶 Generate  →  💾 Save MIDI
```

</div>

---

## 📌 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🧠 How It Works](#-how-it-works)
- [📂 Project Structure](#-project-structure)
- [⚙️ Setup & Installation](#️-setup--installation)
- [▶️ Running the Project](#️-running-the-project)
- [🎸 Changing Instruments](#-changing-instruments)
- [🎧 Listening to the Output](#-listening-to-the-output)
- [📊 Output Files](#-output-files)
- [🛠️ Technologies Used](#️-technologies-used)
- [🤝 Contributing](#-contributing)

---

## 🎯 Project Overview

This project builds an **AI Music Composer** using **LSTM (Long Short-Term Memory)** deep learning. The model is trained on MIDI music data and learns musical patterns — which notes follow which, how chords progress, and how melodies flow. After training, it generates **completely original music** and saves it as a playable MIDI file.

> 💡 Think of it like teaching a student: expose them to thousands of songs, and they start writing their own — without being given the rules.

### ✨ Key Features

- 🎼 **Collects & preprocesses** real MIDI music data (classical, jazz)
- 🔧 **Extracts note sequences** using `music21` library
- 🧠 **Trains a stacked LSTM model** to learn musical patterns
- 🎶 **Generates new music** from a learned seed sequence
- 🎸 **Supports multiple instruments** — Piano, Guitar, Violin, Saxophone & more
- 💾 **Exports to MIDI** — playable in any media player
- 📊 **Visualizes** training loss and piano roll of generated music
- 💻 **100% CPU friendly** — no GPU required!

---

## 🧠 How It Works

### The Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  MIDI Files │───▶│ Extract Notes│───▶│  Encode to  │
│ (Classical) │    │  (music21)   │    │   Numbers   │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                              │
┌─────────────┐    ┌──────────────┐    ┌──────▼──────┐
│  Save MIDI  │◀───│   Generate   │◀───│ Train LSTM  │
│   Output    │    │  New Music   │    │   Model     │
└─────────────┘    └──────────────┘    └─────────────┘
```

### What is LSTM?

**LSTM (Long Short-Term Memory)** is a type of Recurrent Neural Network (RNN) that is excellent at learning from **sequences** — data that comes in order, one item after another.

Music is a perfect sequence problem:
```
C4 → E4 → G4 → C5 → E5 → ...
```
Each note depends on the notes before it. LSTM **remembers** this history and uses it to predict the next note.

### Model Architecture

```
Input (50 notes)
      │
      ▼
┌─────────────────┐
│  LSTM Layer 1   │  256 units  →  return_sequences=True
│    Dropout 30%  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LSTM Layer 2   │  256 units  →  return_sequences=True
│    Dropout 30%  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LSTM Layer 3   │  128 units  →  return_sequences=False
│    Dropout 30%  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dense Output   │  n_vocab units
│    Softmax      │  → probability of each note
└─────────────────┘
         │
         ▼
  Predicted Note 🎵
```

### Temperature — Controlling Creativity

| Temperature | Behaviour | Result |
|:-----------:|-----------|--------|
| `0.5` 🧊 | Very safe, picks most likely notes | Repetitive but musical |
| `0.9` ⚖️ | Balanced creativity | Natural flowing melody |
| `1.3` 🔥 | Takes risks, unusual note choices | Experimental & creative |

---

## 📂 Project Structure

```
CodeAlpha/
│
├── 📓 music_generation_ai.ipynb    ← Main Jupyter Notebook
│
├── 📁 midi_data/                   ← Training MIDI files (auto-generated)
│   ├── synthetic_1.mid
│   ├── synthetic_2.mid
│   ├── synthetic_3.mid
│   ├── synthetic_4.mid
│   └── synthetic_5.mid
│
├── 📁 output/                      ← All generated outputs
│   ├── 🎵 generated_music.mid      ← Main AI composition
│   ├── 🎵 generated_conservative.mid
│   ├── 🎵 generated_balanced.mid
│   ├── 🎵 generated_creative.mid
│   ├── 🎵 generated_guitar.mid     ← Instrument variation
│   ├── 📊 piano_roll.png           ← Visual of generated notes
│   ├── 📉 training_loss.png        ← Model learning curve
│   ├── 💾 notes.pkl                ← Extracted note data
│   ├── 💾 note_to_int.pkl          ← Note encoding map
│   ├── 💾 int_to_note.pkl          ← Note decoding map
│   ├── 💾 music_model_final.keras  ← Trained AI model
│   └── 📁 checkpoints/
│       └── best_model.keras        ← Best checkpoint
│
├── 📁 music_ai_env/                ← Virtual environment
└── 📄 README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python `3.9` / `3.10` / `3.11` / `3.13`
- VS Code ([download here](https://code.visualstudio.com/))
- Windows PowerShell

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/music-generation-ai.git
cd music-generation-ai
```

### Step 2 — Create Virtual Environment

```bash
python -m venv music_ai_env
```

### Step 3 — Activate Virtual Environment

**Windows (PowerShell):**
```powershell
music_ai_env\Scripts\activate
```

> ⚠️ If you get a scripts error, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Mac / Linux:**
```bash
source music_ai_env/bin/activate
```

### Step 4 — Install Jupyter Kernel

```bash
pip install ipykernel jupyter
python -m ipykernel install --user --name=music_ai_env --display-name "Music AI"
```

### Step 5 — Open in VS Code

1. Open VS Code → **File → Open Folder** → select project folder
2. Open `music_generation_ai.ipynb`
3. Top-right corner → click kernel selector → choose **"Music AI"**

---

## ▶️ Running the Project

Run each cell **top to bottom** using `Shift + Enter`:

| Cell | Step | What It Does | Time |
|------|------|-------------|------|
| `[0]` | 📦 Install Libraries | Installs all required packages | ~3–5 min |
| `[1]` | 📚 Import Libraries | Loads tools into memory | ~5 sec |
| `[2]` | 🎼 Download MIDI Data | Downloads or generates MIDI files | ~1 min |
| `[3]` | 🔧 Extract Notes | Parses notes from MIDI files | ~1 min |
| `[4]` | 🔢 Prepare Sequences | Encodes notes and creates training data | ~30 sec |
| `[5]` | 🧠 Build Model | Creates the LSTM architecture | ~5 sec |
| `[6]` | 🏋️ Train Model | Trains on music data — **slowest step** | ~10–40 min |
| `[7]` | 📈 Plot Loss | Draws the training loss chart | ~5 sec |
| `[8]` | 🎶 Generate Music | Creates 150 new notes | ~2 min |
| `[9]` | 💾 Save MIDI | Converts notes → MIDI file | ~10 sec |
| `[10]` | 🎹 Piano Roll | Visualizes the generated music | ~5 sec |
| `[11]` | 🔊 Play Music | Plays the MIDI file | varies |
| `[12]` | ✅ Summary | Prints project summary | ~2 sec |

> 💡 **Pro Tip:** Use **Kernel → Restart & Run All** to run everything at once!

---

## 🎸 Changing Instruments

Add a new cell at the bottom of the notebook and change **only one word**:

```python
# ============================================================
# 🎸 CHANGE ONLY THIS ONE LINE:
MY_INSTRUMENT = "Guitar"   # ← Change this!
# ============================================================

instrument_map = {
    "Piano"         : (instrument.Piano(),          0),
    "Guitar"        : (instrument.Guitar(),         25),
    "ElectricGuitar": (instrument.ElectricGuitar(), 27),
    "Violin"        : (instrument.Violin(),         40),
    "Trumpet"       : (instrument.Trumpet(),        56),
    "Saxophone"     : (instrument.Saxophone(),      65),
    "Flute"         : (instrument.Flute(),          73),
    "Clarinet"      : (instrument.Clarinet(),       71),
    "Harp"          : (instrument.Harp(),           46),
    "Cello"         : (instrument.Violoncello(),    42),
}

sel_instr, prog_num = instrument_map.get(MY_INSTRUMENT, instrument_map["Piano"])
sel_instr.midiProgram = prog_num

out_notes = []
offset = 0
for pattern in generated_notes:
    if "." in pattern:
        chord_list = []
        for p in pattern.split("."):
            try: chord_list.append(note.Note(int(p)))
            except: pass
        if chord_list:
            c = chord.Chord(chord_list)
            c.offset = offset
            out_notes.append(c)
    elif pattern in ("R", "rest"):
        r = note.Rest(); r.offset = offset; out_notes.append(r)
    else:
        try:
            n = note.Note(pattern); n.offset = offset; out_notes.append(n)
        except: pass
    offset += random.choice([0.5, 0.5, 0.5, 1.0, 0.25])

midi_part = stream.Part()
midi_part.insert(0, sel_instr)
for n in out_notes: midi_part.append(n)
score = stream.Score()
score.append(midi_part)
out_path = f"output/generated_{MY_INSTRUMENT.lower()}.mid"
score.write("midi", fp=out_path)
print(f"✅ Saved: {out_path}  |  🎸 Instrument: {MY_INSTRUMENT}  (Program #{prog_num})")
```

### Available Instruments

| Emoji | Instrument | Code to Use |
|-------|-----------|-------------|
| 🎹 | Piano | `"Piano"` |
| 🎸 | Acoustic Guitar | `"Guitar"` |
| 🎸 | Electric Guitar | `"ElectricGuitar"` |
| 🎻 | Violin | `"Violin"` |
| 🎺 | Trumpet | `"Trumpet"` |
| 🎷 | Saxophone | `"Saxophone"` |
| 🪗 | Flute | `"Flute"` |
| 🎵 | Clarinet | `"Clarinet"` |
| 🎵 | Harp | `"Harp"` |
| 🎻 | Cello | `"Cello"` |

---

## 🎧 Listening to the Output

| Method | Steps | Hears Correct Instrument? |
|--------|-------|:-------------------------:|
| **Windows Media Player** | Double-click `.mid` file | ⚠️ Sometimes |
| **VLC Media Player** | Drag `.mid` into VLC | ✅ Yes |
| **[onlinesequencer.net](https://onlinesequencer.net/import)** | Upload `.mid` → Press Play | ✅ Yes |
| **[musescore.com](https://musescore.com)** | Import MIDI | ✅ Yes |

> 💡 **Recommended:** Use **VLC** or **onlinesequencer.net** to hear the correct instrument sound!

---

## 📊 Output Files

After running all cells:

```
output/
├── 🎵 generated_music.mid          → Main AI-composed piece
├── 🎵 generated_conservative.mid   → Temperature 0.5 — safe & structured
├── 🎵 generated_balanced.mid       → Temperature 0.9 — natural melody
├── 🎵 generated_creative.mid       → Temperature 1.3 — experimental
├── 📊 piano_roll.png               → Visual piano roll of notes
└── 📉 training_loss.png            → AI learning curve chart
```

---

## 🛠️ Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| `Python` | 3.9+ | Core programming language |
| `TensorFlow` | 2.x | Deep learning framework |
| `Keras` | Built-in | LSTM model building |
| `music21` | Latest | MIDI parsing and generation |
| `NumPy` | Latest | Numerical operations |
| `Matplotlib` | Latest | Plotting and visualization |
| `Pygame` | Latest | MIDI audio playback |
| `tqdm` | Latest | Progress bars |
| `requests` | Latest | Downloading MIDI files |

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas to extend this project:

- [ ] Add more MIDI training data (jazz, pop, folk)
- [ ] Try GAN (Generative Adversarial Network) instead of LSTM
- [ ] Add a web UI to generate music in the browser
- [ ] Export to MP3/WAV using FluidSynth
- [ ] Add rhythm/tempo variation
- [ ] Train on genre-specific data

---

<div align="center">

### 🎉 Made with ❤️ as part of CodeAlpha Internship

**If you found this project helpful, please ⭐ star this repository!**

</div>
