# ✋ Aether Hands

**A real-time, sensor-free hand-tracking experience that turns bare-hand gestures in front of an ordinary webcam into glowing energy effects — built for brand activations, exhibitions, and live events.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-CV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-00A98F)](https://developers.google.com/mediapipe)

---

## 💡 About the Project

Aether Hands turns a single ordinary webcam into an interactive energy-effects experience — no gloves, no markers, no external sensors. Point, open your palm, or make a fist, and the system responds in real time with glowing visual effects layered directly onto the live camera feed.

Built with brand activations in mind: **malls, exhibitions, events, and brand booths**, where the goal isn't just to show an ad — it's to turn the visitor into part of the experience, and send them home with a shareable video of themselves in it.

## ✨ Key Features

| Feature | Description |
|---|---|
| ⚡ **Energy Ball** | Point a finger and a glowing energy orb appears, tracking your hand's movement in every direction |
| ⚡ **Lightning Mode** | Open your palm to trigger branching lightning and live particle effects erupting from your hand |
| 💥 **Power Boost** | Clench your fist to release an amplified burst of energy that fills the screen |
| 👻 **Ghost + Combo Mode** | Layer a gradual disappearing "ghost" effect on top of the energy effects simultaneously, for a combined visual experience |
| 🖥️ **Simple Launch UI** | Double-click to launch, pick a mode from the interface, and hit Start — no command-line needed for the end user |
| 📹 **Instant Local Sharing** | Hit Record, and when finished a QR code appears on screen instantly — scan it to get the video on your phone in seconds, no external servers, everything served locally over the same network |

## 🎯 Why It Works for Brand Activations

Interactive, gesture-driven experiences like this are a natural fit for:

- 🏢 Malls
- 🎪 Exhibitions
- 🎤 Live events
- 🏷️ Brand activations

Instead of a visitor passively watching an ad, they become part of the content itself — and walk away with a shareable video carrying the brand experience, ready to post on social media.

## 🛠 Tech Stack

`Python` · `OpenCV` · `MediaPipe` · `Computer Vision` · `Real-time Gesture Recognition`

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A webcam

### Installation
```bash
git clone https://github.com/yahya-waked/Magic_Hand_Project.git
cd Magic_Hand_Project
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the app
```bash
python gui_app.py
```
Pick a mode from the launch screen (Energy Ball, Lightning, Power Boost, Ghost/Combo), press **Start**, and hit **Record** to capture a clip — a QR code will appear at the end for instant local download to your phone.

## 🎥 Demo

**[Watch the demo on LinkedIn](https://www.linkedin.com/posts/yahya-waked_computervision-ai-ar-activity-7497278880030900224-UcSM)**

---

*An exploration of how Computer Vision + real-time effects can turn a simple webcam into a full interactive brand experience.*