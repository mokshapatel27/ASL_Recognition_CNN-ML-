# ASL Recognition — CNN + MobileNetV2:
Real-time ASL Sign Language Recognition system that converts hand gestures (A–Z) into text and speech using Deep Learning, MobileNetV2, MediaPipe, and OpenCV.
This was our open-ended final year project for Computer Science (2024–25). The idea was to build something actually useful — a system that can help bridge the communication gap for the 430M+ people worldwide who depend on sign language.

## WHAT IT DOES:
Point your webcam at your hand, sign a letter, and the system recognizes it in real time, builds it into a word, and can speak it out loud. That's basically it. The tricky part was making it actually work in real conditions — bad lighting, different hand sizes, noisy backgrounds — not just on a clean dataset.
We ended up at ~94% validation accuracy with <200ms latency at 30 FPS, which we're pretty happy with for a college project.

## STATISTICS: 
Training Accuracy -> ~99% (Phase 2)
Validation Accuracy -> ~94%
Real-Time Speed -> 30FPS
Latency < 200ms
Classes26 -> (A–Z)

## TRAINED MODEL FILES: Google Drive:(https://drive.google.com/drive/folders/1j1mv2SUJ9yimuWP1KJ2j6wlXIJTpzWWc?usp=sharing)
**The trained model files are too large for GitHub so we've put them on Drive. Download whichever you need:**<br>
- best_asl_model.h5Best(overall model checkpoint) -> [Download](https://drive.google.com/file/d/1-TV51hk2753F7T1BDDO95tg1aFDJ33kv/view?usp=drive_link)<br>
- best_phase1.keras(End of Phase 1 (just the dense head trained)) -> [Download](https://drive.google.com/file/d/1R9TbUIFfCZ0L85AC1qu_AMS_sWzW0PEu/view?usp=drive_link)<br>
- best_sign_model.keras(Best sign model in Keras format) -> [Download](https://drive.google.com/file/d/12c0Ze39xpcQzqWW8iU4SOYltZ3tWnDrm/view?usp=drive_link)<br>
- SLR_final.h5(Final trained model) -> [Download](https://drive.google.com/file/d/1xCga8GR7vCW8dUnbpFhS4gMqMspdb2aE/view?usp=drive_link)<br>

## MODEL:
We used MobileNetV2 as the backbone (pretrained on ImageNet) and added our own classification head on top:
Input (224×224×3)
  → MobileNetV2 (pretrained, ImageNet)
  → Global Average Pooling
  → Dense(128) + Dropout(0.3)
  → Dense(64) + Dropout(0.2)
  → Softmax → 26 classes
  
Training was split into two phases:
- Phase 1:1–5Dense head only — fast initial accuracy gain
- Phase 2:6–35Full model fine-tuning — slower, more stable

- Optimizer: Adam (lr=0.001)
- Loss: Categorical Cross-Entropy
- Batch size: 32
- ReduceLROnPlateau to halve LR when it gets stuck

## SETUP:
pip install tensorflow opencv-python mediapipe pyttsx3 numpy matplotlib
- Then just run:
python realtime_sign.py
You'll need Python 3.8+ and a webcam. That's it.

## CONTROLS:
- SPACE -> Speak the current word
- Backspace -> Delete last letter
- C -> Clear everything
- Q -> Quit

## THE HUD:
The live window shows:
- Camera feed with hand detection overlay
- Confidence bar — green (>80%), yellow (>60%), red (<60%)
- Small ROI preview (top-right) — what the model actually sees (96×96 white canvas)
- Current word being built
- Last 5 spoken words at the bottom

## STACK:
TensorFlow · Keras · MobileNetV2 · MediaPipe · OpenCV · pyttsx3 · NumPy · Matplotlib

## FUTURE SCOPE:
- LSTM for dynamic gestures (J and Z involve movement, static frames don't capture them)
- Facial landmarks for more complete ASL understanding
- BSL / ISL support via domain adaptation
- TFLite export for mobile
- Language model for full sentence prediction instead of letter-by-letter

## Acknowledgements:
- Kaggle ASL Alphabet Dataset
- Google MediaPipe
- ImageNet


