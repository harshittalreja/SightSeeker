# 🏆 SightSeeker — Live Demo Pitch & Presentation Guide
> **Event:** Hack in Hills '26 — Galgotias University
> **Track:** 01 · Vision-Based Autonomous Robot
> **Presenter:** HARSHIT (Solo Participant)
> **Duration:** 3–5 minutes (pitch + live demo), then Q&A

---

## ⏱️ Timeline

| Time | Phase | What's on screen |
|---|---|---|
| **0:00 – 1:30** | Slides 1–7 | `launch.bat` → **1** (HTML deck) |
| **1:30 – 3:30** | **Live demo** | `launch.bat` → **2** (3D simulator), then **4** (Vision Brain on the webcam) |
| **3:30 – 5:00** | Q&A | This sheet ⬇️ |

**Deck keys (`SightSeeker_Pitch.html`, launch.bat → 1):** `→` / `Space` / clicker next · `←` back · `F` fullscreen · `H` speaker notes on screen · `M` sound on/off. Every slide's script is in its speaker notes (press `H`).

---

## 🧾 The numbers (all real, all reproducible)

| Claim | Number | How to show it live |
|---|---|---|
| 3D missions completed | **1,000 / 1,000 (100%)** randomized warehouses | 3D sim → **📊 Trials** (runs 50 new ones in ~5 s) |
| Collisions | **0** in those 1,000 missions | same panel |
| Median time to goal | **51 s robot-time** in a 120 m arena (~17 s on screen at ×3) | timer in the top bar |
| Smart search | **15% faster** than plain frontier search on the same 1,000 warehouses (median 51 s vs 60 s, 10% shorter paths) | 📊 Trials shows both, side by side |
| 2D simulator | **300 / 300**, 0 collisions, median 7.5 s | Dashboard → *Run 2D benchmark* |
| Detector | **YOLO26l** on the NVIDIA GPU: **55.0 mAP50-95** on COCO (YOLO26x in MAXIMUM mode: 57.5; YOLO26n on a plain CPU: 40.9; old YOLOv8n: 37.3). These are Ultralytics' published figures | Window title and panel footer show the model, the GPU, FPS and inference ms |
| Market | AMR market **$18.9B by 2032, 21.8% CAGR** (Allied Market Research). MarketsandMarkets says $7.1B, so forecasts vary | Slide 5 |

> ⚠️ **Be precise about one thing:** the 3D simulator *emulates* the camera detector (it outputs the same thing YOLO outputs: class, box, confidence, with confidence dropping with distance, angle and occlusion). **Real YOLO26 runs in the Vision Brain on the webcam.** Say it this way and nobody can catch you out.

---

## 🎤 Slide-by-slide script

### Slide 1 — Title (15 s)
> *"Good morning, judges. I'm Harshit, and this is **SightSeeker**: you tell a robot what to find in plain English, and it looks around, recognises objects with AI vision, builds its own map, avoids obstacles and drives to the goal."*

### Slide 2 — Problem (20 s) · *invite a judge to click the dark arena*
> *"Most mobile robots need a pre-built map and hard-coded waypoints. The moment a pallet or a person moves, they're stuck. Here's our planner in miniature. Click anywhere to drop an obstacle, and watch the range rays turn red as it re-plans around it."*

### Slide 3 — Solution (20 s)
> *"Four layers. **Language:** 'find the red barrel' becomes a detector class. **Vision:** YOLO26, Ultralytics' newest detector. **Planning:** a free-space planner using VFH, the Vector Field Histogram method, plus a separate safety layer that always leaves room to stop. **Proof:** a 120-metre 3D warehouse where it completed 1,000 random missions with zero collisions."*

### Slide 4 — Architecture (20 s) · *click stages 2 → 3 → 4*
> *"Every frame runs Sense, Think, Act. Sense: camera plus a 360° LiDAR in the sim. Perceive: YOLO boxes and language grounding. Think: a state machine (Searching, Navigating, Avoiding, Goal Reached) and a semantic map of everything it has seen. Plan: pick the open direction closest to the goal. Act: left and right wheel speeds, which we can stream to an Arduino."*

### Slide 5 — Market (10 s)
> *"Autonomous mobile robots are projected at about $18.9 billion by 2032. The calculator is an illustrative model, and its assumptions are on the slide."*

### Slide 6 — Roadmap (5 s, optional)
> *"Next steps: game NPCs, signed mission logs for Web3, and real hardware. We already ship the Arduino firmware."*

### Slide 7 → Demo (5 s)
> *"Rather than more slides, let me show it working."*

---

## 🎬 Live demo (≈ 2 minutes)

### Step 1 · Open the 3D simulator: `launch.bat` → **2**
> *"A 120-metre warehouse with 28 objects. The robot has a forward camera (that's the blue wedge) and a 360° LiDAR (those rays)."*

### Step 2 · Type a goal: **`Find the red barrel`** → Enter
- A toast shows how the sentence was grounded: *"find the red barrel" → Red Barrel (matched: red, barrel)*.
> *"It doesn't know where the barrel is. Watch the **decision log** on the left. It does a 360° scan, then explores. And it searches smart: as soon as it maps a steel barrel, it knows red barrels are usually stored with them, so it looks there first. That's the same idea as VLFM from Boston Dynamics' AI Institute, at ICRA 2024."*
- Look for the log line *"💡 Steel Barrel on my map: red barrels are usually nearby → searching there first"*.

### Step 3 · Point at the right-hand panel while it drives
- **Robot Camera:** detector boxes such as `cargo box 0.84`. *"This is what the robot sees, in YOLO's output format."*
- **Robot's map:** white = seen, dots = objects it has mapped, orange × = where it's heading next.
- **Collisions: 0**, Explored %, Target belief: *not seen* → *12.3 m* once spotted.
- When it swerves: status turns **OBSTACLE AVOIDANCE**, rays go red, and the log says *"Obstacle ahead: Safety Pillar at 6.1 m → steering around."*

### Step 4 · Goal reached → ask for something it has already seen
- After **GOAL REACHED**, type **`go to the yellow pillar`**. Pillars, steel barrels and cargo boxes appear several times, so it has almost always mapped one already (check the log for *"Mapped Safety Pillar…"*).
- The log says *"already in my semantic map → navigating from memory"*, and it drives straight there.
> *"It remembers what it saw while exploring. That's the semantic map at work: no second search needed."*
- (If you ask for something it hasn't seen yet, it simply explores again, which is also fine to show.)

### Step 5 · Prove it's not a lucky run: click **📊 Trials** → **Run 50 trials**
> *"Fifty brand-new random warehouses, same code, headless, in a few seconds. Each one is run twice, with and without smart search, so you see the difference on the same maps. Success rate, median time, and collisions, all counted honestly."*

> 🎬 **Recording the video?** Press **D** in the 3D simulator: it plays steps 2–5 by itself with captions (red barrel → yellow pillar from memory → 50 trials). **Esc** stops it.

### Step 6 · Real vision: `launch.bat` → **4**, goal `bottle`, then press **Enter** twice (no open-vocabulary, no MAXIMUM mode)
- Hold a bottle **left** → **TURN LEFT** (left wheel slow, right fast). **Centre** → **FORWARD**. Bring it **close** → **GOAL REACHED / STOP**. It stays stopped even when the bottle fills the camera; move the bottle back to resume.
- Keep the bottle in view further back, then put a **cup inside the shaded path, close to the camera** → **AVOID CUP → LEFT/RIGHT**.
- Press **T**, type `find my phone` → Enter → it switches to *cell phone*. Or press **V** and *say* it (offline voice).
- Point out the **distance in cm** on the target label, the **top-down view** in the corner, and the **mission summary** card when it reaches the goal. Press **C** once beforehand with the bottle at 50 cm to calibrate the distance.
- Press **R** to record the window to `recordings/` (a backup clip for the video).
> *"This is real YOLO26 on my webcam, running on the laptop's GPU. The same decision loop turns real detections into steering and wheel commands. With `--serial COM5` they go straight to an Arduino motor driver."*

- **Optional, if asked about "any object":** quit with **Q**, run option **4** again and answer **y** to *open-vocabulary*. Type a goal COCO doesn't have, e.g. `find the blue marker`. YOLOE matches the words directly, no retraining.

> 💡 People and the desk are ignored as obstacles here ("presenter mode", **P** toggles it), because I'm in the frame and the webcam sees the table. On a robot, people are obstacles.

---

## 🛟 If something goes wrong

| Problem | Fix |
|---|---|
| 3D sim is slow | Press **V** to hide the robot camera, or **F** to speed up the sim |
| Robot is still searching after 30 s | Press **F** (×5), or **R** for a new warehouse. Meanwhile, narrate the decision log |
| Webcam won't open | Close other apps using the camera (Zoom/Meet), or run `python yolo_detection.py --camera 1` |
| Python packages missing | `launch.bat` → **6** (needs internet once) |
| Wi-Fi is down | Everything runs offline once option 6 has run: three.js and the YOLO26 weights are local |
| Vision Brain lags or the laptop gets hot | Plug in the charger and close Chrome tabs. Answer **N** to *MAXIMUM mode*. The video stays smooth either way; only the `AI x/s` number depends on the GPU/CPU |
| Want a backup video | `python yolo_detection.py --save backup_demo.mp4` records the annotated feed |

---

## 🛡️ Judges' questions — honest, strong answers

**Q: Is YOLO running inside the 3D simulator?**
> *"No. In the simulator the detector is emulated: it outputs the same class, box and confidence, with realistic drop-off for distance, angle and occlusion. That lets us test the decision logic over 1,000 runs. The real YOLO26 runs in the Vision Brain on the webcam, and both feed the same kind of decision loop."*

**Q: Why simulation instead of hardware?**
> *"It's standard sim-to-real practice. We can run thousands of randomized missions and measure success and collisions, which is impossible with one physical robot in a weekend. The hardware path exists: the Vision Brain outputs wheel commands, and the Arduino + L298N firmware is in `hardware/`. A basic 2WD build is roughly a few thousand rupees (Arduino, L298N, chassis, battery, webcam), with the laptop running YOLO."*

**Q: How do you know the distance with one camera?**
> *"From the object's size: the pinhole model says distance = focal length × real size ÷ size in pixels, and we know a bottle's long side is about 24 cm. It's about ±20–30% until you press C with the object held at 50 cm, which calibrates the camera. To decide 'arrived' we use how much of the frame the target fills, which doesn't depend on that estimate. For metric distance you use the pinhole model (distance = focal length × real size ÷ pixel size), or a depth camera like an OAK-D. In the simulator, the LiDAR provides range."*

**Q: How does it decide where to search?**
> *"Frontier exploration with common sense. It keeps a coverage grid of what the camera has already seen and drives toward the unexplored region with the most unseen area per metre travelled. On top of that it knows where things usually are: red barrels with steel barrels, the desk next to the workbench. Once it has mapped one of those neighbours, unexplored areas near it count four times as much. On the same 1,000 warehouses that's 15% faster than plain frontier search."*

**Q: Isn't the smart search only working because you built the warehouse that way?**
> *"Yes, and that's the honest point: it helps because warehouses really are organised like that. In a randomly arranged space the hint wouldn't help. The next step is learning those habits from data instead of a hand-written table, which is what VLFM does with a vision-language model."*

**Q: What if the target is hidden behind something?**
> *"The detector models occlusion: hidden objects aren't detected. The robot keeps exploring from new viewpoints until it sees the target. In 1,000 runs it always found it."*

**Q: How does obstacle avoidance work exactly?**
> *"From the 360° LiDAR scan we compute how far the robot can drive in each of 72 directions before it gets too close to something. That's a free-space histogram, the VFH method. We pick the open direction closest to the goal. A separate safety layer caps speed so the robot can always stop in time, and if the robot gets stuck it backs up and re-plans."*

**Q: What happens if two objects of the target class are visible?**
> *"The Vision Brain tracks the one it locked onto (box overlap between frames), so it doesn't flip-flop. In the simulator it goes to the nearest instance in its map."*

**Q: What about the camera being blocked, or vision failing?**
> *"Sensor fusion. The LiDAR safety layer doesn't depend on vision, so the robot never drives into something it can't see. If the target is lost, it heads for where it last saw it, then searches again."*

**Q: How did you improve detection accuracy?**
> *"Three things. We moved to YOLO26, Ultralytics' newest model family, and on the GPU we run the large YOLO26l (55.0 mAP, against 37.3 for the YOLOv8n we started with). A temporal filter only accepts an object seen in 2 of 3 consecutive frames, which kills one-frame false alarms. And we restrict to an indoor class set with a 0.35 confidence floor. For objects COCO doesn't know, we have a fine-tuning pipeline that trains on a free Colab GPU."*

**Q: Real-time performance?**
> *"Detection runs on a background thread, so the video stays at camera speed and the boxes update independently. Both numbers are on screen. On an NVIDIA GPU it runs YOLO26l in half precision; we cap it at 15 detections a second because the robot doesn't need more, which keeps the laptop cool. Without a GPU it falls back to YOLO26n with an OpenVINO export, built for edge devices."*

**Q: How does this connect to Web3 or gaming (Hack in Hills themes)?**
> *"Today it's a vision-robotics project. The 3D robot is already effectively an NPC that sees and plans for itself in a game-style engine. For Web3, the mission log and map it produces could be signed and anchored on-chain as proof of work (DePIN-style rewards). That's roadmap, not built yet."*

---

## 🎒 Checklist before the demo

- [ ] Laptop charged + charger packed (laptops throttle the GPU on battery).
- [ ] Run `launch.bat` → **6** once **with internet**. It installs packages, checks the NVIDIA GPU and downloads every model. The last screen should list each model as **OK**.
- [ ] Open the **3D sim** once in Chrome/Edge; check it's smooth (press **V** if not).
- [ ] Run the **Vision Brain** once; allow camera access; check the panel says `GPU: NVIDIA ...` and the FPS.
- [ ] Have a **bottle** and a **cup** on the table for the webcam demo.
- [ ] Turn off notifications; close Zoom/Meet (they lock the webcam).
- [ ] Optional: record a backup video with `--save backup_demo.mp4`.
- [ ] Breathe, smile, and let the decision log do the talking. 🚀
