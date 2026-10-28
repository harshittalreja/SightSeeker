# SightSeeker: Vision-Guided Autonomous Navigator

**Hack in Hills '26 · Track 01: Vision-Based Autonomous Robot**

Tell the robot what to find in plain English (*"find the red barrel"*). It understands the goal, looks around with AI vision, builds its own map, avoids obstacles and drives to the object.

| Problem-statement requirement | Where it lives |
|---|---|
| Object detection / image classification | `yolo_detection.py`: YOLO26 on the webcam (80 COCO classes), YOLOE for any described object |
| Scene understanding / VLM-style goals | Language grounding: "find my phone" → `cell phone`; semantic map of everything seen (3D sim) |
| Sensors & perception | Robot camera + 360° LiDAR (3D sim), camera cone + 32-ray proximity ring (2D sim) |
| Autonomous navigation | Finite-state machine + smart frontier exploration (searches where the target is likely first) + VFH free-space planner |
| Obstacle detection & avoidance | LiDAR free-space planner + independent safety layer + stuck recovery |
| Real-time decision making | Vision Brain turns live detections into TURN LEFT / FORWARD / AVOID / STOP |
| Motor control | Differential-drive wheel commands (−255…255); `--serial` streams them to an Arduino + L298N |

## Measured results

| | Result |
|---|---|
| 3D simulator, 1,000 randomized warehouses (120 m) | **100% goal reached, 0 collisions**, median 51 s robot-time |
| Smart search vs plain frontier search, same 1,000 warehouses | median **51 s vs 60 s** (15% faster), mean 55 s vs 60 s, 10% shorter paths; both 100% / 0 collisions |
| 2D simulator, 300 randomized rooms | **100% goal reached, 0 collisions**, median 7.5 s |
| Detector | YOLO26l (55.0 mAP50-95 on COCO) on an NVIDIA GPU, YOLO26n (40.9, ~39 ms per frame on a CPU) on a laptop without one; YOLO26x (57.5) for maximum accuracy. YOLOv8n (37.3) is the offline fallback. Figures are Ultralytics' published numbers |

You can re-run these yourself: 3D sim → **📊 Trials** (runs each new warehouse with and without smart search), or `python robot_simulation.py --benchmark 300`.

## Run it (Windows)

Cloned from GitHub? The model files aren't in the repository (they're large), so run `launch.bat` → **6** once with internet first. After that everything works offline.

Double-click **`launch.bat`**:

1. Pitch: the interactive presentation, `SightSeeker_Pitch.html` (`→`/`Space`/clicker next, `F` fullscreen, `H` speaker notes, `M` sound)
2. **3D Simulator**, the main demo (opens in Chrome/Edge, works offline)
3. 2D Simulator (Pygame)
4. **Vision Brain**: real YOLO26 on the webcam. It asks for a goal, then *open-vocabulary mode?* and *MAXIMUM mode?* (Enter = no for both)
5. Mission Control dashboard
6. **Install / check** (needs internet once): Python packages, the NVIDIA GPU build of PyTorch if you have an NVIDIA card (~3 GB, asks first), and every model the demo uses, so it then runs offline

Or from a terminal:

```bash
pip install -r requirements.txt
python yolo_detection.py --goal "find my phone"          # webcam
python yolo_detection.py --max                           # MAXIMUM mode: best accuracy, uses the whole laptop
python yolo_detection.py --open-vocab --goal "red toolbox" # any object you can describe (YOLOE)
python yolo_detection.py --video clip.mp4 --save out.mp4  # file in, annotated video out
python yolo_detection.py --serial COM5                    # also drive an Arduino rover
python check_gpu.py                                       # is the NVIDIA GPU being used? how to fix it
python robot_simulation.py                                # 2D sim
python robot_simulation.py --benchmark 300                # headless trials
python dashboard.py
```

### Controls
- **3D sim:** type a goal and press Enter · `Space` pause · `C` camera · `V` robot camera on/off · `R` new warehouse · `F` sim speed · `B` benchmark · `K` smart search on/off · `D` guided demo for recording · `/` jump to the prompt
- **Vision Brain:** `T` type a goal · `V` say a goal (offline voice) · `W` walk mode (spoken directions) · `D` digital twin · `G` next demo goal · `C` calibrate distance · `R` start/stop recording · `N` navigation overlay + top-down view · `A` all 80 classes / indoor set · `P` presenter mode (people and the desk aren't obstacles) · `Space` pause · `S` screenshot · `Q` quit
- **2D sim:** type a goal at the start · `R` new goal · `Space` pause · `Q` quit

## How it works

```
camera frame ──► YOLO26 detections ──┐
text goal ─────► language grounding ─┼─► state machine ─► free-space planner ─► wheel commands (L/R)
LiDAR / range ─► free-space histogram ┘   (+ semantic map,     + safety layer        └─► Arduino (optional)
                                            decision log)
```

- **State machine:** `SEARCHING` (360° scan + frontier exploration) → `NAVIGATING` (target seen or remembered) → `AVOIDING` (path blocked) → `GOAL_REACHED`.
- **Exploration:** a coverage grid tracks what the camera has seen; the next waypoint is the unexplored area with the most unseen cells per metre of travel.
- **Smart search:** the robot knows where things usually are in a warehouse (red barrels are stored with steel barrels, the blue crate sits with the cargo boxes, the desk is in the workshop with the table). Once it has mapped one of those neighbours, frontiers near it score 4× higher, and the decision log says why: *"Steel Barrel on my map: red barrels are usually nearby → searching there first"*. The same idea as VLFM (below), with a hand-written table instead of a vision-language model.
- **Planner:** for each of 72 directions, how far the robot's safety bubble can travel (from the LiDAR scan). It picks the open direction closest to the goal (VFH, the Vector Field Histogram method).
- **Safety layer:** caps speed so the robot can always stop, predicts the next 0.4 s of motion, and never turns into contact.

## Vision Brain extras

- **Distance in cm:** the pinhole camera model, distance = focal length × real size ÷ size in pixels, with typical sizes of everyday objects (a bottle's long side is ~24 cm). It's an estimate (±20–30%) until you press **C** once with the target held 50 cm from the camera; that calibration is saved in `camera_calibration.json`. Objects cut off by the picture edge show "very close" instead of a wrong number.
- **Top-down view:** a small map in the corner places the target (green) and obstacles (red) by bearing and estimated distance, inside the camera's field of view.
- **Mission summary:** on GOAL REACHED a card shows the time taken, course corrections, obstacles avoided and the stopping distance.
- **Voice goals:** press **V** and say "find my phone". Offline speech recognition (Vosk small English model, installed by option 6); in normal mode it only listens for the words it can act on, which makes it far more accurate.
- **Walk mode (no robot needed):** press **W** (or answer *y* to *Walk mode* in `launch.bat` option 4). A person carries the camera and SightSeeker **speaks** its decisions: "Turn a little left", "Careful, chair ahead. Step to the right", "Stop. The bottle is right in front of you." It uses the speech voice built into Windows, offline, nothing to install. The person becomes the robot's wheels, so the whole see → decide → move loop runs in a real room. Also a real use case: guiding a visually impaired person to an object.
- **Phone as the camera:** install the free **IP Webcam** app, tap *Start server*, then give its address as the camera: `python yolo_detection.py --camera 192.168.1.5:8080` (or type it at the *Camera* question in `launch.bat` option 4). Phone and laptop must be on the same Wi-Fi or hotspot. Frames are read on a background thread, so the picture never lags behind.
- **Digital twin:** press **D** (or `--twin`). A third panel shows a virtual 2WD rover driven by exactly the wheel commands the Arduino receives (`L55,R198`): where those commands take it in the next seconds (yellow), its spinning wheels, and the path it would have driven so far.
- **Recording:** **R** starts/stops recording the annotated window to `recordings/`; `--record` records from the start.

## Hardware: rover design and simulation

The rover's firmware, wiring diagram, component specs, a **Wokwi simulation** you can run in the browser, a **3D rover twin** (`rover_3d.html`), its limitations and the physical build plan are in **[hardware/README.md](hardware/README.md)**.

![Wiring diagram](hardware/docs/wiring_diagram.png)

## Speed, GPU and heat

- **Smooth video:** on a webcam the detector runs on a background thread. The video plays at the camera's frame rate and the boxes update as fast as the model can go. The panel shows both, e.g. `video 30 FPS | AI 15 FPS`.
- **NVIDIA GPU:** used automatically when PyTorch can see it. Then it runs YOLO26l at 640 px in FP16 with the webcam in HD (1280×720). Plain `pip install` on Windows gives the CPU-only PyTorch, so `launch.bat` → 6 checks for an NVIDIA card and offers to install the GPU build (CUDA 12.8, which RTX 50-series cards need). `python check_gpu.py` shows what's going on.
- **No NVIDIA GPU:** YOLO26n at 480 px with the webcam at 640×480. `launch.bat` → 6 also builds an **OpenVINO** copy (`yolo26n_openvino_model/`), up to ~3x faster on Intel CPUs, which is picked automatically. Intel/AMD graphics can't run PyTorch.
- **Heat:** detection is capped at 15 per second and the video at 30 FPS, and PyTorch/OpenCV use 2 CPU threads, so a laptop doesn't run flat out for no benefit. Lower it further with `--ai-fps 10`; `--ai-fps 0` removes the cap.
- The window title and the panel's `model:` line show which model and device are really in use.
- **If the GPU fails mid-demo** (driver hiccup), detection switches to the CPU automatically instead of silently finding nothing, and with `--serial` the rover is sent STOP whenever detections stop arriving.
- Plug the laptop in. On battery, laptops throttle the CPU and GPU.

## Detection accuracy

- **Newest model family:** YOLO26. On a CPU, YOLO26n beats YOLOv8n (40.9 vs 37.3 mAP) at about twice the speed. On an NVIDIA GPU the Vision Brain uses the large YOLO26l (55.0 mAP).
- **MAXIMUM mode:** answer `y` to *MAXIMUM mode* in `launch.bat` option 4 (= `--max`). On an NVIDIA GPU it runs YOLO26x (57.5 mAP) on 960 px images with the webcam in Full HD, removes the 15-detections and 30-FPS caps, and lets PyTorch use every CPU core. Without a GPU it uses YOLO26l at 640 px (YOLO26x would take seconds per frame). More pixels per object helps most with small or distant objects. The laptop runs hot, so keep it plugged in and use this mode when accuracy matters. Any size can be picked with `--model n/s/m/l/x`.
- **Open vocabulary:** `--open-vocab` (or answer `y` in `launch.bat` option 4) uses YOLOE, so the goal can be any described object ("red toolbox"), not just the 80 COCO classes. It's slower on a CPU.
- **Flicker filter:** an object is shown only once it appears in 2 of the last 3 frames, so one-frame false alarms disappear.
- **Indoor class set + confidence ≥ 0.35** by default. This removes odd guesses like "toilet". Press `A` for all 80 classes.
- **Custom objects:** `train_custom.py` / `train_on_colab.ipynb` fine-tune on your own labelled photos. Re-training the COCO classes on a small free dataset won't beat the pretrained model.

## Related work

| Project | What it does | How SightSeeker compares |
|---|---|---|
| [VLFM](https://github.com/bdaiinstitute/vlfm) (Boston Dynamics AI Institute, ICRA 2024) | Finds an object category in an unknown building: frontier exploration, with a vision-language model scoring which frontier is most promising. Runs on a Spot robot with a depth camera | Same loop as our 3D simulator: explore frontiers, prefer likely ones, drive to the target. We use a hand-written "where things usually are" table instead of a vision-language model, sized for a ₹3,000 rover |
| [OK-Robot](https://github.com/ok-robot/ok-robot) (NYU) | Open-vocabulary "pick up X, put it on Y" in real homes; 58.5% success over 10 homes | Open-vocabulary goals like our YOLOE mode, on a Stretch robot with an arm and an iPhone LiDAR scan |
| [VAFOR](https://github.com/bekatan/vafor) (IEEE RO-MAN 2023) | Voice assistant that fetches objects with YOLOv3 + GPT-3 on a TurtleBot3 | Same language-to-object idea; ours runs offline and adds exploration, mapping and collision benchmarks |

## Honest notes

- In the **3D simulator, the camera detector is emulated**. It outputs exactly what YOLO outputs (class, box, confidence), and confidence drops with distance, viewing angle and occlusion. **Real YOLO26 runs in the Vision Brain** (`yolo_detection.py`).
- Distance in the Vision Brain is an **estimate from object size** (pinhole model, about ±20–30%, better after pressing C to calibrate), not a depth sensor. The robot still decides "arrived" from how much of the picture the target fills.
- Smart search only helps because the simulated warehouses follow the same habits it assumes (barrels stored together, and so on), as real warehouses usually do. In a randomly arranged space it behaves like plain frontier search.
- `hardware/sightseeker_rover/sightseeker_rover.ino` is written and was tested on a PC against a simulated Arduino (command parsing, 500 ms watchdog, ultrasonic stop), but **not yet run on a physical robot**. Open it in the Arduino IDE and press Verify before uploading.
- Language grounding is keyword, synonym and typo matching. It's designed so a VLM such as CLIP can replace it.

## Files

| File | What it is |
|---|---|
| `rover_3d.html` | **3D rover twin**: the actual 2WD rover in 3D, running the firmware's rules (watchdog, 25 cm stop) and finding a bottle; exploded parts view (E). Needs `three.min.js` + `OrbitControls.js` beside it |
| `robot_simulation_3d.html` | 3D simulator (Three.js r128, offline): needs `three.min.js` + `OrbitControls.js` beside it |
| `yolo_detection.py` | Vision Brain: YOLO26 → decisions → wheel commands (+ optional serial) |
| `launch.bat` | Windows menu for everything above |
| `check_gpu.py` | Checks whether PyTorch can use the NVIDIA GPU, and installs the GPU build if asked |
| `download_models.py` | Downloads every model the demo needs (run by `launch.bat` option 6) |
| `robot_simulation.py` | 2D simulator + headless benchmark |
| `dashboard.py` | Mission Control (launch buttons, measured results, architecture) |
| `SightSeeker_Pitch.html` | The pitch: an interactive presentation seen through the robot's camera (live race, distance demo, embedded 3D simulator, mission summary). Works offline |
| `presentation_mascot.html` | Older pitch deck with Sparky the mascot |
| `PITCH_SCRIPT_AND_CHEAT_SHEET.md` | What to say, demo steps, judge Q&A |
| `hardware/sightseeker_rover/` | Arduino + L298N firmware |
| `train_custom.py`, `train_on_colab.ipynb` | Fine-tune YOLO on your own labelled objects (use Colab's free GPU) |
| `DEMO_VIDEO_SCRIPT.md` | Shot list and voice-over for the demo video |
| `*.pt`, `mobileclip2_b.ts` | Model weights (YOLO26 sizes, YOLOE and its text encoder) downloaded by option 6. `yolov8n.pt` is the offline fallback |
| `LICENSE`, `.gitignore` | MIT licence; keeps model weights, recordings and calibration out of git |
