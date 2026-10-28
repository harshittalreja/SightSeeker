# SightSeeker: demo video script (about 2 min 45 s)

**Before recording**
- Charger plugged in, notifications off, Zoom/Meet closed (they lock the webcam).
- `launch.bat` → **6** done once with internet (all models show **OK**).
- In the Vision Brain, press **C** once with a bottle held 50 cm from the webcam (calibrates distance).
- Record the screen with **Win + Alt + R** (Xbox Game Bar) or OBS, 1080p. Speak slowly; it's fine to record the voice-over separately and lay it on top.
- Upload to YouTube as **Unlisted** and paste the link in the portal.

| Time | On screen | Say |
|---|---|---|
| 0:00 – 0:15 | Title slide (`launch.bat` → 1) | "Robots today need a map and fixed routes. With SightSeeker you just say what to find, like *find the red barrel*, and the robot searches, avoids obstacles and stops at the target." |
| 0:15 – 0:25 | 3D simulator opens (`launch.bat` → 2), press **D** | "This is our 120-metre test warehouse. The robot has a camera and a 360-degree LiDAR, and it doesn't know where anything is." |
| 0:25 – 0:45 | Demo types *find the red barrel*; 360° scan; decision log | "The sentence becomes a detector class. It scans, then explores. Watch the decision log: it explains every choice it makes." |
| 0:45 – 1:05 | Caption *Smart search: steel barrels spotted* | "And it searches smart. It just mapped a steel barrel, and red barrels are usually stored with them, so it looks there first. That's the same idea as VLFM from Boston Dynamics' AI Institute, at ICRA 2024, sized for a three-thousand-rupee robot." |
| 1:05 – 1:20 | Target spotted, drives around obstacles, GOAL REACHED | "Target spotted. The planner steers around obstacles and a separate safety layer always leaves room to stop. Goal reached, zero collisions." |
| 1:20 – 1:35 | *go to the yellow pillar*, drives straight there | "It remembers everything it saw. Ask for a pillar it passed earlier and it drives there straight from memory." |
| 1:35 – 1:55 | 📊 Trials: 50 warehouses, comparison box | "Not a lucky run: fifty brand-new warehouses, each run with and without smart search. Across a thousand warehouses it's a hundred percent success, zero collisions, and smart search is fifteen percent faster." |
| 1:55 – 2:30 | Vision Brain (`launch.bat` → 4, goal *bottle*, Enter, Enter). Bottle left → TURN LEFT, centre → FORWARD, close → GOAL REACHED card. Cup in the shaded path → AVOID | "This is real AI on my laptop's camera: YOLO26 on the GPU. Bottle on the left, turn left. Centre, forward. It estimates the distance in centimetres and shows a top-down view. Close enough: stop, with a mission summary. A cup in the path: it steers around it." |
| 2:30 – 2:40 | Press **V**, say *find my phone* | "Goals can be spoken too, fully offline." |
| 2:40 – 2:45 | Final slide or README results table | "The wheel commands already stream to our Arduino firmware. Next, we put it on a real rover. Thank you." |

**Numbers to keep consistent with the slides:** 1,000/1,000 simulated missions · 0 collisions · median 51 s (smart search) vs 60 s (plain) · YOLO26l 55.0 mAP on the GPU.
