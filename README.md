<<<<<<< HEAD
# 🎓 FaceAttend - AI-Based Attendance System Using Face Recognition

Automatic attendance from a webcam: register students, recognise them live, and
record attendance once per day with date and time. Runs 100 % locally - no paid
APIs, no cloud.

**Stack:** Python · OpenCV (webcam + Haar face detection) · FaceNet / InceptionResnetV1
(facenet-pytorch, CPU) · NumPy + scikit-learn (cosine similarity) · SQLite · Pandas · Streamlit

## Folder structure

```
ai_attendance/
├── app.py                  # Streamlit entry point + sidebar navigation
├── config.json             # Sample configuration (thresholds, camera, paths)
├── requirements.txt
├── run.bat                 # Windows shortcut to start the app
├── .streamlit/config.toml  # Dark theme
├── core/                   # Logic - no Streamlit code in here
│   ├── config.py           # Loads settings from config.json
│   ├── camera.py           # Webcam open/read/release + error handling
│   ├── face_detection.py   # OpenCV Haar cascade detector
│   ├── preprocessing.py    # Crop, eye-based alignment, resize, blur check, normalise
│   ├── embeddings.py       # FaceNet 512-d embeddings
│   ├── recognition.py      # Cosine-similarity matching, unknown detection
│   ├── database.py         # SQLite schema + all queries
│   ├── attendance.py       # Mark attendance, duplicate prevention, statistics
│   └── visualize.py        # Boxes/labels drawn on frames
├── ui/                     # Streamlit pages
│   ├── styles.py           # Theme CSS, cards, headers
│   ├── resources.py        # Cached detector + FaceNet model
│   ├── dashboard.py        # Totals, present/absent, attendance %, charts
│   ├── registration.py     # Register (webcam/upload) + manage students
│   ├── live_attendance.py  # Real-time recognition page
│   ├── records.py          # Filterable records + CSV
│   └── reports.py          # Daily sheet, student summary, full export
├── tests/test_core.py      # Tests that need no webcam
└── data/                   # Created automatically
    ├── attendance.db       # SQLite database
    ├── faces/<student_id>/ # Saved face samples
    └── exports/            # Saved CSV reports
```

## Installation (Windows + VS Code)

1. Install **Python 3.10 or 3.11 (64-bit)** from python.org (tick "Add Python to PATH").
   Python 3.13 is not supported by the pinned PyTorch version.
2. Open the project folder in VS Code, then open a terminal (**Terminal → New Terminal**):

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

macOS/Linux: `source venv/bin/activate` instead of `venv\Scripts\activate`.

## Run

```powershell
streamlit run app.py
```

(or double-click `run.bat`). The app opens at http://localhost:8501.

> **First run only:** FaceNet's pretrained weights (~107 MB) download once from GitHub and are
> cached. After that everything works offline.

## Step-by-step test

1. **Register** - *Student Registration → Register student*. Enter ID `S1001`, a name and class
   `10-A`, keep *Webcam (automatic)*, click **Register student**. Face the camera and slowly turn
   your head slightly until the sample counter fills. (No webcam? use *Upload photos* with 5+ photos.)
2. Register a second person (or a friend) the same way to test unknown/other faces.
3. **Live recognition** - *Live Attendance → Start camera*. Your box turns green with your
   name; after a few consecutive recognitions you get "✅ marked present".
4. **Duplicate check** - leave and re-enter the frame, or restart the camera. You are *not* marked
   twice; the message says "already marked today".
5. **Unknown face** - show someone unregistered (or a photo of one). They get a red *Unknown* box
   and are never marked.
6. **Multiple faces** - several people in view are each recognised (up to 5). During
   registration only one face is allowed.
7. **Records** - *Attendance Records*: filter by student, class and period, download CSV.
8. **Dashboard / Reports** - check totals and percentage, download the daily sheet.
9. **Camera errors** - start the camera while another app is using it, or enter a wrong camera
   index; you should see a clear error instead of a crash.

Run the unit tests (no webcam needed): `python -m unittest discover tests -v`

## How it works

```
webcam frame → Haar face detection → crop + eye alignment + resize 160×160
            → FaceNet (512-d vector, L2-normalised) → cosine similarity vs. stored vectors
            → accepted if score ≥ threshold AND beats runner-up by ≥ margin
            → must be seen `confirm_frames` times in a row → INSERT (student, date) in SQLite
```

- **No duplicates:** the `attendance` table has `UNIQUE(student_id, date)`; the database enforces
  one record per student per day.
- **Speed:** detection runs on a downscaled frame and the CNN runs on every 3rd frame.

## Configuration (`config.json`)

| Key | Meaning |
|---|---|
| `similarity_threshold` | Cosine similarity needed to accept a face (default 0.65). Raise if wrong people are accepted; lower if registered students show *Unknown*. Also adjustable live on the Live Attendance page. |
| `min_margin` | Best match must beat the second-best student by this much (0.03). |
| `confirm_frames` | Consecutive recognitions needed before marking (3). |
| `process_every_n_frames` | Run FaceNet every n-th frame (3). Increase on slow laptops. |
| `camera_index` | 0 = default webcam; try 1, 2 for external cameras. |
| `registration_samples`, `min_samples`, `blur_threshold` | Registration capture quality. |

## Troubleshooting

- **Camera won't open:** close Zoom/Teams/browser tabs using it; Windows *Settings → Privacy →
  Camera → allow desktop apps*; try another camera index.
- **Registered students show Unknown:** re-register in good, even lighting with 15+ samples, face the
  camera, or lower the threshold slightly.
- **Wrong person accepted:** raise the threshold (0.70-0.75).
- **FaceNet install fails:** confirm Python is 3.10/3.11 64-bit and pip is up to date.
- **Weights download fails:** the first run needs internet; check your firewall/proxy.

## Limits and good practice

Haar detection works best on frontal faces in decent light. For real classroom use, obtain
consent from students, store data securely, and treat face recognition as an aid, not as
sole proof of presence.
=======
# Face-Recognition-Attendance-System
An AI-powered attendance management system using Python, OpenCV, FaceNet, CNN embeddings, and cosine similarity for real-time face recognition and automated attendance tracking.
>>>>>>> e65e58abd3b86bea87b744947f6a97f7388e7b57
