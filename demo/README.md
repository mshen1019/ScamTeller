# Demo

- `ScamTeller_demo_2min.mp4`: the 2-minute demo video (problem, method, two replayed calls,
  benchmark, threshold protocol, limitations). Narration is synthesized (macOS `say`).
- `render_demo.py`: renders that video from the stored scores in `datasets/bench/`; no model is
  run. `pip install -e ".[demo]"`, then `python demo/render_demo.py --voice "Ava (Premium)"`.
- `replay_demo.py`: the same calls replayed utterance by utterance in a terminal, for live demos.
  Fully offline: `python demo/replay_demo.py --pace 1.2`.

The video credits the data factory as `scam-call-factory`, the name `callfactory` was developed
under before this repository.
