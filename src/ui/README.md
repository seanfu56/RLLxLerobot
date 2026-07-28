# Single-camera Streamlit teleoperation

This application reproduces the ROBOTIS leader to Piper follower path from
[3_teleoperate_single_cam.sh](../scripts/3_teleoperate_single_cam.sh) while
keeping Streamlit outside the timing-sensitive control path.

## Components

- **teleop_runtime.py** starts a dedicated spawned hardware process. That child
  discovers the installed LeRobot plugins, constructs devices through the
  LeRobot factories, applies both default action processors, and owns the
  control, camera, watchdog, and recording workers. Streamlit never imports or
  calls the live hardware objects.
- **app.py** is the supervisory Streamlit page. It submits lifecycle and
  recording commands through a process-safe proxy and reads cached immutable
  runtime snapshots.
- The Piper follower owns the `overhead` LeRobot OpenCV camera exactly as it
  does in `3_teleoperate_single_cam.sh`. JPEG publication, video writing, and
  action sampling consume its latest cached frames on separate workers.
- The browser refreshes the latest preview JPEG directly. Preview frames do not
  trigger Streamlit reruns or hardware-process snapshot requests.
- Every control iteration follows the CLI order:
  `robot.get_observation() → teleop.get_action() → processors →
  robot.send_action()`. Recording never changes this configured control rate.
- **tests/test_teleop_runtime.py** covers lifecycle ownership, processor
  routing, recording-rate isolation, deadline misses, failure state, and
  emergency stop using mock devices.

The runtime-process proxy is shared by every browser session in the Streamlit
server. Opening another tab therefore does not open a second CAN, serial, or
camera connection, and UI rendering cannot take GIL time from robot control.

## Environment

Use the Python environment that already runs the repository's LeRobot command,
then install the UI dependency if needed:

~~~bash
python -m pip install -r ui/requirements.txt
python -m pip install -e ./lerobot
python -m pip install -e ./plugins/lerobot-robot-piper
python -m pip install -e ./plugins/lerobot-teleoperator-robotis
~~~

Saving MP4 video also requires **ffmpeg** on the Ubuntu PC. The two plugin
imports must resolve to the intended editable copies; verify them before
hardware use:

~~~bash
python -c "import inspect, lerobot_robot_piper; print(inspect.getfile(lerobot_robot_piper))"
python -c "import inspect, lerobot_teleoperator_robotis; print(inspect.getfile(lerobot_teleoperator_robotis))"
~~~

## Start the server

Run from the repository root on the Ubuntu PC connected to the hardware:

~~~bash
streamlit run ui/app.py \
  --server.address=127.0.0.1 \
  --server.port=8501 \
  --server.enableStaticServing=true \
  --server.headless=true
~~~

Forward remote port 8501 in VS Code Remote-SSH, or use:

~~~bash
ssh -L 8501:127.0.0.1:8501 user@ubuntu-pc
~~~

Then open **http://127.0.0.1:8501** on the laptop. CAN, serial, camera,
recording, and control-loop work all stay on the Ubuntu PC.

The recording sidebar has a five-second start countdown by default. The
countdown is non-blocking and runs only in the supervisory UI; the dedicated
hardware process continues teleoperation normally, and the episode clock begins
after the countdown finishes.

The connection sidebar defaults to the installed **100 mm large jaws**. Select
70 mm only when controlling the small-jaw gripper. The chosen travel is applied
to both the ROBOTIS leader mapping and Piper command clamp.

## UI responsiveness

The page is deliberately cheap to refresh, because every rerun competes with the
operator's own browser for a link that is usually an SSH tunnel:

- The recording panel refreshes at 5 Hz because it drives the start countdown.
  The telemetry panel refreshes at 1 Hz — its two tables are re-serialized on
  every pass and were the dominant cost at the old shared 5 Hz rate.
- The next-episode index is cached rather than rescanning the output directory
  on every refresh. Saving or discarding an episode invalidates it immediately,
  and a 5 s TTL covers writes from outside the UI.
- The preview `<img>` re-fetches only after the previous frame has settled. A
  fixed interval queues requests faster than a slow tunnel can serve them, which
  is what makes the tab appear to freeze.
- **Preview publish rate** defaults to 10 Hz. Each frame is a JPEG pulled across
  the tunnel; raise it only on a fast link. It does not affect recorded video.

If the page still feels slow, lower the preview rate first, then untick **Publish
browser preview** — recording continues normally without it.

## Recording output

Each saved episode is committed atomically from a hidden pending directory:

~~~text
episode_000/
├── video.mp4
├── actions.csv
├── observations.csv
├── frame_timestamps.csv
└── meta.json
~~~

Recording is action-driven. At every action sampling deadline the recorder
takes the latest successfully sent control sample and the exact cached camera
frame attached to that control iteration, then enqueues them as one pair.
Therefore action row `N` always corresponds to video frame `N`.

**actions.csv** contains `sample_index`, matching `frame_index`, the source
control sequence and timestamp, the camera capture sequence and timestamp,
`action_frame_delta_ms`, six joint commands, and the gripper command.
**observations.csv** contains the same sample/frame and control-sequence
identity, the timestamp of the hardware read, the action delay, and the six
measured Piper joint positions plus measured gripper position. The observation
is captured before its paired action is sent; it is not reconstructed from a
command.
**frame_timestamps.csv** records the same frame index, capture sequence, and
monotonic timestamp for every written video frame. **meta.json** records target
rates, action/observation counts, deadline misses, dropped frames, task text,
configured gripper travel (`gripper_max_mm`), the observation source, the
common monotonic clock definition, and the one-action-and-observation-per-frame
alignment contract.

If MP4 conversion fails, the pending episode and AVI are retained so Save can
be retried. Discard explicitly removes the pending episode.

## Post-process collected episodes

Start the separate post-processing page on another local port:

~~~bash
streamlit run ui/postprocess_app.py \
  --server.address=127.0.0.1 \
  --server.port=8502 \
  --server.enableStaticServing=true \
  --server.headless=true
~~~

Forward port 8502 and open **http://127.0.0.1:8502**. Select a collected
episode and inspect it in the source-frame monitor at the top of the page. Drag
the browser-side playhead to display any source frame without rerunning or
clearing the Streamlit page; its readout has millisecond resolution and retains
the current position across editor reruns. Set the cut range with its
floating-point handles, precise second inputs, or one-frame nudge buttons.
Click **Preview selected cut** to render and loop the exact retained frames in a
separate player below the range editor. **Save previewed cut** is enabled only
while the current In and Out points still match that preview, so changing either
point requires a new preview before saving. Outputs default to
`outputs/teleop_processed`; source episodes are never modified or overwritten.

For aligned episodes, the crop keeps action row `N`, measured observation row
`N`, video frame `N`, and frame timestamp row `N` together, then rebases all
timestamps to the first retained frame while preserving original indices in
`source_sample_index` and `source_frame_index`. Older unaligned episodes are
cropped independently by their available timestamps and remain explicitly
marked as legacy.

## Run a trained policy

`policy_app.py` is the inference counterpart of `app.py`: the same supervisory
split, but joint targets come from a `src/policy` checkpoint instead of the
ROBOTIS leader. Start it from the repository root:

~~~bash
bash scripts/9_policy_ui.sh          # PORT=8503 by default
~~~

Forward port 8503 and open **http://127.0.0.1:8503**.

- **policy_runtime.py** owns the hardware process. It reuses this package's
  camera system, preview publisher, video writer and episode format, and adds
  the closed loop: `robot.get_observation() → PolicyRunner.select_action() →
  robot.send_action()`.
- **src/policy/checkpoints.py** lists runs and their `.pt` files from
  `run.json` and `log.jsonl` alone, so the browser process never imports torch.
- **src/policy/inference.py** owns the checkpoint: preprocessing identical to
  `data_prep.py`, EMA weights, chunk queueing, the goal frame for
  goal-conditioned checkpoints, and one warm-up inference.

### Choosing a checkpoint

The sidebar lists every run directory under **Model root** (`models/sweeps` by
default) with its policy, objective, bundle and best validation loss, then the
checkpoints inside it — `best`, `final` and each `step_*` snapshot. **Rescan
checkpoints** picks up runs that finished while the page was open.

Loading is a separate command from connecting, because building a CUDA context
and running the warm-up inference takes far longer than opening the CAN bus.
Load a checkpoint with no robot attached, compare a second one, then connect.
The checkpoint can be replaced while connected and idle, but never while the
arm is being driven.

A goal-conditioned (`gc`) run also needs a **goal source**: a recorded episode
directory (its `video.mp4`), a video file, or a still image. Frame `-1` is the
pose the demonstration ended in.

**Actions executed per prediction** is LeRobot's `n_action_steps`. These
checkpoints predict relative to the measured state, so a shorter horizon
re-anchors more often and tracks better near contact; the trade is one sampler
pass more often. The telemetry panel warns when a prediction at p95 no longer
fits inside one control step.

### Rollouts

The arm is commanded **only** between Start and Stop. While idle the control
worker still reads observations, so the preview, telemetry and watchdog stay
live but nothing is sent. A rollout stops on its duration, on the Stop button,
on any worker failure, and on the watchdog; each of those clears the driving
flag before anything else.

When a rollout ends the arm is parked back at the **start pose**, using the
plugin's own smoothstep trajectory (100 Hz, 30 deg/s, measured endpoint
verified) — the same one that homes it on connect. A policy otherwise leaves
the arm wherever the episode ended, which is not a pose the next trial can
start from, and the per-step clamp would begin from an arbitrary position. The
move runs on the control thread, so the policy and the parking trajectory can
never command the arm at once, and the staleness watchdog is suspended for its
duration. The preview freezes while it runs.

The start pose includes the gripper, so **parking releases anything the policy
was holding**. Untick **Return to the start pose when the rollout ends** to
leave the arm where it stopped — for inspecting a grasp, for instance — and use
**↩ Return to start pose** to park it manually afterwards. A parking move that
does not reach the pose is reported without tearing down control, so it can be
retried; Safe disconnect rests at the same pose either way.

A recorded rollout is written in exactly the layout the teleoperation page
produces (`video.mp4`, `actions.csv`, `observations.csv`,
`frame_timestamps.csv`, `meta.json`), so it can be replayed in the
post-processing page or fed back through `data_prep.py`. One control step is
one action row and one video frame, `actions.csv` also carries
`inference_latency_ms`, and `meta.json` gains a `policy` block naming the
checkpoint, step, sampler steps and goal source. Rollouts default to
`outputs/policy_eval`; untick **Record the rollout** to drive the arm without
writing anything.

Connect homes the arm to the configured **start pose** and disconnect rests it
there, matching `scripts/9_eval_policy_lab.sh`. Evaluation is only comparable
from the pose the demonstrations started in.

Run the mock suite without hardware or torch:

~~~bash
PYTHONPATH=src python -m unittest ui.tests.test_policy_runtime -v
PYTHONPATH=src python -m unittest policy.tests.test_checkpoints -v
~~~

## Safety and failure policy

- Safe disconnect first stops the control worker. Only after it has stopped
  does the Piper plugin run its rest trajectory and disable the arm.
- A control, camera, recorder, or watchdog failure moves the runtime to ERROR
  and prevents further robot commands. Use Safe disconnect if communication is
  healthy enough for a rest move.
- The first command has a separate three-second startup watchdog grace. After
  the first successful command, the default two-second hard stale-control
  timeout applies.
  Telemetry identifies whether a stall occurred in robot observation, leader
  reading, action processing, or robot command transmission.
- The ROBOTIS gripper spring is disabled by default in the Streamlit UI because
  it adds a serial velocity read and current write to every leader sample. It
  can be enabled after the sustainable control rate has been measured.
- The software emergency-stop button stops scheduling and directly requests
  Piper disable without a rest move. It cannot replace the physical emergency
  stop and cannot cover a blocked driver, operating-system failure, or power
  failure.
- Closing the browser or losing SSH does not stop the control worker. A timed
  episode stops in the runtime even if no browser is connected.
- Normal process exit attempts Safe disconnect through an exit handler. A hard
  kill cannot run that handler.
- Keep Streamlit bound to localhost and access it only through SSH forwarding;
  the page has no built-in authentication.
- The teleoperation page and the policy page each open their own CAN, camera
  and hardware process. Never point both at the same arm at once.
- Under policy inference the per-step clamp (`max_relative_target`, 2 deg/mm by
  default) is the main protection against a bad prediction, and the speed rate
  defaults to 30%. Raise either only after a checkpoint has behaved.

Run the mock suite without hardware:

~~~bash
python -m unittest discover -s ui/tests -v
~~~

When the UI is not running but an arm still needs to be parked and disabled,
use the standalone safe-close command from the repository root:

~~~bash
python scripts/close_robot_arm.py --can-port piper_left
~~~

The command reconnects using Piper's startup hold, runs the same configured
rest trajectory used by UI Safe disconnect, and then disables the arm and
gripper. It refuses to run when it detects `streamlit ... ui/app.py`, because a
second process must never issue CAN commands while the UI control worker owns
the arm.

Hardware acceptance still needs the staged timing and safety checks described
in [the implementation plan](../doc/10-streamlit-teleoperation-plan.md); mock
tests do not validate the sustainable CAN/serial rate or the physical emergency
procedure.
