"""Convert annotated robot trajectories into the SPAVR episode format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.rotations import (
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotvec,
    rotation_matrix_to_quaternion,
)


@dataclass(frozen=True)
class EpisodeSpec:
    episode_dir: Path
    source_relative_path: str
    episode_id: str
    source_episode_id: str
    group_id: str
    task: str
    outcome: str
    annotation: dict
    external_video: Path
    external_view: str


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    num_frames: int
    width: int
    height: int


@dataclass(frozen=True)
class VideoSelection:
    path: Path
    view: str


@dataclass(frozen=True)
class PreparedEpisode:
    episode: EpisodeSpec
    source_video: VideoInfo
    num_steps: int


def raw_end_step_for_episode(spec: EpisodeSpec):
    if spec.outcome == "failure":
        return int(spec.annotation["failure"]["failure_timing"]["failure_end_step"])
    data = spec.annotation["data"]
    pose_path = spec.episode_dir / data["state"]["tcp_pose"]
    action_path = spec.episode_dir / data["action"]
    num_poses = len(np.load(pose_path, mmap_mode="r"))
    num_actions = len(np.load(action_path, mmap_mode="r"))
    # Some sources store one action for every captured observation, including a
    # final command for which no next observation was recorded. Only intervals
    # with both boundary poses can be converted into realized EE deltas.
    return min(num_actions, num_poses - 1)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_view_map(path: Path | None):
    return {} if path is None else read_json(path)


def view_name(video_path: Path):
    name = video_path.stem
    return name[len("episode_") :] if name.startswith("episode_") else name


def choose_external_video(
    episode_dir: Path,
    source_relative_path: str,
    annotation: dict,
    video_glob: str,
    excluded_view_substrings: tuple[str, ...],
    view_map: dict,
    ambiguous_view_policy: str,
):
    """Select a video together with the matching annotation camera key.

    Newer datasets explicitly map view keys to video paths in ``data.videos``.
    Those keys are authoritative because filenames such as ``episode.mp4`` do
    not encode that their calibration key is ``primary``. Older datasets fall
    back to filename-based discovery.
    """

    def excluded(selection: VideoSelection):
        searchable = f"{selection.view} {selection.path.name}"
        return any(token in searchable for token in excluded_view_substrings)

    annotated_videos = annotation.get("data", {}).get("videos")
    if annotated_videos:
        if not isinstance(annotated_videos, dict):
            raise ValueError(
                f"{source_relative_path}: annotation data.videos must be an object"
            )
        declared = [
            VideoSelection(episode_dir / relative_path, str(view))
            for view, relative_path in annotated_videos.items()
        ]
        candidates = sorted(
            (
                selection
                for selection in declared
                if not excluded(selection) and selection.path.is_file()
            ),
            key=lambda selection: (selection.view, str(selection.path)),
        )
    else:
        candidates = sorted(
            (
                VideoSelection(path, view_name(path))
                for path in (episode_dir / "video").glob(video_glob)
            ),
            key=lambda selection: (selection.view, str(selection.path)),
        )
        candidates = [selection for selection in candidates if not excluded(selection)]

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"{source_relative_path}: no external video found")

    requested = view_map.get(source_relative_path, view_map.get(episode_dir.name))
    if requested is not None:
        matches = [
            selection
            for selection in candidates
            if requested
            in {
                selection.view,
                selection.path.name,
                selection.path.stem,
                view_name(selection.path),
            }
        ]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"{source_relative_path}: view map value {requested!r} does not select "
            "exactly one of "
            f"{[f'{item.view}:{item.path.name}' for item in candidates]}"
        )

    if ambiguous_view_policy == "skip":
        return None
    raise ValueError(
        f"{source_relative_path}: multiple external videos "
        f"{[f'{item.view}:{item.path.name}' for item in candidates]}; "
        "add an entry to --view-map"
    )


def discover_episodes(args):
    root = args.input_root
    allowed_tasks = None if args.tasks is None else set(args.tasks)
    allowed_episode_ids = None if args.episode_ids is None else set(args.episode_ids)
    view_map = load_view_map(args.view_map)
    episodes = []
    skipped_ambiguous = []
    skipped_permission_denied = []
    skipped_nonpositive_failure_intervals = []

    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if allowed_tasks is not None and task_dir.name not in allowed_tasks:
            continue
        for outcome in args.outcomes:
            outcome_dir = task_dir / outcome
            if not outcome_dir.is_dir():
                continue
            for episode_dir in sorted(path for path in outcome_dir.iterdir() if path.is_dir()):
                if (
                    allowed_episode_ids is not None
                    and episode_dir.name not in allowed_episode_ids
                ):
                    continue
                relative_path = episode_dir.relative_to(root).as_posix()
                annotation_path = episode_dir / "annotation.json"
                try:
                    annotation = read_json(annotation_path)
                except FileNotFoundError:
                    continue
                except PermissionError:
                    skipped_permission_denied.append(relative_path)
                    print(
                        f"warning: skipping permission-denied episode: {relative_path}",
                        file=sys.stderr,
                    )
                    continue
                if outcome == "failure":
                    timing = annotation.get("failure", {}).get("failure_timing")
                    if timing is not None and int(timing["failure_onset_step"]) >= int(
                        timing["failure_end_step"]
                    ):
                        skipped_nonpositive_failure_intervals.append(relative_path)
                        print(
                            "warning: skipping non-positive failure interval: "
                            f"{relative_path}",
                            file=sys.stderr,
                        )
                        continue
                selection = choose_external_video(
                    episode_dir,
                    relative_path,
                    annotation,
                    args.video_glob,
                    tuple(args.excluded_view_substrings),
                    view_map,
                    args.ambiguous_view_policy,
                )
                if selection is None:
                    skipped_ambiguous.append(relative_path)
                    continue
                source_episode_id = annotation.get("episode_id", episode_dir.name)
                episodes.append(
                    EpisodeSpec(
                        episode_dir=episode_dir,
                        source_relative_path=relative_path,
                        episode_id=f"{source_episode_id}__{outcome}",
                        source_episode_id=source_episode_id,
                        group_id=source_episode_id,
                        task=annotation.get("task", {}).get("task_name", task_dir.name),
                        outcome=outcome,
                        annotation=annotation,
                        external_video=selection.path,
                        external_view=selection.view,
                    )
                )

    episode_ids = [episode.episode_id for episode in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episode_id must be unique across the processed dataset")
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    return (
        episodes,
        skipped_ambiguous,
        skipped_permission_denied,
        skipped_nonpositive_failure_intervals,
    )


def probe_video(path: Path, ffprobe_bin: str):
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,width,height",
        "-of",
        "json",
        str(path),
    ]
    stream = json.loads(subprocess.check_output(command, text=True))["streams"][0]
    return VideoInfo(
        fps=float(Fraction(stream["avg_frame_rate"])),
        num_frames=int(stream["nb_frames"]),
        width=int(stream["width"]),
        height=int(stream["height"]),
    )


def ffmpeg_candidates(requested: str):
    """Return executable FFmpeg candidates in preference order."""
    if requested != "auto":
        resolved = shutil.which(requested)
        if resolved is None:
            raise ValueError(f"FFmpeg executable not found: {requested}")
        return [Path(resolved)]

    candidates = []
    seen = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = (Path(directory) / "ffmpeg").resolve()
        if candidate in seen or not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        seen.add(candidate)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("no FFmpeg executable found in PATH")
    return candidates


def ffmpeg_supports_encoder(ffmpeg_bin: Path, encoder: str):
    result = subprocess.run(
        [str(ffmpeg_bin), "-hide_banner", "-encoders"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return False
    return any(
        len(fields) >= 2 and fields[1] == encoder
        for fields in (line.split() for line in result.stdout.splitlines())
    )


def resolve_ffmpeg_bin(requested: str, encoder: str):
    candidates = ffmpeg_candidates(requested)
    for candidate in candidates:
        if ffmpeg_supports_encoder(candidate, encoder):
            return str(candidate)
    inspected = ", ".join(str(candidate) for candidate in candidates)
    raise ValueError(
        f"FFmpeg encoder {encoder!r} is unavailable in: {inspected}. "
        "Install an FFmpeg build with that encoder, choose another "
        "--video-codec, or pass --ffmpeg-bin explicitly."
    )


def make_resample_indices(raw_end_step: int, source_fps: float, target_fps: float):
    """Nearest-neighbor state indices on a regular target-rate time grid."""
    output_end_step = round(raw_end_step * target_fps / source_fps)
    output_steps = np.arange(output_end_step + 1, dtype=np.float64)
    indices = np.rint(output_steps * source_fps / target_fps).astype(np.int64)
    indices = np.clip(indices, 0, raw_end_step)
    indices[-1] = raw_end_step
    if np.any(np.diff(indices) <= 0):
        raise ValueError("target FPS must produce strictly increasing source indices")
    return indices


def balance_failure_boundaries(raw_indices, raw_onset_step: int):
    """Trim an overlong pre-failure prefix while preserving a contiguous episode.

    ``build_failure_reward`` produces ``onset_boundary`` rewards equal to the
    prefix reward and ``num_steps - onset_boundary`` declining rewards.  Keep
    the complete declining part and, when possible, the same number of prefix
    transitions immediately before it.
    """
    onset_boundary = int(np.argmin(np.abs(raw_indices - raw_onset_step)))
    num_steps = len(raw_indices) - 1
    prefix_steps = onset_boundary
    declining_steps = num_steps - onset_boundary
    trim_steps = max(prefix_steps - declining_steps, 0)
    return raw_indices[trim_steps:], trim_steps


def select_resampled_boundaries(spec: EpisodeSpec, source_fps: float, args):
    """Select the boundary states retained in one processed episode."""
    raw_end_step = raw_end_step_for_episode(spec)
    raw_indices = make_resample_indices(
        raw_end_step,
        source_fps,
        args.target_fps,
    )
    start_step = 0
    if spec.outcome == "success":
        required_boundaries = args.success_clip_frames + 1
        if len(raw_indices) < required_boundaries:
            raise ValueError(
                f"{spec.source_relative_path}: only {len(raw_indices) - 1} "
                f"resampled steps, fewer than "
                f"success_clip_frames={args.success_clip_frames}"
            )
        start_step = len(raw_indices) - required_boundaries
        raw_indices = raw_indices[-required_boundaries:]
    elif args.balance_failure_phases:
        raw_onset_step = int(
            spec.annotation["failure"]["failure_timing"]["failure_onset_step"]
        )
        raw_indices, start_step = balance_failure_boundaries(
            raw_indices,
            raw_onset_step,
        )
    return raw_indices, start_step


def prepare_episode(spec: EpisodeSpec, args):
    source_video = probe_video(spec.external_video, args.ffprobe_bin)
    if args.target_fps > source_video.fps:
        raise ValueError(
            f"{spec.source_relative_path}: target FPS {args.target_fps} exceeds "
            f"source FPS {source_video.fps}"
        )

    data = spec.annotation["data"]
    raw_pose = np.load(
        spec.episode_dir / data["state"]["tcp_pose"],
        mmap_mode="r",
    )
    raw_action = np.load(spec.episode_dir / data["action"], mmap_mode="r")
    if (
        raw_pose.ndim != 2
        or raw_action.ndim != 2
        or raw_pose.shape[1:] != (7,)
        or raw_action.shape[1:] != (7,)
        or len(raw_pose) not in {len(raw_action), len(raw_action) + 1}
    ):
        raise ValueError(
            f"{spec.source_relative_path}: expected pose (N,7) or (N+1,7) "
            "with action (N,7), "
            f"got {raw_pose.shape} and {raw_action.shape}"
        )
    # Validate the selected view's calibration during dry-run as well as before
    # starting any output writes.
    load_camera_extrinsics(spec, len(raw_pose))

    raw_end_step = raw_end_step_for_episode(spec)

    if source_video.num_frames < raw_end_step + 1:
        raise ValueError(
            f"{spec.source_relative_path}: video has {source_video.num_frames} frames "
            f"but state index {raw_end_step} is required"
        )

    raw_indices, _ = select_resampled_boundaries(spec, source_video.fps, args)
    num_steps = len(raw_indices) - 1
    return PreparedEpisode(spec, source_video, num_steps)


def prepare_and_filter_episodes(episodes, args):
    prepared = [None] * len(episodes)
    skipped_permission_denied = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(prepare_episode, episode, args): index
            for index, episode in enumerate(episodes)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                prepared[index] = future.result()
            except PermissionError:
                relative_path = episodes[index].source_relative_path
                skipped_permission_denied.append(relative_path)
                print(
                    f"warning: skipping permission-denied episode: {relative_path}",
                    file=sys.stderr,
                )

    prepared = [episode for episode in prepared if episode is not None]
    if not prepared:
        return [], {
            "percentile_range": list(args.length_percentiles),
            "frame_range": None,
            "num_candidates": 0,
            "num_kept": 0,
            "num_removed": 0,
            "removed_episodes": [],
        }, skipped_permission_denied

    frame_counts = np.asarray(
        [episode.num_steps for episode in prepared], dtype=np.int64
    )
    low_percentile, high_percentile = args.length_percentiles
    low_frames, high_frames = np.percentile(
        frame_counts,
        [low_percentile, high_percentile],
        method="nearest",
    ).astype(np.int64)
    kept = [
        episode
        for episode in prepared
        if low_frames <= episode.num_steps <= high_frames
    ]
    removed = [
        {
            "episode_id": episode.episode.episode_id,
            "num_steps": episode.num_steps,
        }
        for episode in prepared
        if episode.num_steps < low_frames or episode.num_steps > high_frames
    ]
    summary = {
        "percentile_range": [low_percentile, high_percentile],
        "frame_range": [int(low_frames), int(high_frames)],
        "num_candidates": len(prepared),
        "num_kept": len(kept),
        "num_removed": len(removed),
        "removed_episodes": removed,
    }
    return kept, summary, skipped_permission_denied


def reorder_quaternion(quaternion, order: str):
    if order == "xyzw":
        return quaternion
    return quaternion[..., [1, 2, 3, 0]]


def load_camera_extrinsics(spec: EpisodeSpec, num_poses: int):
    """Load the selected video view's world-to-camera transforms."""
    camera_paths = spec.annotation["data"].get("camera", {})
    camera_relative_path = camera_paths.get(spec.external_view)
    if camera_relative_path is None:
        raise ValueError(
            f"{spec.source_relative_path}: selected view {spec.external_view!r} "
            "has no camera calibration in annotation data.camera"
        )

    camera_path = spec.episode_dir / camera_relative_path
    camera = read_json(camera_path)
    convention = camera.get("convention")
    if convention != "opencv":
        raise ValueError(
            f"{spec.source_relative_path}: camera {spec.external_view!r} uses "
            f"unsupported convention {convention!r}; expected 'opencv'"
        )

    extrinsics_relative_path = camera.get("T_cw_path")
    if extrinsics_relative_path is not None:
        extrinsics = np.load(spec.episode_dir / extrinsics_relative_path)
    elif "T_cw" in camera:
        extrinsics = np.asarray(camera["T_cw"], dtype=np.float64)
    else:
        raise ValueError(
            f"{spec.source_relative_path}: camera {spec.external_view!r} has "
            "neither T_cw_path nor T_cw"
        )

    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.shape == (3, 4):
        extrinsics = np.concatenate(
            [extrinsics, np.asarray([[0.0, 0.0, 0.0, 1.0]])],
            axis=0,
        )
    elif extrinsics.ndim == 3 and extrinsics.shape[1:] == (3, 4):
        homogeneous_rows = np.broadcast_to(
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            (len(extrinsics), 1, 4),
        )
        extrinsics = np.concatenate([extrinsics, homogeneous_rows], axis=1)

    if extrinsics.shape == (4, 4):
        extrinsics = np.broadcast_to(extrinsics, (num_poses, 4, 4))
    elif extrinsics.shape != (num_poses, 4, 4):
        raise ValueError(
            f"{spec.source_relative_path}: camera {spec.external_view!r} T_cw "
            "must have shape (3,4), (4,4), "
            f"({num_poses},3,4), or ({num_poses},4,4), got {extrinsics.shape}"
        )
    if not np.all(np.isfinite(extrinsics)):
        raise ValueError(f"{spec.source_relative_path}: T_cw contains non-finite values")
    if not np.allclose(extrinsics[:, 3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6):
        raise ValueError(f"{spec.source_relative_path}: T_cw has invalid homogeneous rows")

    rotation = extrinsics[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(rotation @ rotation.transpose(0, 2, 1), identity, atol=1.0e-5):
        raise ValueError(f"{spec.source_relative_path}: T_cw rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
        raise ValueError(f"{spec.source_relative_path}: T_cw rotations must have determinant 1")
    return camera_relative_path, convention, extrinsics


def build_trajectory(
    raw_pose,
    raw_action,
    raw_t_cw,
    raw_indices,
    target_fps: float,
    quaternion_order: str,
    position_scale: float,
    gripper_action_index: int,
):
    sampled_pose = np.asarray(raw_pose[raw_indices], dtype=np.float64)
    sampled_t_cw = np.asarray(raw_t_cw[raw_indices], dtype=np.float64)
    rotation_cw = sampled_t_cw[:, :3, :3]
    translation_cw = sampled_t_cw[:, :3, 3]

    world_position = sampled_pose[:, :3] * position_scale
    world_quaternion = reorder_quaternion(
        sampled_pose[:, 3:7], quaternion_order
    )
    camera_quaternion = rotation_matrix_to_quaternion(rotation_cw)

    # T_ce = T_cw @ T_we: the proprio pose is expressed in the coordinate
    # system of the image captured at the same source boundary.
    camera_position = (
        np.einsum("tij,tj->ti", rotation_cw, world_position)
        + translation_cw
    )
    camera_ee_quaternion = quaternion_multiply(
        camera_quaternion,
        world_quaternion,
    )

    # Express each physical world-frame action in the camera axes at the
    # action's starting observation. Translation must not affect a delta, and
    # a rotation delta changes basis by R_cw dR R_cw^-1. This definition also
    # remains meaningful if a moving camera is explicitly selected.
    world_delta_position = world_position[1:] - world_position[:-1]
    world_delta_quaternion = quaternion_multiply(
        world_quaternion[1:],
        quaternion_conjugate(world_quaternion[:-1]),
    )
    camera_delta_quaternion = quaternion_multiply(
        quaternion_multiply(camera_quaternion[:-1], world_delta_quaternion),
        quaternion_conjugate(camera_quaternion[:-1]),
    )

    action = np.empty((len(raw_indices) - 1, 7), dtype=np.float32)
    action[:, :3] = np.einsum(
        "tij,tj->ti", rotation_cw[:-1], world_delta_position
    )
    action[:, 3:6] = quaternion_to_rotvec(camera_delta_quaternion)
    gripper_indices = np.maximum(raw_indices[1:] - 1, 0)
    action[:, 6] = raw_action[gripper_indices, gripper_action_index]

    proprio = np.empty((len(action), 7), dtype=np.float32)
    proprio[:, :3] = camera_position[:-1]
    proprio[:, 3:6] = quaternion_to_rotvec(camera_ee_quaternion[:-1])
    initial_gripper_index = max(int(raw_indices[0]) - 1, 0)
    proprio[0, 6] = raw_action[initial_gripper_index, gripper_action_index]
    proprio[1:, 6] = action[:-1, 6]
    timestamp = np.arange(len(action), dtype=np.float64) / target_fps
    return proprio, action, timestamp


def build_failure_reward(
    raw_indices,
    raw_onset_step: int,
    prefix_reward: float,
    terminal_reward: float,
):
    onset_boundary = int(np.argmin(np.abs(raw_indices - raw_onset_step)))
    quality = np.full(len(raw_indices), prefix_reward, dtype=np.float32)
    quality[onset_boundary:] = np.linspace(
        prefix_reward,
        terminal_reward,
        len(raw_indices) - onset_boundary,
        dtype=np.float32,
    )
    return quality[1:], max(onset_boundary - 1, 0)


def transcode_video(
    source: Path,
    destination: Path,
    start_step: int,
    num_steps: int,
    args,
):
    height, width = args.image_size
    if args.resize_mode == "cover":
        resize_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
    else:
        resize_filter = f"scale={width}:{height}"
    filters = (
        f"fps={args.target_fps}:start_time=0:round=near,"
        f"trim=start_frame={start_step}:end_frame={start_step + num_steps},"
        f"{resize_filter},"
        f"setpts=N/({args.target_fps}*TB)"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        filters,
        "-an",
        "-frames:v",
        str(num_steps),
        "-r",
        str(args.target_fps),
        "-c:v",
        args.video_codec,
        "-pix_fmt",
        "yuv420p",
    ]
    if args.video_codec == "libx264":
        command.extend([
            "-preset",
            args.video_preset,
            "-crf",
            str(args.video_crf),
        ])
    command.extend([
        "-threads",
        str(args.ffmpeg_threads),
        str(destination),
    ])
    subprocess.run(command, check=True)


def validate_existing_episode_output(
    trajectory_path: Path,
    video_path: Path,
    expected_trajectory: dict,
    args,
    source_relative_path: str,
):
    """Validate completed files before reusing them in --resume-output mode."""
    with np.load(trajectory_path) as stored:
        if set(stored.files) != set(expected_trajectory):
            raise ValueError(
                f"{source_relative_path}: existing trajectory fields do not match: "
                f"{trajectory_path}"
            )
        for name, expected in expected_trajectory.items():
            actual = stored[name]
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=1.0e-5, atol=1.0e-7
            ):
                raise ValueError(
                    f"{source_relative_path}: existing trajectory {name!r} does not "
                    f"match current preprocessing settings: {trajectory_path}"
                )

    video = probe_video(video_path, args.ffprobe_bin)
    expected_height, expected_width = args.image_size
    if (
        video.num_frames != len(expected_trajectory["proprio"])
        or video.width != expected_width
        or video.height != expected_height
        or not np.isclose(video.fps, args.target_fps, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError(
            f"{source_relative_path}: existing video does not match current "
            f"preprocessing settings: {video_path}"
        )
    return video


def split_name(group_id: str, ratios, seed: int):
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    train_ratio, val_ratio, _ = ratios
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def process_episode(prepared: PreparedEpisode, task_id: int, split: str, args):
    spec = prepared.episode
    data = spec.annotation["data"]
    raw_pose = np.load(spec.episode_dir / data["state"]["tcp_pose"])
    raw_action = np.load(spec.episode_dir / data["action"])
    if (
        raw_pose.ndim != 2
        or raw_action.ndim != 2
        or raw_pose.shape[1:] != (7,)
        or raw_action.shape[1:] != (7,)
        or len(raw_pose) not in {len(raw_action), len(raw_action) + 1}
    ):
        raise ValueError(
            f"{spec.source_relative_path}: expected pose (N,7) or (N+1,7) "
            "with action (N,7), "
            f"got {raw_pose.shape} and {raw_action.shape}"
        )
    camera_relative_path, camera_convention, raw_t_cw = load_camera_extrinsics(
        spec,
        len(raw_pose),
    )

    source_video = prepared.source_video

    available_transition_steps = min(len(raw_action), len(raw_pose) - 1)
    raw_end_step = available_transition_steps
    raw_onset_step = None
    if spec.outcome == "failure":
        timing = spec.annotation["failure"]["failure_timing"]
        raw_onset_step = int(timing["failure_onset_step"])
        raw_end_step = int(timing["failure_end_step"])
        if not 0 <= raw_onset_step < raw_end_step <= available_transition_steps:
            raise ValueError(f"{spec.source_relative_path}: invalid failure onset/end")
    if source_video.num_frames < raw_end_step + 1:
        raise ValueError(
            f"{spec.source_relative_path}: video has {source_video.num_frames} frames "
            f"but state index {raw_end_step} is required"
        )

    raw_indices, start_step = select_resampled_boundaries(
        spec,
        source_video.fps,
        args,
    )

    proprio, action, timestamp = build_trajectory(
        raw_pose,
        raw_action,
        raw_t_cw,
        raw_indices,
        args.target_fps,
        args.quaternion_order,
        args.position_scale,
        args.gripper_action_index,
    )
    if spec.outcome == "success":
        reward = np.full(len(action), args.success_reward, dtype=np.float32)
        failure_onset_step = None
        failure_end_step = None
    else:
        reward, failure_onset_step = build_failure_reward(
            raw_indices,
            raw_onset_step,
            args.failure_prefix_reward,
            args.failure_terminal_reward,
        )
        failure_end_step = len(action) - 1

    if not len(proprio) == len(action) == len(reward) == len(timestamp):
        raise ValueError(f"{spec.source_relative_path}: temporal fields are not aligned")
    if len(action) != prepared.num_steps:
        raise ValueError(
            f"{spec.source_relative_path}: prepared length {prepared.num_steps} "
            f"does not match trajectory length {len(action)}"
        )

    trajectory_relative = Path("episodes") / f"{spec.episode_id}.npz"
    video_relative = Path("videos") / spec.episode_id / "external.mp4"
    trajectory_output = args.output_root / trajectory_relative
    video_output = args.output_root / video_relative

    observation_indices = raw_indices[:-1]
    expected_trajectory = {
        "proprio": proprio,
        "action": action,
        "reward": reward,
        "timestamp": timestamp,
    }
    reuse_existing = (
        args.resume_output and trajectory_output.is_file() and video_output.is_file()
    )
    if reuse_existing:
        output_video = validate_existing_episode_output(
            trajectory_output,
            video_output,
            expected_trajectory,
            args,
            spec.source_relative_path,
        )
    else:
        transcode_video(
            spec.external_video,
            video_output,
            start_step,
            len(action),
            args,
        )
        output_video = probe_video(video_output, args.ffprobe_bin)
    if output_video.num_frames != len(proprio):
        raise ValueError(
            f"{spec.source_relative_path}: output video has {output_video.num_frames} "
            f"frames but trajectory has {len(proprio)} states"
        )

    if not reuse_existing:
        trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = trajectory_output.with_suffix(".npz.tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, **expected_trajectory)
        temporary.replace(trajectory_output)

    return {
        "episode_id": spec.episode_id,
        "task": spec.task,
        "task_id": task_id,
        "outcome": spec.outcome,
        "trajectory": trajectory_relative.as_posix(),
        "video": video_relative.as_posix(),
        "external_view": spec.external_view,
        "coordinate_frame": f"camera:{spec.external_view}",
        "camera_convention": camera_convention,
        "source_camera_calibration": camera_relative_path,
        "num_steps": len(action),
        "failure_onset_step": failure_onset_step,
        "failure_end_step": failure_end_step,
        "num_prefix_reward_steps": int(
            np.isclose(reward, args.failure_prefix_reward).sum()
        ) if spec.outcome == "failure" else None,
        "num_declining_reward_steps": int(
            (reward < args.failure_prefix_reward).sum()
        ) if spec.outcome == "failure" else None,
        "split": split,
        "group_id": spec.group_id,
        "source_episode_id": spec.source_episode_id,
        "source_episode": spec.source_relative_path,
        "source_fps": source_video.fps,
        "source_frame_start": int(observation_indices[0]),
        "source_frame_end": int(observation_indices[-1]),
    }


def validate_args(args):
    if not args.input_root.is_dir():
        raise ValueError(f"input root does not exist: {args.input_root}")
    if args.success_clip_frames < 2:
        raise ValueError("success_clip_frames must be at least 2")
    if not (
        0.0 <= args.failure_terminal_reward
        < args.failure_prefix_reward
        < args.success_reward
        <= 1.0
    ):
        raise ValueError(
            "rewards must satisfy 0 <= failure_terminal < failure_prefix "
            "< success <= 1"
        )
    if not np.isclose(sum(args.split_ratios), 1.0):
        raise ValueError("split ratios must sum to 1")
    low_percentile, high_percentile = args.length_percentiles
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError(
            "length percentiles must satisfy 0 <= LOW < HIGH <= 100"
        )
    if args.workers < 1 or args.ffmpeg_threads < 1:
        raise ValueError("workers and ffmpeg_threads must be positive")
    if args.normalization_epsilon <= 0.0:
        raise ValueError("normalization epsilon must be positive")
    if args.position_scale <= 0.0:
        raise ValueError("position_scale must be positive")


def prepare_output_root(output_root: Path, resume_output: bool):
    if output_root.exists() and any(output_root.iterdir()) and not resume_output:
        raise ValueError(f"output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def update_running_stats(stats, values):
    """Merge one ``(N, D)`` array into numerically stable running moments."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"normalization values must have shape (N,D), got {values.shape}")

    batch_count = values.shape[0]
    batch_mean = values.mean(axis=0)
    batch_m2 = np.square(values - batch_mean).sum(axis=0)
    if stats["count"] == 0:
        stats["count"] = batch_count
        stats["mean"] = batch_mean
        stats["m2"] = batch_m2
        return

    total_count = stats["count"] + batch_count
    delta = batch_mean - stats["mean"]
    stats["m2"] += (
        batch_m2
        + np.square(delta) * stats["count"] * batch_count / total_count
    )
    stats["mean"] += delta * batch_count / total_count
    stats["count"] = total_count


def finalize_running_stats(stats, epsilon: float):
    if stats["count"] == 0:
        raise ValueError("cannot compute normalization from an empty split")
    variance = np.maximum(stats["m2"] / stats["count"], 0.0)
    std = np.maximum(np.sqrt(variance), epsilon)
    return {
        "mean": stats["mean"].tolist(),
        "std": std.tolist(),
    }


def compute_normalization(output_root: Path, manifests, epsilon: float):
    """Compute action/proprio moments from processed train episodes only."""
    running = {
        "proprio": {"count": 0, "mean": None, "m2": None},
        "action": {"count": 0, "mean": None, "m2": None},
    }
    train_episodes = 0
    for manifest in manifests:
        if manifest["split"] != "train":
            continue
        train_episodes += 1
        with np.load(output_root / manifest["trajectory"]) as trajectory:
            update_running_stats(running["proprio"], trajectory["proprio"])
            update_running_stats(running["action"], trajectory["action"])

    if running["proprio"]["count"] != running["action"]["count"]:
        raise ValueError("proprio/action normalization counts do not match")
    return {
        "schema_version": "1.0",
        "computed_from_split": "train",
        "num_episodes": train_episodes,
        "num_steps": running["action"]["count"],
        "epsilon": epsilon,
        "formula": "(x - mean) / std",
        "proprio": finalize_running_stats(running["proprio"], epsilon),
        "action": finalize_running_stats(running["action"], epsilon),
    }


def build_metadata(
    args,
    num_episodes: int,
    skipped_ambiguous,
    skipped_permission_denied,
    skipped_nonpositive_failure_intervals,
    length_filter,
):
    height, width = args.image_size
    return {
        "schema_version": "1.1",
        "num_episodes": num_episodes,
        "fps": args.target_fps,
        "image_size": [height, width],
        "resize_mode": args.resize_mode,
        "video_input": "external_only",
        "coordinate_frame": "selected_external_camera",
        "camera_convention": "opencv",
        "camera_extrinsics": "T_cw_world_to_camera",
        "proprio_coordinate_transform": "T_ce[t] = T_cw[t] @ T_we[t]",
        "action_coordinate_transform": "world_delta_rotated_by_R_cw_at_action_start",
        "success_clip_policy": "tail",
        "success_clip_frames": args.success_clip_frames,
        "proprio_layout": "xyz_rotvec_previous_gripper",
        "proprio_dim": 7,
        "action_layout": "delta_xyz_delta_rotvec_gripper_command",
        "action_dim": 7,
        "gripper_action_index": args.gripper_action_index,
        "reward_layout": "trajectory_quality_v1",
        "reward_range": [0.0, 1.0],
        "success_reward": args.success_reward,
        "failure_prefix_reward": args.failure_prefix_reward,
        "failure_terminal_reward": args.failure_terminal_reward,
        "balance_failure_phases": args.balance_failure_phases,
        "failure_balance_policy": (
            "trim_leading_prefix_to_declining_length"
            if args.balance_failure_phases
            else None
        ),
        "normalization": "normalization.json",
        "normalization_split": "train",
        "quaternion_order": args.quaternion_order,
        "position_scale": args.position_scale,
        "position_unit": "meter",
        "rotation_unit": "radian",
        "resampling": "nearest_source_frame",
        "temporal_alignment": "observation_action_reward_same_step",
        "video_codec": args.video_codec,
        "ffmpeg_bin": args.ffmpeg_bin,
        "video_crf": args.video_crf if args.video_codec == "libx264" else None,
        "video_preset": args.video_preset if args.video_codec == "libx264" else None,
        "split_ratios": {
            "train": args.split_ratios[0],
            "val": args.split_ratios[1],
            "test": args.split_ratios[2],
        },
        "split_seed": args.split_seed,
        "skipped_ambiguous_views": skipped_ambiguous,
        "skipped_permission_denied": skipped_permission_denied,
        "skipped_nonpositive_failure_intervals": (
            skipped_nonpositive_failure_intervals
        ),
        "resumed_output": bool(args.resume_output),
        "length_filter": length_filter,
    }


def run(args):
    validate_args(args)
    if not args.dry_run:
        args.ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin, args.video_codec)
        print(
            f"using FFmpeg: {args.ffmpeg_bin} (encoder={args.video_codec})",
            file=sys.stderr,
        )
    (
        episodes,
        skipped_ambiguous,
        skipped_permission_denied,
        skipped_nonpositive_failure_intervals,
    ) = discover_episodes(args)
    if not episodes:
        raise ValueError("no episodes selected")
    prepared_episodes, length_filter, skipped_during_prepare = (
        prepare_and_filter_episodes(episodes, args)
    )
    skipped_permission_denied.extend(skipped_during_prepare)
    skipped_permission_denied = sorted(set(skipped_permission_denied))
    if not prepared_episodes:
        raise ValueError("no episodes remain after length percentile filtering")
    if args.dry_run:
        counts = {}
        for prepared in prepared_episodes:
            episode = prepared.episode
            key = f"{episode.task}/{episode.outcome}"
            counts[key] = counts.get(key, 0) + 1
        print(json.dumps({
            "episodes": len(prepared_episodes),
            "counts": counts,
            "length_filter": length_filter,
            "skipped_ambiguous_views": skipped_ambiguous,
            "skipped_permission_denied": skipped_permission_denied,
            "skipped_nonpositive_failure_intervals": (
                skipped_nonpositive_failure_intervals
            ),
        }, ensure_ascii=False, indent=2))
        return

    prepare_output_root(args.output_root, args.resume_output)
    tasks = sorted({prepared.episode.task for prepared in prepared_episodes})
    task_vocab = {task: task_id for task_id, task in enumerate(tasks)}
    assignments = {
        prepared.episode.episode_id: split_name(
            prepared.episode.group_id,
            args.split_ratios,
            args.split_seed,
        )
        for prepared in prepared_episodes
    }

    manifests = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_episode,
                prepared,
                task_vocab[prepared.episode.task],
                assignments[prepared.episode.episode_id],
                args,
            ): prepared.episode
            for prepared in prepared_episodes
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            episode = futures[future]
            manifests.append(future.result())
            print(
                f"[{completed}/{len(prepared_episodes)}] "
                f"{episode.source_relative_path}"
            )

    manifests.sort(key=lambda item: (item["task"], item["outcome"], item["episode_id"]))
    with (args.output_root / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for manifest in manifests:
            file.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    splits = {"train": [], "val": [], "test": []}
    for manifest in manifests:
        splits[manifest["split"]].append(manifest["episode_id"])
    write_json(args.output_root / "metadata.json", build_metadata(
        args,
        len(manifests),
        skipped_ambiguous,
        skipped_permission_denied,
        skipped_nonpositive_failure_intervals,
        length_filter,
    ))
    write_json(args.output_root / "task_vocab.json", task_vocab)
    write_json(args.output_root / "splits.json", splits)
    write_json(
        args.output_root / "normalization.json",
        compute_normalization(
            args.output_root,
            manifests,
            args.normalization_epsilon,
        ),
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess annotated robot episodes for SPAVR training."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument(
        "--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"),
        default=(224, 224),
    )
    parser.add_argument("--resize-mode", choices=("cover", "stretch"), default="cover")
    parser.add_argument("--success-clip-frames", type=int, default=16)
    parser.add_argument(
        "--length-percentiles", type=float, nargs=2, metavar=("LOW", "HIGH"),
        default=(1.0, 99.0),
        help="keep episodes whose processed frame counts fall within this percentile range",
    )
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--failure-prefix-reward", type=float, default=0.8)
    parser.add_argument("--failure-terminal-reward", type=float, default=0.2)
    parser.add_argument(
        "--balance-failure-phases",
        action="store_true",
        help=(
            "keep the full declining failure phase and trim an overlong leading "
            "prefix so the two phases have approximately equal frame counts"
        ),
    )
    parser.add_argument("--normalization-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--quaternion-order", choices=("xyzw", "wxyz"), default="wxyz")
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--gripper-action-index", type=int, default=6)
    parser.add_argument(
        "--outcomes", nargs="+", choices=("success", "failure"),
        default=("success", "failure"),
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--episode-ids", nargs="+", default=None)
    parser.add_argument("--video-glob", default="episode_*.mp4")
    parser.add_argument(
        "--excluded-view-substrings", nargs="+", default=("eye_in_hand", "wrist")
    )
    parser.add_argument("--view-map", type=Path, default=None)
    parser.add_argument(
        "--ambiguous-view-policy", choices=("error", "skip"), default="error"
    )
    parser.add_argument(
        "--split-ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-preset", default="fast")
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument(
        "--ffmpeg-bin",
        default="auto",
        help=(
            "FFmpeg executable or 'auto'; auto scans PATH and selects the first "
            "binary that provides --video-codec"
        ),
    )
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--resume-output",
        action="store_true",
        help=(
            "reuse complete episode files in a non-empty output root after "
            "validating them against the current preprocessing settings"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
