

import os
import copy
import json
import traceback
import gnn_main_mot17



MOT17_TEST_SDP_SEQS = [
    "MOT17-01-SDP",
    "MOT17-03-SDP",
    "MOT17-06-SDP",
    "MOT17-07-SDP",
    "MOT17-08-SDP",
    "MOT17-12-SDP",
    "MOT17-14-SDP",
]


def run_all():
    base_args = gnn_main_mot17.get_main_args()

    image_root = "/home/chenjiazhou/datasets/MOT17/images/test"

    failed = []
    sequence_timings = []

    for seq_name in MOT17_TEST_SDP_SEQS:
        args = copy.deepcopy(base_args)
        args.video_id = seq_name
        args.input = os.path.join(image_root, seq_name, "img1")

        print("\n" + "=" * 80)
        print(f"[RUN] {seq_name}")
        print(f"[INPUT] {args.input}")
        print(f"[OUTPUT] {os.path.join(args.output, args.exp_name, seq_name + '.txt')}")
        print("=" * 80)


        try:
            timing = gnn_main_mot17.main(args)
            sequence_timings.append(timing)
            print(f"[DONE] {seq_name}")
        except Exception as e:
            print(f"[FAILED] {seq_name}: {repr(e)}")
            traceback.print_exc()
            failed.append((seq_name, repr(e)))

    print("\n" + "=" * 80)
    if sequence_timings:
        total_frames = sum(item["timed_frames"] for item in sequence_timings)
        total_time = sum(item["update_time_seconds"] for item in sequence_timings)
        overall_fps = total_frames / total_time
        overall_latency_ms = total_time * 1000.0 / total_frames
        summary = {
            "dataset": "MOT17 test (SDP)",
            "successful_sequences": len(sequence_timings),
            "timed_frames": total_frames,
            "update_time_seconds": total_time,
            "latency_ms_per_frame": overall_latency_ms,
            "tracking_fps": overall_fps,
            "aggregation": "sum(timed_frames) / sum(tracker.update time)",
            "scope": sequence_timings[0]["scope"],
            "sequences": sequence_timings,
        }
        timing_dir = os.path.join(base_args.output, base_args.exp_name)
        os.makedirs(timing_dir, exist_ok=True)
        timing_path = os.path.join(timing_dir, "tracking_efficiency.json")
        with open(timing_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

        print("Tracking efficiency summary")
        print(f"  Successful sequences: {len(sequence_timings)}")
        print(f"  Timed frames: {total_frames}")
        print(f"  tracker.update time: {total_time:.3f} s")
        print(f"  Average latency: {overall_latency_ms:.2f} ms/frame")
        print(f"  Overall speed: {overall_fps:.2f} FPS")
        print(f"  Saved to: {timing_path}")
        print("-" * 80)

    if len(failed) == 0:
        print("All MOT17 test SDP sequences finished successfully.")
    else:
        print("Finished with failures/skips:")
        for seq_name, reason in failed:
            print(f"  - {seq_name}: {reason}")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
