

import os
import copy
import traceback

import gnn_main_mot16



MOT17_TEST_SDP_SEQS = [
    "MOT16-01",
    "MOT16-03",
    "MOT16-06",
    "MOT16-07",
    "MOT16-08",
    "MOT16-12",
    "MOT16-14",
]


def run_all():
    base_args = gnn_main_mot16.get_main_args()

    image_root = "datasets/MOT16/test"

    failed = []

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
            gnn_main_mot16.main(args)
            print(f"[DONE] {seq_name}")
        except Exception as e:
            print(f"[FAILED] {seq_name}: {repr(e)}")
            traceback.print_exc()
            failed.append((seq_name, repr(e)))

    print("\n" + "=" * 80)
    if len(failed) == 0:
        print("All MOT16 test sequences finished successfully.")
    else:
        print("Finished with failures/skips:")
        for seq_name, reason in failed:
            print(f"  - {seq_name}: {reason}")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
