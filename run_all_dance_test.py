

import os
import copy
import traceback

import gnn_main_dance as gnn_main



dancetrack_test_seq_names = ['dancetrack0003',
    'dancetrack0009',
    'dancetrack0011',
    'dancetrack0013',
    'dancetrack0017',
    'dancetrack0021',
    'dancetrack0022',
    'dancetrack0028',
    'dancetrack0031',
    'dancetrack0036',
    'dancetrack0038',
    'dancetrack0040',
    'dancetrack0042',
    'dancetrack0046',
    'dancetrack0048',
    'dancetrack0050',
    'dancetrack0054',
    'dancetrack0056',
    'dancetrack0059',
    'dancetrack0060',
    'dancetrack0064',
    'dancetrack0067',
    'dancetrack0070',
    'dancetrack0071',
    'dancetrack0076',
    'dancetrack0078',
    'dancetrack0084',
    'dancetrack0085',
    'dancetrack0088',
    'dancetrack0089',
    'dancetrack0091',
    'dancetrack0092',
    'dancetrack0093',
    'dancetrack0095',
    'dancetrack0100']



def run_all():
    base_args = gnn_main.get_main_args()

    image_root = "datasets/dancetrack/test"

    failed = []

    for seq_name in dancetrack_test_seq_names:
        args = copy.deepcopy(base_args)
        args.video_id = seq_name
        args.aspect_ratio_thresh = 1000.
        args.input = os.path.join(image_root, seq_name, "img1")

        print("\n" + "=" * 80)
        print(f"[RUN] {seq_name}")
        print(f"[INPUT] {args.input}")
        print(f"[OUTPUT] {os.path.join(args.output, args.exp_name, seq_name + '.txt')}")
        print("=" * 80)


        try:
            gnn_main.main(args)
            print(f"[DONE] {seq_name}")
        except Exception as e:
            print(f"[FAILED] {seq_name}: {repr(e)}")
            traceback.print_exc()
            failed.append((seq_name, repr(e)))

    print("\n" + "=" * 80)
    if len(failed) == 0:
        print("All DANCE sequences finished successfully.")
    else:
        print("Finished with failures/skips:")
        for seq_name, reason in failed:
            print(f"  - {seq_name}: {reason}")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
