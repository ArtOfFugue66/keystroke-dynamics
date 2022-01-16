from __future__ import annotations

import copy
from typing import List, Tuple

import keyboard as kb
import numpy as np
import pandas as pd
from loguru import logger
import conf

# finished_capturing = False
# event_timing_dict = {}
timestamp_pairs = []
no_up_events = 0


def sequence_events_to_df(event_pair_list: List[Tuple[kb.KeyboardEvent, kb.KeyboardEvent]]) -> pd.DataFrame | None:
    press_times, release_times, letters = [], [], []

    for timestamp_tuple in event_pair_list:
        letters.append(timestamp_tuple[0].name)
        press_times.append(timestamp_tuple[0].time)
        release_times.append(timestamp_tuple[1].time)

    df = pd.DataFrame()

    participant_id_col = pd.Series([conf.PARTICIPANT_ID] * len(press_times), dtype=np.int32)
    df['PARTICIPANT_ID'] = participant_id_col
    test_section_id_col = pd.Series([conf.TEST_SECTION_ID] * len(press_times), dtype=np.int32)
    df['TEST_SECTION_ID'] = test_section_id_col
    press_times_col = pd.Series(press_times, dtype=np.float64)
    df['PRESS_TIME'] = press_times_col
    release_times_col = pd.Series(release_times, dtype=np.float64)
    df['RELEASE_TIME'] = release_times_col
    letters_col = pd.Series(letters, dtype=str)
    df['LETTER'] = letters_col

    return df


def handle_key_hold(kb_event: kb.KeyboardEvent) -> bool:
    """
    Check whether the user is holding down the key associated with an event
    :param kb_event: Event to check for duplicates
    :return: True if user is holding down the key, False otherwise
    """
    global event_timing_dict

    if len(event_timing_dict) <= 0:
        return False

    scan_code = kb_event.scan_code

    # If the key has already been pressed but has no release timestamp
    if scan_code in event_timing_dict.keys() and kb_event.event_type == "down":
        return True
    else:
        return False  # Key already has a press timestamp and a release timestamp


@logger.catch
def keypress_callback(kb_event: kb.KeyboardEvent) -> None:
    global finished_capturing, event_timing_dict, timestamp_pairs, no_up_events

    if len(timestamp_pairs) >= conf.MAX_SEQUENCE_LENGTH:  # If enough keyup events are captured
        print("[INFO] Keystroke buffer limit reached!")
        return

    if kb_event.event_type == "down":
        if handle_key_hold(kb_event):  # If key is being held down
            return  # Do not add event to the list
        else:
            event_timing_dict[kb_event.scan_code] = kb_event.time
    else:
        scan_code = kb_event.scan_code
        timestamp_pairs.append((kb_event.name, event_timing_dict[scan_code], kb_event.time))


def pair_events(event_list: List[kb.KeyboardEvent]) -> List[Tuple[kb.KeyboardEvent, kb.KeyboardEvent]]:
    event_pairs = []

    # Split events into two lists: keydown & keyup
    keydown_events = [e for e in event_list if e.event_type == "down"]
    keyup_events = [e for e in event_list if e.event_type == "up"]

    # Sort events by time
    keydown_events = sorted(keydown_events, key=lambda x: x.time, reverse=False)
    keyup_events = sorted(keyup_events, key=lambda x: x.time, reverse=False)

    # Go through both lists and pair events based on event (key) name
    for press in keydown_events:
        for release in keyup_events:
            if release.name == press.name:
                event_pairs.append((press, release))
                keyup_events.remove(release)
                break

    return event_pairs

def main():
    global finished_capturing

    conf.PARTICIPANT_ID += 1
    sequence_events, dfs = [], []

    print("[INFO] Capturing keystroke data...")
    for i in range(conf.NUM_SECTIONS_TO_CAPTURE):
        print(f"[INFO] Capturing info for sequence #{i}. Press ESC when done, if necessary.")
        for j in range(conf.MAX_SEQUENCE_LENGTH * 2):
            event = kb.read_event()
            if event.name == "esc":
                print(f"[INFO] Finished typing sentence. Continuing to {i+1}")
            if handle_key_hold(event):  # If the event indicates that the key is being held down,
                j -= 1  # decrement counter
                continue  # and move on to the next event read.
            else:
                sequence_events.append(event)
        print("\n[INFO] Sequence length achieved.")
        conf.TEST_SECTION_ID += 1

        sequence_event_pairs = pair_events(sequence_events)
        dfs.append(sequence_events_to_df(sequence_event_pairs))

        sequence_events = []  # Empty list of sequence events

    for df in dfs:
        df.to_csv(conf.FILENAME, sep='\t', mode='a+', encoding='ISO-8859-1', line_terminator='\n', index=False)

    print('done')


if __name__ == "__main__":
    main()
