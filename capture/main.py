from __future__ import annotations

from typing import List, Tuple

import keyboard as kb
import numpy as np
import pandas as pd

from conf import ConfManager

event_list = []
sequence_list = []
no_up_events = 0

participant_name = input("[!] Please provide your name: ")
conf_manager = ConfManager(participant_name=participant_name)
max_seq_len = conf_manager.get_max_len()  # Used to eliminate the need to call ConfManager methods in callback

def event_list_to_df(event_pairs: List[Tuple[float, float, str]], section_id: int) -> pd.DataFrame | None:
    """
    TODO: Test and tune this function
    :return:
    """
    global conf_manager
    press_times, release_times, letters = [], [], []

    for pair in event_pairs:
        press_times.append(pair[0])  # KEYDOWN timestamp
        release_times.append(pair[1])  # KEYUP timestamp
        letters.append(pair[2])  # Name of key that was typed

    df = pd.DataFrame()

    participant_id_col = pd.Series([conf_manager.get_participant_id()] * len(press_times), dtype=np.int32)
    df['PARTICIPANT_ID'] = participant_id_col
    test_section_id_col = pd.Series([section_id] * len(press_times), dtype=np.int32)
    df['TEST_SECTION_ID'] = test_section_id_col
    press_times_col = pd.Series(press_times, dtype=np.float64)
    df['PRESS_TIMES'] = press_times_col
    release_times_col = pd.Series(release_times, dtype=np.float64)
    df['RELEASE_TIMES'] = release_times_col
    letters_col = pd.Series(letters, dtype=str)
    df['LETTER'] = letters_col

    return df


def handle_key_hold(kb_event: kb.KeyboardEvent) -> bool:
    """
    Check whether the user is holding down the key associated with an event
    :param kb_event: Event to check for duplicates
    :return: True if user is holding down the key, False otherwise
    """
    global event_list
    last_event = event_list[-1]  # Get last event in the list

    if last_event.scan_code == kb_event.scan_code and last_event.event_type == kb_event.event_type:
        return True  # Event is not a duplicate of the last one
    else:
        return False  # Key is not being held down


def keypress_callback(kb_event: kb.KeyboardEvent) -> None:
    global event_list, no_up_events, max_seq_len

    if len(event_list) == 0:  # If this is the first event
        event_list.append(kb_event)  # Skip checks and just add the event to the list
        return

    if no_up_events == max_seq_len:  # If enough keyup events are captured
        print("[INFO] Sequence length achieved. Press ESC.\n")
        return

    if kb_event.event_type == "down":
        if handle_key_hold(kb_event):  # If key is being held down
            return  # Do not add event to the list
        else:
            event_list.append(kb_event)  # Otherwise, add it
    else:
        event_list.append(kb_event)
        no_up_events += 1  # Keep count of keyup events that occur


def process_sequence_events():
    global sequence_list

    sequence_timestamp_pairs = []
    sequence_timestamp_pairs_list = []

    for sequence in sequence_list:
        # Pair events together
        keydown_events = [ev for ev in sequence if ev.event_type == "down"]
        keyup_events = [ev for ev in sequence if ev.event_type == "up"]

        for up_ev in keyup_events:  # Look through KEYUP events
            for down_ev in keydown_events:
                if down_ev.scan_code == up_ev.scan_code:  # Found the corresponding KEYDOWN event
                    sequence_timestamp_pairs.append((down_ev.time, up_ev.time, up_ev.name))  # Save the timestamps of both events
                    keydown_events.remove(down_ev)  # Remove KEYDOWN event from list in order to not use it again
                    break

        sequence_timestamp_pairs = sorted(sequence_timestamp_pairs, key=lambda x: x[0], reverse=False)  # Sort sequence pairs by KEYDOWN timestamps to ensure they are ordered correctly
        sequence_timestamp_pairs_list.append(sequence_timestamp_pairs)  # Save list of pair timestamps for current sequence

        sequence_timestamp_pairs = []

    return sequence_timestamp_pairs_list


def main():
    from keyboard import hook, wait, unhook_all

    global conf_manager
    global event_list
    global sequence_list
    global no_up_events

    conf_manager.update_participant_id()  # Increment participant ID in conf file

    print("[INFO] Capturing keystroke data...")
    for i in range(conf_manager.get_no_sequences()):
        print(f"[INFO] Capturing data for sequence #{i}.")
        hook(keypress_callback)
        wait('esc')
        unhook_all()

        no_up_events = 0  # Reset counter for KEYUP events to maintain logic in callback on 2nd+ iterations
        sequence_list.append(event_list)  # Save list of events for later processing
        event_list = []  # Empty the list of events, to be populated in the next sequence

    all_sequences_pairs = process_sequence_events()
    dfs = []
    for pair_index, pairs in enumerate(all_sequences_pairs):
        dfs.append(event_list_to_df(pairs, pair_index + 1))

    final_df = pd.concat(dfs, axis='rows')
    with open(conf_manager.get_filename(), "w+") as fh:
        final_df.to_csv(fh, sep='\t', encoding='ISO-8859-1', line_terminator='\n', index=False)

    print("[INFO] Finished processing user data.")

    # TODO: Debug & find out why conf.json is not updated on program exit, when the
    #       ConfManager instance is destroyed (is ConfManager.__del__() even called?)


if __name__ == "__main__":
    main()
    # TODO: Maybe I should get an instance of ConfManager here?
