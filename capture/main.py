from __future__ import annotations
import keyboard as kb
import numpy as np
import pandas as pd

import conf

finished_capturing = False
event_list = []
no_up_events = 0


def event_list_to_df() -> pd.DataFrame | None:
    """
    TODO: Pair up keyup and keydown events
    :return:
    """
    global event_list

    press_times, release_times = [], []

    for i, event1 in enumerate(event_list):
        if event1.event_type == "down":
            press_times.append(event1.time)
            for event2 in event_list[i:]:  # Search for a later event
                if event2.scan_code == event1.scan_code and event2.event_type == "up":  # with the same scan code
                    release_times.append(event2.time)
                    event_list.remove(event1)  # Remove the two events from the list
                    event_list.remove(event2)

    try:
        # Raise an AssertionError if the lengths of the two lists are not equal
        assert len(press_times) == len(release_times)
    except AssertionError:
        print("[ERROR] Nr of keydown timestamps & keyup timestamps does not match!")
        return None

    df = pd.DataFrame()

    participant_id_col = pd.Series(conf.PARTICIPANT_ID * len(press_times), dtype=np.int32)
    df['PARTICIPANT_ID'] = participant_id_col
    test_section_id_col = pd.Series(conf.TEST_SECTION_ID * len(press_times), dtype=np.int32)
    df['TEST_SECTION_ID'] = test_section_id_col
    press_times_col = pd.Series(press_times, dtype=np.float64)
    df['PRESS_TIMES'] = press_times_col
    release_times_col = pd.Series(release_times, dtype=np.float64)
    df['RELEASE_TIMES'] = release_times_col

    return df

def write_keystroke_data():
    """
    TODO: Later on, implement functions that make DataFrames from JSON data & call them here
    :return: None
    """
    global event_list

    with open(conf.FILENAME, "w+") as file_handle:
        df = event_list_to_df()
        if df:
            df.to_csv(file_handle, sep='\t', encoding='ISO-8859-1', line_terminator='\n', index=False)


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
    global finished_capturing, event_list, no_up_events

    if no_up_events == conf.SEQUENCE_LENGTH:  # If enough keyup events are captured
        finished_capturing = True  # Set key capture stop flag
        write_keystroke_data()  # Write user's session data to the disk
        return

    if len(event_list) == 0:  # If this is the first event
        event_list.append(kb_event)  # Skip checks and just add the event to the list
        return

    if kb_event.event_type == "down":
        if handle_key_hold(kb_event):  # If key is being held down
            return  # Do not add event to the list
        else:
            event_list.append(kb_event)  # Otherwise, add it
    else:
        event_list.append(kb_event)
        no_up_events += 1  # Keep count of keyup events that occur


def main():
    global finished_capturing
    kb.hook(keypress_callback)
    kb.wait()
    if finished_capturing:
        kb.unhook_all()


if __name__ == "__main__":
    main()
