PARTICIPANT_ID = 0
TEST_SECTION_ID = 1
PARTICIPANT_NAME = 'George'
FILE_EXT = 'txt'
FILENAME = f'{PARTICIPANT_ID}.{FILE_EXT}'

NUM_SECTIONS_TO_CAPTURE = 5
MAX_SEQUENCE_LENGTH = 70


def update_participant_id():
    """
    Increment ID of participant
    :return: None
    """
    global PARTICIPANT_ID
    PARTICIPANT_ID += 1

