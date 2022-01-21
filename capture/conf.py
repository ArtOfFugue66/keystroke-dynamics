CONF_FILENAME = 'conf.json'

class ConfManager:
    def __init__(self, participant_name):
        import json
        import os

        self.participant_name = participant_name

        if os.path.isfile(CONF_FILENAME) and os.path.getsize(CONF_FILENAME) > 0:  # If the file exists and is not empty
            with open(CONF_FILENAME, 'r') as fh:
                data = json.load(fh)

            self.participant_id = data['PARTICIPANT_ID']
            self.file_ext = data['FILE_EXT']
            self.num_sequences_to_capture = data['NUM_SEQUENCES_TO_CAPTURE']
            self.max_sequence_len = data['MAX_SEQUENCE_LENGTH']

            fh.close()
        else:
            if not os.path.isfile(CONF_FILENAME):  # If the file does not exist,
                fh = open(CONF_FILENAME, 'x')  # create it
                fh.close()

            self.participant_id = 0
            self.file_ext = 'txt'
            self.num_sequences_to_capture = 15
            self.max_sequence_len = 70

    # Getters & setters
    def get_filename(self):
        return f"{self.participant_id}.{self.file_ext}"

    def set_participant_name(self, participant_name):
        self.participant_name = participant_name

    def get_participant_name(self):
        return self.participant_name

    def set_no_sequences(self, num_sequences):
        self.num_sequences_to_capture = num_sequences

    def get_no_sequences(self):
        return self.num_sequences_to_capture

    def set_max_len(self, max_seq_len):
        self.max_sequence_len = max_seq_len

    def get_max_len(self):
        return self.max_sequence_len

    # No setter for participant ID since it should only be incremented by 1 on program run
    def update_participant_id(self):
        self.participant_id += 1

    def get_participant_id(self):
        return self.participant_id

    def __del__(self):
        import json

        # When the object is destroyed,
        data = {'PARTICIPANT_ID': self.participant_id,
                'PARTICIPANT_NAME': self.participant_name,
                'FILE_EXT': self.file_ext,
                'NUM_SECTIONS_TO_CAPTURE': self.num_sequences_to_capture,
                "MAX_SEQUENCE_LENGTH": self.max_sequence_len}

        with open(CONF_FILENAME, 'w') as fh:
            json.dump(data, fh)  # write its internal state to the JSON conf file

        fh.close()
