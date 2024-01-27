import ujson

def write_message(message, filepath):
    """
    Writes (log-)message to a file located on the given filepath.
    Parameters:
        message: The message to be written
        filepath: Location of the (log-)file the message should be written to
    """
    with open(filepath, 'a') as file:
        file.write(message + '\n')

def save_as_json(data, filepath):
    with open(filepath, 'w') as data_file:
        ujson.dump(data, data_file)

def load_json(filepath):
    data_file = open(filepath, "r")
    data = ujson.load(data_file)
    data_file.close()
    return data
    