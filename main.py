import numpy as np

DIGITS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

NON_DIGIT_VALUE = 1
GEAR_VALUE = 2
GEAR_NEIGHBOUR_COUNT = 2


# Returns the contents of the given file as a string
def read_data_string(path: str) -> str:
    with open(path, "r") as file:
        return file.read()


# Converts the multi-line string to a numpy array
def string_to_list(data: str) -> np.ndarray:
    return np.array([list(row) for row in data.split("\n")])


# Sets the value at the given coordinates and its immediate neighbours to val
def set_diameter(subject: np.ndarray, coords: (int, int)) -> None:
    up = max(coords[1] - 1, 0)
    down = min(coords[1] + 1, subject.shape[1] - 1)
    left = max(coords[0] - 1, 0)
    right = min(coords[0] + 1, subject.shape[0] - 1)

    targets = {
        (left, up),
        (coords[0], up),
        (right, up),
        (left, coords[1]),
        coords,
        (right, coords[1]),
        (left, down),
        (coords[0], down),
        (right, down),
    }

    for target in targets:
        # Sets the higher value to avoid overwriting of higher priority fields
        subject[target] = 1


# Generates a mask where '1' signifies the radius of a non-digit symbol and '2'
# signifies the radius of an '*'
def generate_mask(source: np.ndarray) -> np.ndarray:
    mask = np.zeros(source.shape, dtype=int)
    for coords, value in np.ndenumerate(source):
        if value not in DIGITS and value != ".":
            set_diameter(mask, coords)
    # File for test purposes only
    np.savetxt("mask.txt", mask, delimiter="", fmt="%d")
    return mask


# Returns a list of coordinates all asterisks contained within the data
def get_asterisks(data: np.ndarray) -> (int, int):
    return set(coords for coords, value in np.ndenumerate(data) if value == "*")


# Returns the sum of all valid numbers
# todo: Split into multiple functions, utilise array comprehension
def sum_valid_numbers(data: np.ndarray) -> int:
    total = 0
    mask = generate_mask(data)
    valid = False
    number = ""
    for y, row in enumerate(data):
        if valid and number:
            total += int(number)
        valid = False
        number = ""
        for x, symbol in enumerate(row):
            if symbol in DIGITS:
                valid |= mask[y, x]
                number += symbol
            else:
                if valid and number:
                    total += int(number)
                valid = False
                number = ""
    return total


# Returns a slice of the numpy array limited to the rows that neighbour the
# value at the given coordinates and the introduced vertical offset
def get_neighbouring_rows(
        data: np.ndarray, coords: (int, int)
) -> (np.ndarray, int):
    span = [max(coords[0] - 1, 0), min(coords[0] + 2, data.shape[0])]
    return data[span[0]: span[1], :], span[0]


# Returns a slice of the numpy array limited to the fields that neighbour the
# value at the given coordinates and the introduced offset
def get_neighbouring_fields(
        data: np.ndarray, coords: (int, int)
) -> (np.ndarray, (int, int)):
    span_y = [max(coords[0] - 1, 0), min(coords[0] + 2, data.shape[0])]
    span_x = [max(coords[1] - 1, 0), min(coords[1] + 2, data.shape[1])]
    return data[span_y[0]: span_y[1], span_x[0]: span_x[1]], (
        span_y[0],
        span_x[0],
    )


# Returns the given coords offset by a given vector
def offset_coords(coords: (int, int), offset: (int, int)):
    return coords[0] + offset[0], coords[1] + offset[1]


# Returns the set of coordinates of all fields within a list with a given offset
def get_coords_set(data: np.ndarray, offset=(0, 0)) -> {(int, int)}:
    return {offset_coords(coords, offset) for coords, _ in np.ndenumerate(data)}


# Raised when a gear is deemed invalid during evaluation
class InvalidGearException(Exception):
    pass


# Returns the value of the gear, which is the product of its two valid
# neighbouring numbers. If there are fewer or more such numbers, an exeption is
# raised instead.
def evaluate_gear(data: np.ndarray, gear: (int, int)) -> int:
    mask = generate_mask(data)
    mask_slice, y_offset = get_neighbouring_rows(mask, gear)
    data_slice, _ = get_neighbouring_rows(data, gear)

    neighbour_fields, offset = get_neighbouring_fields(data, gear)
    neighbour_coords = get_coords_set(neighbour_fields, offset)

    numbers = []
    valid = False
    number = ""
    for y, row in enumerate(data_slice):
        if valid and number:
            if len(numbers) >= GEAR_NEIGHBOUR_COUNT:
                raise InvalidGearException(
                    f"Invalid gear at {gear}: Too many neighbours"
                )
            numbers.append(int(number))
        valid = False
        number = ""
        for x, symbol in enumerate(row):
            if symbol in DIGITS:
                valid |= (
                        mask_slice[y, x]
                        and offset_coords((y, x),
                                          (y_offset, 0)) in neighbour_coords
                )
                number += symbol
            else:
                if valid and number:
                    if len(numbers) >= GEAR_NEIGHBOUR_COUNT:
                        raise InvalidGearException(
                            f"Invalid gear at {gear}: Too many neighbours"
                        )
                    numbers.append(int(number))
                valid = False
                number = ""
    # Checking for edge case of number at the very end
    if valid and number:
        if len(numbers) >= GEAR_NEIGHBOUR_COUNT:
            raise InvalidGearException(
                f"Invalid gear at {gear}: Too many neighbours"
            )
        numbers.append(int(number))

    if len(numbers) < 2:
        raise InvalidGearException(
            f"Invalid gear at {gear}: Too few neighbours, {len(numbers)} < {GEAR_NEIGHBOUR_COUNT}, got {numbers[0]}"
        )

    return numbers[0] * numbers[1]


def sum_gears(data: np.ndarray) -> int:
    gears = get_asterisks(numpy_data)
    total = 0
    for gear in gears:
        try:
            total += evaluate_gear(numpy_data, gear)
        except InvalidGearException:
            pass
    return total

if __name__ == "__main__":
    data = read_data_string("data.txt")
    numpy_data = string_to_list(data)
    print(numpy_data)
    print(f"Task 1: {sum_valid_numbers(numpy_data)}")
    print(f"Task 2: {sum_gears(numpy_data)}")
