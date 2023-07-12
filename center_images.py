import cv2
import argparse
import os
import sys
import numpy as np


# https://github.com/Jirka-Mayer/Mashcima/blob/master/mashcima/utils.py#L30 #######################
def get_connected_components_not_touching_image_border(mask: np.ndarray) -> list[np.ndarray]:
    """
    https://github.com/Jirka-Mayer/Mashcima/blob/master/mashcima/utils.py#L30
    Takes a binary image and finds all components (areas with value 1)
    that don't touch the image border.
    """
    height, width = mask.shape
    ret, labels = cv2.connectedComponents(mask)

    indices_to_remove = set()
    for x in range(width):
        indices_to_remove.add(labels[0, x])
        indices_to_remove.add(labels[height - 1, x])
    for y in range(height):
        indices_to_remove.add(labels[y, 0])
        indices_to_remove.add(labels[y, width - 1])
    indices = set(range(1, ret)) - indices_to_remove

    out_masks: list[np.ndarray] = []
    for i in indices:
        out_masks.append(labels == i)
    return out_masks


def get_center_of_component(mask: np.ndarray) -> tuple[int, int]:
    """
    https://github.com/Jirka-Mayer/Mashcima/blob/master/mashcima/utils.py#L30
    """
    m = cv2.moments(mask.astype(np.uint8))
    if m["m00"] == 0:
        print("ERROR")
        import matplotlib.pyplot as plt
        plt.imshow(mask)
        plt.show()
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return x, y
###################################################################################################


def get_unique_non_border_almost_component(
        binary_image: np.ndarray,
        n_of_erosions: int = 10,
        kernel: np.ndarray = np.ones((3, 3), np.uint8),
        erode_first: bool = True,
) -> np.ndarray | None:
    component = None
    inner_components = get_connected_components_not_touching_image_border(binary_image)
    if len(inner_components) == 1:
        if not erode_first:
            return inner_components[0]
        else:
            component = inner_components[0]

    for _ in range(n_of_erosions):
        binary_image = cv2.erode(binary_image, kernel, iterations=1)
        inner_components = get_connected_components_not_touching_image_border(binary_image)
        if len(inner_components) == 1:
            return inner_components[0]

    return component


def center_image_to_point(
        image: np.ndarray,
        center_x: int, center_y: int,
        pad_value: int = 255
) -> np.ndarray:
    delta_x = center_x*2 - image.shape[1]
    delta_y = center_y*2 - image.shape[0]
    return cv2.copyMakeBorder(
        image,
        max(0, -delta_y),
        max(0, delta_y),
        max(0, -delta_x),
        max(0, delta_x),
        cv2.BORDER_CONSTANT, value=pad_value,
    )


def center_image_to_single_inner_component(
        image: np.ndarray,
        max_value: int = 255,
        n_of_erosions: int = 10,
        kernel: np.ndarray = np.ones((2, 2), np.uint8),
        erode_first: bool = True,
) -> np.ndarray | None:
    binary_image = np.zeros_like(image)
    binary_image[image > max_value // 2] = 1

    component = get_unique_non_border_almost_component(
        binary_image,
        n_of_erosions=n_of_erosions, kernel=kernel, erode_first=erode_first
    )
    if component is None:
        return None

    component_center_x, component_center_y = get_center_of_component(component)
    return center_image_to_point(image, component_center_x, component_center_y, pad_value=max_value)


def center_image_to_its_center(image: np.ndarray) -> np.ndarray:
    return image




def center_flat(image: np.ndarray, **kwargs) -> np.ndarray | None:
    return center_image_to_single_inner_component(image, erode_first=True, **kwargs)


def center_natural(image: np.ndarray, **kwargs) -> np.ndarray | None:
    return center_image_to_single_inner_component(image, erode_first=True, **kwargs)


def center_sharp(image: np.ndarray, **kwargs) -> np.ndarray | None:
    return center_image_to_single_inner_component(image, erode_first=False, **kwargs)


center_whole_note = center_image_to_its_center
center_quarter_rest = center_image_to_its_center
center_eighth_rest = center_image_to_its_center
center_sixteenth_rest = center_image_to_its_center
center_thirty_second_rest = center_image_to_its_center
center_c_clef = center_image_to_its_center



# GENERATE_IMAGES
def bounding_box(image, maxvalue) -> tuple[int, int, int, int]:
    binary_image = image[...] > maxvalue/2
    top = np.min(np.argmax(np.pad(binary_image, [[0, 1], [0, 0]], constant_values=True), axis=0))
    left = np.min(np.argmax(np.pad(binary_image, [[0, 0], [0, 1]], constant_values=True), axis=1))
    binary_image = binary_image[::-1, ::-1]
    bottom = image.shape[0] - np.min(np.argmax(np.pad(binary_image, [[0, 1], [0, 0]], constant_values=True), axis=0))
    right = image.shape[1] - np.min(np.argmax(np.pad(binary_image, [[0, 0], [0, 1]], constant_values=True), axis=1))
    return top, left, bottom, right


def center_g_clef(
        image: np.ndarray,
        max_value: int = 255,
) -> np.ndarray | None:
    bb = bounding_box(max_value - image, max_value)
    image = image[bb[0]:bb[2], bb[1]:bb[3]]

    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 1, param2=5, minRadius=5, maxRadius=10)

    acceptable = (circles[0, :, 1] > image.shape[0]//2) & (circles[0, :, 1] < image.shape[0]*4//5)
    if not any(acceptable):
        return None

    center_x = int(np.median(circles[0, :, 0][acceptable]))
    center_y = int(np.median(circles[0, :, 1][acceptable]))
    return center_image_to_point(image, center_x, center_y, pad_value=max_value)


def center_f_clef(
        image: np.ndarray,
        max_value: int = 255,
        kernel: np.ndarray = np.ones((2, 2), np.uint8),
) -> np.ndarray | None:
    bb = bounding_box(max_value - image, max_value)
    image = image[bb[0]:bb[2], bb[1]:bb[3]]

    baseline = image.shape[1]*3//4
    binary_image = image[:, baseline:] < max_value/2
    binary_image = cv2.erode(binary_image.astype(np.uint8), kernel, iterations=1).astype(bool)

    to_x_line = np.any(binary_image, axis=0)
    if np.all(to_x_line):
        fromm = -2
        center_x = image.shape[1] - 1
    else:
        fromm = baseline + np.min(
            np.arange(image.shape[1] - baseline)
            [np.logical_not(to_x_line)]
        )
        center_x = fromm - 1

    rightest = np.any(image[:, fromm:] < max_value/2, axis=1)
    center_y = int(np.mean(np.arange(image.shape[0])[rightest]))

    return center_image_to_point(image, center_x, center_y, pad_value=max_value)


def center_quarter_note(
        image: np.ndarray,
        max_value: int = 255,
) -> np.ndarray | None:
    bb = bounding_box(max_value - image, max_value)
    image = image[bb[0]:bb[2], bb[1]:bb[3]]
    binary_image = image < max_value/2
    binary_image = np.pad(binary_image, [[1, 1], [1, 1]], constant_values=False)
    binary_image = binary_image.astype(np.uint8)

    distances = cv2.distanceTransform(binary_image, cv2.DIST_C, cv2.DIST_MASK_3)
    max_distance = np.max(distances)
    binary_image = (distances >= max_distance - 1).astype(np.uint8)

    ret, labels = cv2.connectedComponents(binary_image)
    if ret != 2:
        return None

    component_center_x, component_center_y = get_center_of_component(labels)

    if component_center_y < image.shape[0]//2:
        image = image[::-1, :]
        component_center_y = image.shape[0] - component_center_y

    stem = np.argmin(image[0, :])
    if stem < component_center_x:
        image = image[:, ::-1]
        component_center_x = image.shape[1] - component_center_x

    return center_image_to_point(image, component_center_x, component_center_y, pad_value=max_value)
    

center_eighth_note = center_quarter_note

def center_half_note(
    image: np.ndarray,
    max_value: int = 255,
    kernel: np.ndarray = np.ones((3, 3), np.uint8),
) -> np.ndarray | None:
    bb = bounding_box(max_value - image, max_value)
    image = image[bb[0]:bb[2], bb[1]:bb[3]]
    binary_image = image > max_value/2
    binary_image = binary_image.astype(np.uint8)
    binary_image = cv2.erode(binary_image.astype(np.uint8), kernel, iterations=1)

    components = get_connected_components_not_touching_image_border(binary_image)
    if len(components) != 1:
        return None

    component_center_x, component_center_y = get_center_of_component(components[0])

    if component_center_y < image.shape[0]//2:
        image = image[::-1, :]
        component_center_y = image.shape[0] - component_center_y

    stem = np.argmin(image[0, :])
    if stem < component_center_x:
        image = image[:, ::-1]
        component_center_x = image.shape[1] - component_center_x

    return center_image_to_point(image, component_center_x, component_center_y, pad_value=max_value)


def whole_note_from_half(
    image: np.ndarray,
    max_value: int = 255,
    kernel: np.ndarray = np.ones((3, 3), np.uint8),
):
    image = center_half_note(image, max_value, kernel)
    if image is None:
        return None
    center_x = image.shape[1]//2
    center_y = image.shape[0]//2
    height = center_y
    ink = False
    while True:
        if image[height, center_x] < max_value // 2:
            ink = True
        elif ink:
            break
        height -= 1

    image[0:height+1, :] = max_value
    bb = bounding_box(max_value - image, max_value)
    if bb[2] - bb[0] > image.shape[0]//3:
        return None
    return image[bb[0]:bb[2], bb[1]:bb[3]]






def center_image(input_dirs: list[str], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for i, input_dir in enumerate(input_dirs):
        for file in os.listdir(input_dir):
            filename = os.path.join(input_dir, file)
            image = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE)
            image = center_f_clef(image)
            if image is None:
                print("skipping: " + filename, file=sys.stderr)
                continue

            cv2.imwrite(os.path.join(output_dir, f"d{i}" + file), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs="+")
    parser.add_argument("output_dir")
    # parser.add_argument("type", type=SymbolType)
    args = parser.parse_args()
    center_image(args.input_dirs, args.output_dir)
